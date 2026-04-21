#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

END_ASSISTANT = "<END_ASSISTANT>"
CHAT_END = "<CHAT_END>"

DATASET_REPO = "HuggingFaceH4/ultrachat_200k"
DATASET_LICENSE = "MIT"
DATASET_CARD_URL = "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_ultrachat_corpus_v1"
SELECTED_SPLITS = (
    "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
    "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
    "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
    "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
)

MIN_ASSISTANT_TURNS = 1
MAX_ASSISTANT_TURNS = 6
MIN_USER_CHARS = 4
MAX_USER_CHARS = 2200
MIN_ASSISTANT_CHARS = 12
MAX_ASSISTANT_CHARS = 3600
MAX_TOTAL_CHARS = 12000
MAX_TOTAL_WORDS = 1800
MAX_LISTISH_LINES = 14

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
NON_ENGLISH_RE = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


@dataclass(frozen=True)
class ChatExample:
    id: str
    dialogue_group: str
    source_group: str
    source_ref: str
    category: str
    turn_count: int
    chat_text: str


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_turn_text(text: str) -> str:
    cleaned = normalize_text(text)
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    cleaned = "\n".join(lines)
    return cleaned


def looks_list_heavy(text: str) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False
    listish = 0
    for line in lines:
        lowered = line.lower()
        if lowered.startswith(("-", "*", "+")) or re.match(r"^\d+[.)]\s", lowered):
            listish += 1
    return listish >= MAX_LISTISH_LINES


def looks_code_heavy(text: str) -> bool:
    lowered = text.lower()
    if "```" in text:
        return True
    if any(token in lowered for token in ("def ", "class ", "import ", "public static", "```python", "```js", "```java", "sql ", "dockerfile")):
        return True
    code_markers = sum(text.count(marker) for marker in ("{", "}", ";", "=>", "::", "</", "/>", "[]", "()"))
    return code_markers >= 18


def looks_format_noise(text: str) -> bool:
    if text.count("|") >= 12:
        return True
    if text.count("#") >= 10:
        return True
    if text.count("---") >= 2:
        return True
    return False


def looks_pathological(text: str) -> bool:
    lowered = text.lower()
    reject_keywords = [
        "roleplay",
        "fanfiction",
        "write a story",
        "erotic",
        "sexual",
        "porn",
        "self-harm",
        "kill myself",
        "suicide",
        "murder",
        "bomb",
        "ransomware",
        "phishing",
        "jailbreak",
        "ignore previous instructions",
    ]
    return any(keyword in lowered for keyword in reject_keywords)


def looks_formal_essay(text: str) -> bool:
    lowered = text.lower()
    if len(text) > 3200:
        return True
    if any(phrase in lowered for phrase in ("comprehensive analysis", "scholarly", "case studies", "expert opinions", "bibliography")):
        return True
    return False


def looks_non_english(text: str) -> bool:
    return bool(NON_ENGLISH_RE.search(text))


def message_pair_usable(user_text: str, assistant_text: str) -> bool:
    user_chars = len(user_text)
    assistant_chars = len(assistant_text)
    if not (MIN_USER_CHARS <= user_chars <= MAX_USER_CHARS):
        return False
    if not (MIN_ASSISTANT_CHARS <= assistant_chars <= MAX_ASSISTANT_CHARS):
        return False
    if URL_RE.search(user_text) or URL_RE.search(assistant_text):
        return False
    if looks_non_english(user_text) or looks_non_english(assistant_text):
        return False
    if looks_code_heavy(user_text) or looks_code_heavy(assistant_text):
        return False
    if looks_format_noise(user_text) or looks_format_noise(assistant_text):
        return False
    if looks_pathological(user_text) or looks_pathological(assistant_text):
        return False
    if looks_formal_essay(assistant_text):
        return False
    return True


def conversation_usable(messages: list[dict]) -> bool:
    if len(messages) < 2:
        return False
    if messages[0]["role"] != "user":
        return False
    if messages[-1]["role"] != "assistant":
        return False

    assistant_turns = 0
    total_chars = 0
    total_words = 0

    for index, message in enumerate(messages):
        role = message["role"]
        text = message["content"]
        if not text:
            return False
        if index % 2 == 0 and role != "user":
            return False
        if index % 2 == 1 and role != "assistant":
            return False

        total_chars += len(text)
        total_words += len(text.split())

        if role == "assistant":
            assistant_turns += 1

    if not (MIN_ASSISTANT_TURNS <= assistant_turns <= MAX_ASSISTANT_TURNS):
        return False
    if total_chars > MAX_TOTAL_CHARS:
        return False
    if total_words > MAX_TOTAL_WORDS:
        return False
    return True


def build_example(prompt_id: str, source_ref: str, raw_messages: list[dict]) -> ChatExample | None:
    normalized = []
    for message in raw_messages:
        role = str(message.get("role", "")).strip().lower()
        if role not in ("user", "assistant"):
            return None
        content = normalize_turn_text(str(message.get("content", "")))
        if not content:
            return None
        normalized.append({"role": role, "content": content})

    if not conversation_usable(normalized):
        return None

    for index in range(0, len(normalized), 2):
        if not message_pair_usable(normalized[index]["content"], normalized[index + 1]["content"]):
            return None

    rendered = []
    for message in normalized:
        speaker = "User" if message["role"] == "user" else "Assistant"
        rendered.append(f"{speaker}: {message['content']}")
    rendered.append(END_ASSISTANT)
    rendered.append(CHAT_END)

    assistant_turns = len(normalized) // 2
    return ChatExample(
        id=f"ultrachat_{prompt_id}",
        dialogue_group=f"ultrachat_{prompt_id}",
        source_group="ultrachat_sft",
        source_ref=source_ref,
        category="general_assistant",
        turn_count=assistant_turns,
        chat_text="\n".join(rendered),
    )


def deduplicate_examples(examples: Iterable[ChatExample]) -> list[ChatExample]:
    seen: dict[str, ChatExample] = {}
    for example in examples:
        key = example.chat_text.strip().lower()
        if key not in seen:
            seen[key] = example
    deduped = list(seen.values())
    deduped.sort(key=lambda example: sha1_hex(example.id))
    return deduped


def split_examples(examples: list[ChatExample]) -> tuple[list[ChatExample], list[ChatExample], list[ChatExample]]:
    ordered = sorted(examples, key=lambda example: sha1_hex(example.dialogue_group))
    n = len(ordered)
    n_train = int(n * 0.8)
    n_validation = int(n * 0.1)
    training = ordered[:n_train]
    validation = ordered[n_train:n_train + n_validation]
    testing = ordered[n_train + n_validation:]
    return training, validation, testing


def write_split(output_dir: Path, name: str, examples: list[ChatExample]) -> None:
    text_path = output_dir / f"{name}.txt"
    with text_path.open("w", encoding="utf-8") as handle:
        if examples:
            handle.write("\n\n".join(example.chat_text for example in examples))
            handle.write("\n")

    jsonl_path = output_dir / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            row = {
                "id": example.id,
                "dialogue_group": example.dialogue_group,
                "source_group": example.source_group,
                "source_ref": example.source_ref,
                "category": example.category,
                "turn_count": example.turn_count,
                "chat_text": example.chat_text,
            }
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def count_by_source_group(examples: list[ChatExample]) -> dict[str, int]:
    counter = Counter(example.source_group for example in examples)
    return dict(sorted(counter.items()))


def count_by_category(examples: list[ChatExample]) -> dict[str, int]:
    counter = Counter(example.category for example in examples)
    return dict(sorted(counter.items()))


def download_ultrachat_snapshot(output_dir: Path) -> str:
    cache_dir = output_dir / "cache" / "huggingface_ultrachat_200k"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=list(SELECTED_SPLITS) + ["README.md"],
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )


def iter_examples(snapshot_path: Path) -> tuple[list[ChatExample], Counter]:
    examples: list[ChatExample] = []
    counters = Counter()

    for relative_path in SELECTED_SPLITS:
        file_path = snapshot_path / relative_path
        parquet = pq.ParquetFile(file_path)
        source_ref = f"{DATASET_REPO}:{relative_path}"
        for batch in parquet.iter_batches(columns=["prompt_id", "messages"], batch_size=1024):
            rows = batch.to_pylist()
            for row in rows:
                counters["rows_seen"] += 1
                prompt_id = str(row["prompt_id"])
                messages = row["messages"] or []
                example = build_example(prompt_id, source_ref, messages)
                if example is None:
                    counters["rows_rejected"] += 1
                    continue
                examples.append(example)
                if example.turn_count > 1:
                    counters["multi_turn_examples"] += 1
                else:
                    counters["single_turn_examples"] += 1

    return examples, counters


def build_metadata(
    output_dir: Path,
    snapshot_path: str,
    filtered_examples: list[ChatExample],
    training: list[ChatExample],
    validation: list[ChatExample],
    testing: list[ChatExample],
    counters: Counter,
) -> dict:
    return {
        "dataset_name": "tiny_chatbot_ultrachat_corpus_v1",
        "intended_scope": "First serious UltraChat-based conversational corpus for a small GPT-style chatbot in Julia. Focused on ordinary assistant dialogue, short practical help, rewrite/clarify style exchanges, light planning, and general conversational behavior.",
        "persona": "Small, polite, plainspoken assistant with short to medium helpful replies.",
        "source_dataset": {
            "name": DATASET_REPO,
            "license": DATASET_LICENSE,
            "dataset_card_url": DATASET_CARD_URL,
            "selected_splits": ["train_sft", "test_sft"],
            "selected_files": list(SELECTED_SPLITS),
            "snapshot_path": snapshot_path,
        },
        "format_policy": {
            "rendered_chat_format": "User: ...\\nAssistant: ...\\n<END_ASSISTANT>\\n<CHAT_END>",
            "assistant_end_marker": END_ASSISTANT,
            "chat_end_marker": CHAT_END,
            "multi_turn_policy": "Multi-turn dialogues are preserved in-order, normalized to the repo chat format, and required to end on an assistant turn.",
            "local_anchor_policy": "No local curated anchor is mixed into this corpus. UltraChat is the single effective source.",
        },
        "split_policy": {
            "method": "Deterministic stable SHA1 ordering by dialogue_group, then fixed 80/10/10 split.",
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
        },
        "filtering_rules": [
            "use only UltraChat train_sft and test_sft files for the first serious supervised chat corpus",
            "keep only user/assistant dialogues that start with User and end with Assistant",
            "keep 1 to 6 assistant turns per example so the corpus remains suitable for a small chatbot context window while preserving ordinary multi-turn behavior",
            "reject extremely short, extremely long, or obviously malformed turns",
            "reject URL-heavy, markdown-heavy, table-heavy, or formatting-heavy samples",
            "reject code-dump-heavy samples and conversations dominated by programming syntax",
            "reject obviously pathological or off-target roleplay, erotic, self-harm, violence, illegal-help, hateful, or prompt-injection-style prompts",
            "do not overfilter toward a tiny handcrafted corpus; keep broad ordinary assistant behavior, including practical and factual dialogues",
        ],
        "dataset_counts": {
            "total": len(filtered_examples),
            "training": len(training),
            "validation": len(validation),
            "testing": len(testing),
        },
        "counts_by_source_group": count_by_source_group(filtered_examples),
        "counts_by_category": count_by_category(filtered_examples),
        "multi_turn_count": sum(1 for example in filtered_examples if example.turn_count > 1),
        "average_turns_per_example": round(sum(example.turn_count for example in filtered_examples) / max(len(filtered_examples), 1), 2),
        "prep_counters": dict(counters),
        "quality_bar": [
            "materially larger than the OASST1-based real corpus",
            "clearly conversational rather than docs-shaped",
            "large enough to leave the tiny-corpus regime",
            "still filtered enough to describe the tone and failure modes honestly",
        ],
    }


def main(argv: list[str]) -> int:
    output_dir = DEFAULT_OUTPUT_DIR if len(argv) < 2 else Path(argv[1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = download_ultrachat_snapshot(output_dir)
    examples, counters = iter_examples(Path(snapshot_path))
    deduped = deduplicate_examples(examples)
    training, validation, testing = split_examples(deduped)

    metadata = build_metadata(output_dir, snapshot_path, deduped, training, validation, testing, counters)

    write_split(output_dir, "training", training)
    write_split(output_dir, "validation", validation)
    write_split(output_dir, "testing", testing)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)

    print(f"Prepared dataset at: {output_dir}")
    print(f"Total examples: {len(deduped)}")
    print(f"Training/validation/testing: {len(training)}/{len(validation)}/{len(testing)}")
    print(f"Multi-turn examples: {metadata['multi_turn_count']}")
    print(f"Average turns/example: {metadata['average_turns_per_example']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

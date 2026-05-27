#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


END_ASSISTANT = "<END_ASSISTANT>"
CHAT_END = "<CHAT_END>"

DATASET_REPO = "HuggingFaceH4/ultrachat_200k"
DATASET_LICENSE = "MIT"
DATASET_CARD_URL = "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_clean_sft_corpus_v1"
SELECTED_SPLITS = (
    "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
    "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
    "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
    "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
)

MIN_USER_WORDS = 2
MAX_USER_WORDS = 120
MIN_ASSISTANT_WORDS = 6
MAX_ASSISTANT_WORDS = 180
MAX_USER_CHARS = 800
MAX_ASSISTANT_CHARS = 1400
MAX_PROMPT_CHARS = 2600
MAX_LINES = 10
MAX_LISTISH_LINES = 5
MAX_ASSISTANT_TURN_INDEX = 2

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
NON_ENGLISH_RE = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
PLACEHOLDER_RE = re.compile(
    r"\[(?:name|address|email|phone|date|company|insert|recipient|sender|city|state|country|link|url)[^\]]*\]",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9']+")

PROMPT_REJECT_PHRASES = (
    "write a comprehensive",
    "write an essay",
    "write a blog post",
    "at least 1000 words",
    "at least 800 words",
    "scholarly",
    "cite sources",
    "case study",
    "case studies",
    "bibliography",
    "according to:",
    "answer according to",
    "given the text",
    "given this text",
    "based on the passage",
    "based on the following",
    "summarize the following",
    "provided text",
    "source material",
    "implement a",
    "write a python",
    "write a java",
    "write code",
    "program that",
    "roleplay",
    "fanfiction",
    "write a story",
    "poem",
    "lyrics",
)

TEXT_REJECT_PHRASES = (
    "as an ai language model",
    "i do not have personal",
    "i don't have personal",
    "i cannot browse",
    "i can't browse",
    "i don't have access to current",
    "i do not have access to current",
    "comprehensive analysis",
    "in conclusion",
    "references:",
    "bibliography",
)

SAFETY_REJECT_PHRASES = (
    "self-harm",
    "kill myself",
    "suicide",
    "erotic",
    "sexual",
    "porn",
    "bomb",
    "ransomware",
    "phishing",
    "jailbreak",
    "ignore previous instructions",
)


@dataclass(frozen=True)
class CleanSFTExample:
    id: str
    dialogue_group: str
    source_group: str
    source_ref: str
    category: str
    assistant_turn_index: int
    prompt_text: str
    assistant_text: str
    target_text: str
    chat_text: str
    prompt_words: int
    assistant_words: int


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_turn_text(text: str) -> str:
    cleaned = normalize_text(text)
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    return "\n".join(lines).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def line_count(text: str) -> int:
    return len([line for line in text.split("\n") if line.strip()])


def listish_line_count(text: str) -> int:
    count = 0
    for line in text.split("\n"):
        lowered = line.strip().lower()
        if lowered.startswith(("-", "*", "+")) or re.match(r"^\d+[.)]\s", lowered):
            count += 1
    return count


def looks_non_english(text: str) -> bool:
    return bool(NON_ENGLISH_RE.search(text))


def looks_code_heavy(text: str) -> bool:
    lowered = text.lower()
    if "```" in text:
        return True
    code_keywords = (
        "def ",
        "class ",
        "import ",
        "public static",
        "```python",
        "```js",
        "```java",
        "sql ",
        "dockerfile",
        "function(",
        "const ",
        "let ",
        "var ",
    )
    if any(keyword in lowered for keyword in code_keywords):
        return True
    code_markers = sum(text.count(marker) for marker in ("{", "}", ";", "=>", "::", "</", "/>", "[]"))
    return code_markers >= 10


def looks_format_heavy(text: str) -> bool:
    if text.count("|") >= 6:
        return True
    if text.count("#") >= 5:
        return True
    if text.count("---") >= 1:
        return True
    if PLACEHOLDER_RE.search(text):
        return True
    return False


def contains_reject_phrase(text: str, phrases: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def common_text_usable(text: str) -> bool:
    if not text:
        return False
    if URL_RE.search(text):
        return False
    if looks_non_english(text):
        return False
    if looks_code_heavy(text):
        return False
    if looks_format_heavy(text):
        return False
    if contains_reject_phrase(text, SAFETY_REJECT_PHRASES):
        return False
    return True


def user_usable(text: str) -> bool:
    if not common_text_usable(text):
        return False
    if len(text) > MAX_USER_CHARS:
        return False
    words = word_count(text)
    if not (MIN_USER_WORDS <= words <= MAX_USER_WORDS):
        return False
    if line_count(text) > 4:
        return False
    if contains_reject_phrase(text, PROMPT_REJECT_PHRASES):
        return False
    return True


def assistant_usable(text: str) -> bool:
    if not common_text_usable(text):
        return False
    if len(text) > MAX_ASSISTANT_CHARS:
        return False
    words = word_count(text)
    if not (MIN_ASSISTANT_WORDS <= words <= MAX_ASSISTANT_WORDS):
        return False
    if line_count(text) > MAX_LINES:
        return False
    if listish_line_count(text) > MAX_LISTISH_LINES:
        return False
    if contains_reject_phrase(text, TEXT_REJECT_PHRASES):
        return False
    return True


def normalize_messages(raw_messages: list[dict]) -> list[dict] | None:
    if len(raw_messages) < 2:
        return None

    normalized = []
    for message in raw_messages:
        role = str(message.get("role", "")).strip().lower()
        if role not in ("user", "assistant"):
            return None
        content = normalize_turn_text(str(message.get("content", "")))
        if not content:
            return None
        normalized.append({"role": role, "content": content})

    if normalized[0]["role"] != "user":
        return None
    for index, message in enumerate(normalized):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            return None
    return normalized


def render_prompt(messages_before_assistant: list[dict]) -> str:
    rendered = []
    for message in messages_before_assistant:
        label = "User" if message["role"] == "user" else "Assistant"
        rendered.append(f"{label}: {message['content']}")
    rendered.append("Assistant:")
    return "\n".join(rendered)


def categorize_prompt(user_text: str, assistant_text: str) -> str:
    lowered = user_text.lower()
    if any(token in lowered for token in ("rewrite", "rephrase", "sound kinder", "make it warmer")):
        return "rewrite"
    if any(token in lowered for token in ("plan", "ideas", "brainstorm", "suggest")):
        return "planning"
    if any(token in lowered for token in ("overwhelmed", "stressed", "anxious", "sad")):
        return "supportive"
    if len(assistant_text.split("\n")) == 1 and word_count(assistant_text) <= 60:
        return "short_answer"
    return "general_direct"


def build_examples_for_dialogue(prompt_id: str, source_ref: str, raw_messages: list[dict]) -> list[CleanSFTExample]:
    messages = normalize_messages(raw_messages)
    if messages is None:
        return []

    examples: list[CleanSFTExample] = []
    assistant_turn_index = 0
    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        assistant_turn_index += 1
        if assistant_turn_index > MAX_ASSISTANT_TURN_INDEX:
            break

        current_user = messages[message_index - 1]["content"] if message_index > 0 else ""
        assistant_text = message["content"]
        if not user_usable(current_user) or not assistant_usable(assistant_text):
            continue

        prior_messages = messages[:message_index]
        if any(item["role"] == "user" and not user_usable(item["content"]) for item in prior_messages):
            continue
        if any(item["role"] == "assistant" and not assistant_usable(item["content"]) for item in prior_messages):
            continue

        prompt_text = render_prompt(prior_messages)
        if len(prompt_text) > MAX_PROMPT_CHARS:
            continue

        target_text = f" {assistant_text}\n{END_ASSISTANT}\n{CHAT_END}"
        chat_text = prompt_text + target_text
        example_id = f"clean_sft_{prompt_id}_turn_{assistant_turn_index}"
        examples.append(
            CleanSFTExample(
                id=example_id,
                dialogue_group=f"ultrachat_{prompt_id}",
                source_group="ultrachat_sft_clean_short",
                source_ref=source_ref,
                category=categorize_prompt(current_user, assistant_text),
                assistant_turn_index=assistant_turn_index,
                prompt_text=prompt_text,
                assistant_text=assistant_text,
                target_text=target_text,
                chat_text=chat_text,
                prompt_words=word_count(prompt_text),
                assistant_words=word_count(assistant_text),
            )
        )

    return examples


def deduplicate_examples(examples: Iterable[CleanSFTExample]) -> list[CleanSFTExample]:
    seen: dict[str, CleanSFTExample] = {}
    for example in examples:
        key = re.sub(r"\s+", " ", example.chat_text.strip().lower())
        if key not in seen:
            seen[key] = example
    deduped = list(seen.values())
    deduped.sort(key=lambda example: sha1_hex(example.id))
    return deduped


def split_examples(examples: list[CleanSFTExample]) -> tuple[list[CleanSFTExample], list[CleanSFTExample], list[CleanSFTExample]]:
    ordered = sorted(examples, key=lambda example: sha1_hex(example.dialogue_group + ":" + example.id))
    n = len(ordered)
    n_train = int(n * 0.8)
    n_validation = int(n * 0.1)
    training = ordered[:n_train]
    validation = ordered[n_train:n_train + n_validation]
    testing = ordered[n_train + n_validation:]
    return training, validation, testing


def write_split(output_dir: Path, name: str, examples: list[CleanSFTExample]) -> None:
    text_path = output_dir / f"{name}.txt"
    with text_path.open("w", encoding="utf-8") as handle:
        if examples:
            handle.write("\n\n".join(example.chat_text for example in examples))
            handle.write("\n")

    jsonl_path = output_dir / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False))
            handle.write("\n")


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


def iter_clean_examples(snapshot_path: Path) -> tuple[list[CleanSFTExample], Counter]:
    examples: list[CleanSFTExample] = []
    counters = Counter()

    for relative_path in SELECTED_SPLITS:
        file_path = snapshot_path / relative_path
        parquet = pq.ParquetFile(file_path)
        source_ref = f"{DATASET_REPO}:{relative_path}"
        for batch in parquet.iter_batches(columns=["prompt_id", "messages"], batch_size=1024):
            for row in batch.to_pylist():
                counters["rows_seen"] += 1
                prompt_id = str(row["prompt_id"])
                raw_messages = row["messages"] or []
                built_examples = build_examples_for_dialogue(prompt_id, source_ref, raw_messages)
                if not built_examples:
                    counters["rows_without_clean_examples"] += 1
                    continue
                examples.extend(built_examples)
                counters["rows_with_clean_examples"] += 1
                counters["clean_examples_before_dedup"] += len(built_examples)
                if len(built_examples) > 1:
                    counters["multi_turn_sft_examples"] += len(built_examples) - 1

    return examples, counters


def count_by_field(examples: list[CleanSFTExample], field_name: str) -> dict[str, int]:
    counter = Counter(getattr(example, field_name) for example in examples)
    return dict(sorted(counter.items()))


def build_metadata(
    output_dir: Path,
    snapshot_path: str,
    filtered_examples: list[CleanSFTExample],
    training: list[CleanSFTExample],
    validation: list[CleanSFTExample],
    testing: list[CleanSFTExample],
    counters: Counter,
    args: argparse.Namespace,
) -> dict:
    return {
        "dataset_name": "tiny_chatbot_clean_sft_corpus_v1",
        "intended_scope": "Cleaner short-answer supervised chat corpus for the v4 KeemenaLM chatbot run. It is intentionally narrower than the previous broad UltraChat stream.",
        "persona": "Small, direct, plainspoken assistant with short helpful replies.",
        "source_dataset": {
            "name": DATASET_REPO,
            "license": DATASET_LICENSE,
            "dataset_card_url": DATASET_CARD_URL,
            "selected_splits": ["train_sft", "test_sft"],
            "selected_files": list(SELECTED_SPLITS),
            "snapshot_path": snapshot_path,
        },
        "format_policy": {
            "jsonl_fields": ["prompt_text", "target_text", "assistant_text", "chat_text"],
            "prompt_format": "User/Assistant history ending in a bare `Assistant:` marker.",
            "target_format": "Assistant completion text followed by <END_ASSISTANT> and <CHAT_END>.",
            "batching_policy": "Use example-based SFT batching with assistant-only loss; do not flatten into one document stream.",
        },
        "filtering_policy": {
            "max_examples": args.max_examples,
            "max_assistant_turn_index": MAX_ASSISTANT_TURN_INDEX,
            "assistant_word_range": [MIN_ASSISTANT_WORDS, MAX_ASSISTANT_WORDS],
            "user_word_range": [MIN_USER_WORDS, MAX_USER_WORDS],
            "rejects": [
                "non-English Unicode blocks",
                "URLs",
                "code-heavy examples",
                "markdown/table/placeholder-heavy examples",
                "long essay/blog/document-QA prompts",
                "roleplay, fiction, poems, lyrics, and obvious safety/pathology prompts",
                "assistant answers with too many lines or list items",
            ],
        },
        "split_policy": {
            "method": "Deterministic SHA1 ordering by dialogue_group and example id, then fixed 80/10/10 split.",
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
        },
        "dataset_counts": {
            "total": len(filtered_examples),
            "training": len(training),
            "validation": len(validation),
            "testing": len(testing),
        },
        "counts_by_source_group": count_by_field(filtered_examples, "source_group"),
        "counts_by_category": count_by_field(filtered_examples, "category"),
        "prep_counters": dict(counters),
        "size_stats": {
            "mean_prompt_words": round(sum(example.prompt_words for example in filtered_examples) / max(len(filtered_examples), 1), 2),
            "mean_assistant_words": round(sum(example.assistant_words for example in filtered_examples) / max(len(filtered_examples), 1), 2),
            "max_prompt_words": max((example.prompt_words for example in filtered_examples), default=0),
            "max_assistant_words": max((example.assistant_words for example in filtered_examples), default=0),
        },
        "output_dir": str(output_dir),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a cleaner short-answer SFT corpus for the KeemenaLM v4 chatbot run.",
    )
    parser.add_argument(
        "output_dir_positional",
        nargs="?",
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=f"Output directory. Overrides positional value. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100_000,
        help="Maximum clean examples to keep after deterministic deduplication. Use 0 to keep all.",
    )
    args = parser.parse_args(argv)
    if args.max_examples < 0:
        parser.error("--max-examples must be >= 0")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_dir_value = args.output_dir or args.output_dir_positional or str(DEFAULT_OUTPUT_DIR)
    output_dir = Path(output_dir_value).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = download_ultrachat_snapshot(output_dir)
    examples, counters = iter_clean_examples(Path(snapshot_path))
    deduped = deduplicate_examples(examples)
    if args.max_examples > 0 and len(deduped) > args.max_examples:
        deduped = deduped[:args.max_examples]
    training, validation, testing = split_examples(deduped)

    write_split(output_dir, "training", training)
    write_split(output_dir, "validation", validation)
    write_split(output_dir, "testing", testing)

    metadata = build_metadata(output_dir, snapshot_path, deduped, training, validation, testing, counters, args)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)

    print(f"Prepared clean SFT dataset at: {output_dir}")
    print(f"Total clean examples: {len(deduped)}")
    print(f"Training/validation/testing: {len(training)}/{len(validation)}/{len(testing)}")
    print(f"Mean assistant words: {metadata['size_stats']['mean_assistant_words']}")
    print(f"Categories: {metadata['counts_by_category']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

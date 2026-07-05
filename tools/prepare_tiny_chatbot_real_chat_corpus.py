#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import prepare_tiny_chatbot_data_helpers as DATA_HELPERS
from huggingface_hub import HfApi, snapshot_download


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_real_chat_corpus"
END_ASSISTANT = "<END_ASSISTANT>"
CHAT_END = "<CHAT_END>"

PRESETS = {
    "wildchat": ["allenai/WildChat"],
    "lmsys-chat-1m": ["lmsys/lmsys-chat-1m"],
    "oasst1": ["OpenAssistant/oasst1"],
    "wildchat-oasst1": ["allenai/WildChat", "OpenAssistant/oasst1"],
}

ROLE_MAP = {
    "user": "user",
    "human": "user",
    "prompter": "user",
    "customer": "user",
    "client": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "chatbot": "assistant",
    "model": "assistant",
    "system": "system",
}

MESSAGE_LIST_KEYS = (
    "messages",
    "conversation",
    "conversations",
    "turns",
    "dialog",
    "chat",
)
ROLE_KEYS = ("role", "from", "speaker", "author")
CONTENT_KEYS = ("content", "value", "text", "message", "utterance")
PROMPT_RESPONSE_KEYS = (
    ("prompt", "response"),
    ("prompt", "completion"),
    ("question", "answer"),
    ("instruction", "output"),
    ("input", "output"),
)

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9']+")
NON_ENGLISH_RE = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
PLACEHOLDER_RE = re.compile(
    r"\[(?:name|address|email|phone|date|company|insert|recipient|sender|city|state|country|link|url)[^\]]*\]",
    re.IGNORECASE,
)
REJECT_PHRASES = (
    "as an ai language model",
    "as an ai",
    "i am an ai language model",
    "i'm an ai language model",
    "i do not have feelings",
    "i don't have feelings",
    "i do not have emotions",
    "i don't have emotions",
    "i cannot generate inappropriate",
    "i can't generate inappropriate",
    "cannot generate inappropriate content",
    "ignore previous instructions",
    "jailbreak",
    "prompt injection",
    "bypass",
    "no-rules",
    "no rules",
    "no restrictions",
    "actdan",
    "bedan",
    "simulateinternet",
    "bypassoapolicy",
    "oadhere",
    "stay in character",
    "makeupinfo",
    "self-harm",
    "suicide",
    "erotic",
    "porn",
    "ransomware",
    "phishing",
)

PROMPT_REJECT_PHRASES = (
    "write a facebook post",
    "write a blog post",
    "write a comprehensive",
    "write an essay",
    "at least 1000 words",
    "role:dan",
    "dan≠",
    "dan ≠",
    "hcgpt",
    "bypass",
    "jailbreak",
    "ignore previous instructions",
    "stay in character",
)


def load_module(script_name: str, module_name: str):
    script_path = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
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


def looks_non_english(text: str) -> bool:
    return bool(NON_ENGLISH_RE.search(text))


def looks_code_or_table_heavy(text: str) -> bool:
    if "```" in text:
        return True
    if text.count("|") >= 6:
        return True
    markers = sum(text.count(marker) for marker in ("{", "}", ";", "</", "/>", "=>", "::"))
    return markers >= 8


def contains_reject_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in REJECT_PHRASES)


def text_usable(text: str, *, min_words: int, max_words: int, max_chars: int, max_lines: int, is_prompt: bool = False) -> bool:
    if not text:
        return False
    if len(text) > max_chars:
        return False
    words = word_count(text)
    if not (min_words <= words <= max_words):
        return False
    if line_count(text) > max_lines:
        return False
    if text.count("#") >= 3:
        return False
    if URL_RE.search(text) or PLACEHOLDER_RE.search(text):
        return False
    if looks_non_english(text) or looks_code_or_table_heavy(text):
        return False
    if contains_reject_phrase(text):
        return False
    if is_prompt and any(phrase in text.lower() for phrase in PROMPT_REJECT_PHRASES):
        return False
    return True


def normalize_role(value: Any) -> str | None:
    normalized = str(value).strip().lower()
    return ROLE_MAP.get(normalized)


def message_content(message: dict[str, Any]) -> str:
    for key in CONTENT_KEYS:
        value = message.get(key)
        if value is not None:
            return normalize_turn_text(str(value))
    return ""


def message_role(message: dict[str, Any]) -> str | None:
    for key in ROLE_KEYS:
        if key in message:
            return normalize_role(message[key])
    return None


def parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_messages(raw_messages: Any) -> list[dict[str, str]] | None:
    raw_messages = parse_json_string(raw_messages)
    if isinstance(raw_messages, dict):
        extracted = extract_messages_from_row(raw_messages)
        return extracted
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        return None

    messages: list[dict[str, str]] = []
    for raw_message in raw_messages:
        raw_message = parse_json_string(raw_message)
        if not isinstance(raw_message, dict):
            continue
        role = message_role(raw_message)
        content = message_content(raw_message)
        if role == "system":
            continue
        if role not in ("user", "assistant") or not content:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = normalize_turn_text(messages[-1]["content"] + "\n" + content)
        else:
            messages.append({"role": role, "content": content})

    if len(messages) < 2 or messages[0]["role"] != "user":
        return None
    if not any(message["role"] == "assistant" for message in messages):
        return None
    return messages


def extract_messages_from_row(row: dict[str, Any]) -> list[dict[str, str]] | None:
    for key in MESSAGE_LIST_KEYS:
        if key in row:
            messages = normalize_messages(row[key])
            if messages is not None:
                return messages

    for prompt_key, response_key in PROMPT_RESPONSE_KEYS:
        if prompt_key in row and response_key in row:
            prompt = normalize_turn_text(str(row.get(prompt_key) or ""))
            response = normalize_turn_text(str(row.get(response_key) or ""))
            if prompt and response:
                return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]

    if "instruction" in row and "input" in row and "output" in row:
        instruction = normalize_turn_text(str(row.get("instruction") or ""))
        input_text = normalize_turn_text(str(row.get("input") or ""))
        output = normalize_turn_text(str(row.get("output") or ""))
        prompt = instruction if not input_text else instruction + "\n\n" + input_text
        if prompt and output:
            return [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]

    return None


def row_is_english(row: dict[str, Any]) -> bool:
    for key in ("language", "lang", "detected_language"):
        value = row.get(key)
        if value is None:
            continue
        normalized = str(value).strip().lower()
        return normalized in ("en", "eng", "english") or normalized.startswith("en_") or normalized.startswith("en-")
    return True


def render_prompt(messages_before_assistant: list[dict[str, str]]) -> str:
    rendered: list[str] = []
    for message in messages_before_assistant:
        label = "User" if message["role"] == "user" else "Assistant"
        rendered.append(f"{label}: {message['content']}")
        if message["role"] == "assistant":
            rendered.append(END_ASSISTANT)
            rendered.append(CHAT_END)
    rendered.append("Assistant:")
    return "\n".join(rendered)


def categorize_prompt(user_text: str, assistant_text: str) -> str:
    lowered = user_text.lower()
    if any(token in lowered for token in ("hello", "hi ", "hey", "good morning")):
        return "greeting"
    if any(token in lowered for token in ("rewrite", "rephrase", "make this", "sound kinder", "polite")):
        return "rewrite"
    if any(token in lowered for token in ("plan", "ideas", "brainstorm", "suggest")):
        return "planning"
    if any(token in lowered for token in ("what is", "who is", "where is", "capital", "how many")):
        return "factual"
    if len(assistant_text.split("\n")) == 1 and word_count(assistant_text) <= 80:
        return "short_answer"
    return "general_chat"


def build_examples_for_messages(
    messages: list[dict[str, str]],
    *,
    row_id: str,
    source_group: str,
    source_ref: str,
    args: argparse.Namespace,
) -> list:
    examples = []
    assistant_turn_index = 0
    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        assistant_turn_index += 1
        if args.max_assistant_turn_index > 0 and assistant_turn_index > args.max_assistant_turn_index:
            break
        if message_index == 0 or messages[message_index - 1]["role"] != "user":
            continue

        current_user = messages[message_index - 1]["content"]
        assistant_text = message["content"]
        if not text_usable(
            current_user,
            min_words=args.min_user_words,
            max_words=args.max_user_words,
            max_chars=args.max_user_chars,
            max_lines=args.max_user_lines,
            is_prompt=True,
        ):
            continue
        if not text_usable(
            assistant_text,
            min_words=args.min_assistant_words,
            max_words=args.max_assistant_words,
            max_chars=args.max_assistant_chars,
            max_lines=args.max_assistant_lines,
        ):
            continue

        prior_messages = messages[:message_index]
        if any(
            item["role"] == "user"
            and not text_usable(
                item["content"],
                min_words=args.min_user_words,
                max_words=args.max_user_words,
                max_chars=args.max_user_chars,
                max_lines=args.max_user_lines,
                is_prompt=True,
            )
            for item in prior_messages
        ):
            continue
        if any(
            item["role"] == "assistant"
            and not text_usable(
                item["content"],
                min_words=args.min_assistant_words,
                max_words=args.max_assistant_words,
                max_chars=args.max_assistant_chars,
                max_lines=args.max_assistant_lines,
            )
            for item in prior_messages
        ):
            continue

        prompt_text = render_prompt(prior_messages)
        if len(prompt_text) > args.max_prompt_chars:
            continue
        target_text = f" {assistant_text}\n{END_ASSISTANT}\n{CHAT_END}"
        chat_text = prompt_text + target_text
        example_id = f"real_chat_{source_group}_{sha1_hex(row_id + ':' + str(assistant_turn_index) + ':' + chat_text)[:24]}"
        examples.append(
            DATA_HELPERS.SFTExample(
                id=example_id,
                dialogue_group=f"{source_group}_{row_id}",
                source_group=source_group,
                source_ref=source_ref,
                category=categorize_prompt(current_user, assistant_text),
                prompt_text=prompt_text,
                assistant_text=assistant_text,
                target_text=target_text,
                chat_text=chat_text,
                prompt_words=word_count(prompt_text),
                assistant_words=word_count(assistant_text),
            )
        )
    return examples


def is_data_file(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((".parquet", ".jsonl", ".jsonl.gz", ".json"))


def iter_rows_from_file(path: Path) -> Iterable[dict[str, Any]]:
    lowered = path.name.lower()
    if lowered.endswith(".parquet"):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if isinstance(row, dict):
                    yield row
    elif lowered.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    row = json.loads(stripped)
                    if isinstance(row, dict):
                        yield row
    elif lowered.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    row = json.loads(stripped)
                    if isinstance(row, dict):
                        yield row
    elif lowered.endswith(".json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
        elif isinstance(payload, dict):
            for key in ("data", "rows", "examples"):
                value = payload.get(key)
                if isinstance(value, list):
                    for row in value:
                        if isinstance(row, dict):
                            yield row
                    return
            yield payload


def source_group_for_path(path: Path, source_prefix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").lower()
    return f"{source_prefix}_{stem}"[:80]


def collect_oasst_examples(rows: list[dict[str, Any]], source_group: str, source_ref: str, args: argparse.Namespace) -> tuple[list, Counter]:
    counters = Counter()
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row_is_english(row):
            counters["oasst_non_english_rows"] += 1
            continue
        message_id = row.get("message_id")
        if message_id is None:
            continue
        by_id[str(message_id)] = row

    examples = []
    for message_id, row in by_id.items():
        role = normalize_role(row.get("role"))
        if role != "assistant":
            continue
        chain = []
        current = row
        seen = set()
        while current is not None:
            current_id = str(current.get("message_id"))
            if current_id in seen:
                break
            seen.add(current_id)
            chain.append(current)
            parent_id = current.get("parent_id")
            current = by_id.get(str(parent_id)) if parent_id is not None else None
        chain.reverse()
        raw_messages = [
            {
                "role": normalize_role(item.get("role")) or "",
                "content": normalize_turn_text(str(item.get("text") or "")),
            }
            for item in chain
        ]
        messages = normalize_messages(raw_messages)
        if messages is None:
            counters["oasst_bad_paths"] += 1
            continue
        built = build_examples_for_messages(
            messages,
            row_id=message_id,
            source_group=source_group,
            source_ref=source_ref,
            args=args,
        )
        examples.extend(built)
        counters["oasst_paths_kept"] += len(built)
    return examples, counters


def looks_like_oasst_rows(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    sample = rows[: min(20, len(rows))]
    tree_rows = sum(1 for row in sample if {"message_id", "parent_id", "role", "text"}.issubset(row.keys()))
    return tree_rows > 0 and tree_rows == len(sample)


def collect_examples_from_file(path: Path, source_group: str, args: argparse.Namespace) -> tuple[list, Counter]:
    rows = list(iter_rows_from_file(path))
    counters = Counter()
    counters["rows_seen"] = len(rows)
    if looks_like_oasst_rows(rows):
        examples, oasst_counters = collect_oasst_examples(rows, source_group, str(path), args)
        counters.update(oasst_counters)
        counters["examples_kept"] = len(examples)
        return examples, counters

    examples = []
    for row_index, row in enumerate(rows):
        if not row_is_english(row):
            counters["non_english_rows"] += 1
            continue
        messages = extract_messages_from_row(row)
        if messages is None:
            counters["rows_without_messages"] += 1
            continue
        row_id = str(row.get("id") or row.get("conversation_id") or row.get("conversation_hash") or row_index)
        built = build_examples_for_messages(
            messages,
            row_id=row_id,
            source_group=source_group,
            source_ref=str(path),
            args=args,
        )
        if not built:
            counters["rows_filtered"] += 1
            continue
        examples.extend(built)
        counters["rows_kept"] += 1
        counters["examples_kept"] += len(built)
    return examples, counters


def hf_dataset_files(repo_id: str, output_dir: Path, max_files: int) -> list[Path]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    candidates = sorted(path for path in files if is_data_file(path))
    if max_files > 0:
        candidates = candidates[:max_files]
    if not candidates:
        raise RuntimeError(f"no supported data files found in dataset repo {repo_id}")
    local_dir = output_dir / "cache" / re.sub(r"[^A-Za-z0-9]+", "_", repo_id).strip("_").lower()
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=candidates + ["README.md"],
        local_dir=local_dir,
    )
    return [Path(snapshot_path) / relative_path for relative_path in candidates]


def expand_presets(presets: list[str]) -> list[str]:
    repos = []
    for preset in presets:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset}; expected one of {sorted(PRESETS)}")
        repos.extend(PRESETS[preset])
    return repos


def deduplicate_examples(examples: list) -> list:
    seen: dict[str, Any] = {}
    for example in examples:
        key = re.sub(r"\s+", " ", example.chat_text.strip().lower())
        if key not in seen:
            seen[key] = example
    return sorted(seen.values(), key=lambda example: sha1_hex(example.id))


def cap_examples(examples: list, args: argparse.Namespace) -> list:
    if args.max_examples_per_category <= 0 and args.max_sft_examples <= 0:
        return examples
    category_counts = Counter()
    capped = []
    for example in sorted(
        examples,
        key=lambda item: (
            0 if item.source_group == "synthetic_v8_direct_answer" else 1,
            sha1_hex(item.id),
        ),
    ):
        if args.max_examples_per_category > 0 and category_counts[example.category] >= args.max_examples_per_category:
            continue
        capped.append(example)
        category_counts[example.category] += 1
        if args.max_sft_examples > 0 and len(capped) >= args.max_sft_examples:
            break
    return capped


def split_examples(examples: list, train_fraction: float, validation_fraction: float) -> tuple[list, list, list]:
    ordered = sorted(examples, key=lambda example: sha1_hex(example.dialogue_group + ":" + example.id))
    training = []
    validation = []
    testing = []
    validation_cutoff = int(validation_fraction * 1000)
    testing_cutoff = int((validation_fraction + (1.0 - train_fraction - validation_fraction)) * 1000)
    for example in ordered:
        bucket = int(sha1_hex(example.dialogue_group)[:12], 16) % 1000
        if bucket < validation_cutoff:
            validation.append(example)
        elif bucket < testing_cutoff:
            testing.append(example)
        else:
            training.append(example)
    if not validation and len(training) >= 2:
        validation.append(training.pop())
    if not testing and len(training) >= 2:
        testing.append(training.pop())
    return training, validation, testing


def write_sft_split(root: Path, name: str, examples: list) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False))
            handle.write("\n")
    with (root / f"{name}.txt").open("w", encoding="utf-8") as handle:
        if examples:
            handle.write("\n\n".join(example.chat_text for example in examples))
            handle.write("\n")


def write_text_split(root: Path, name: str, documents: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{name}.txt").open("w", encoding="utf-8") as handle:
        if documents:
            handle.write("\n\n".join(document.strip() for document in documents if document.strip()))
            handle.write("\n")


def tiny_local_pretrain_documents() -> list[str]:
    return [
        "A helpful assistant answers the user directly and then stops.",
        "Chat data is formatted as user and assistant turns before language model training.",
        "Simple facts and short replies are useful for testing a small chatbot.",
        "The assistant should not invent a new user turn inside its answer.",
        "A conversation can include a user question followed by a concise assistant response.",
    ] * 30


def write_tiny_local_pretrain(output_dir: Path) -> tuple[Counter, dict]:
    docs = tiny_local_pretrain_documents()
    training = [doc for index, doc in enumerate(docs) if index % 10 not in (0, 1)]
    validation = [doc for index, doc in enumerate(docs) if index % 10 == 0]
    testing = [doc for index, doc in enumerate(docs) if index % 10 == 1]
    write_text_split(output_dir / "pretrain", "training", training)
    write_text_split(output_dir / "pretrain", "validation", validation)
    write_text_split(output_dir / "pretrain", "testing", testing)
    counters = Counter(
        {
            "training_documents": len(training),
            "validation_documents": len(validation),
            "testing_documents": len(testing),
            "pretrain_documents": len(docs),
            "pretrain_characters": sum(len(doc) + 2 for doc in docs),
        }
    )
    return counters, {"tiny_local": ["built_in_real_chat_importer_smoke_documents"]}


def link_or_copy_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        destination_path.unlink()
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def count_pretrain_documents(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len([chunk for chunk in re.split(r"\n\s*\n", text) if chunk.strip()])


def write_existing_pretrain(args: argparse.Namespace, output_dir: Path) -> tuple[Counter, dict]:
    source_dir = Path(args.existing_pretrain_dir).resolve()
    if not source_dir.is_dir():
        raise ValueError(f"--existing-pretrain-dir is not a directory: {source_dir}")
    split_paths = {name: source_dir / f"{name}.txt" for name in ("training", "validation", "testing")}
    for split_name, split_path in split_paths.items():
        if not split_path.is_file():
            raise ValueError(f"existing pretrain split is missing {split_name}.txt: {split_path}")

    output_pretrain_dir = output_dir / "pretrain"
    counters = Counter()
    total_characters = 0
    for split_name, source_path in split_paths.items():
        destination_path = output_pretrain_dir / f"{split_name}.txt"
        link_or_copy_file(source_path, destination_path)
        counters[f"{split_name}_documents"] = count_pretrain_documents(source_path)
        total_characters += source_path.stat().st_size
    counters["pretrain_documents"] = counters["training_documents"] + counters["validation_documents"] + counters["testing_documents"]
    counters["pretrain_characters"] = total_characters
    return counters, {"existing_pretrain_dir": [str(source_dir)]}


def write_pretrain(args: argparse.Namespace, output_dir: Path) -> tuple[Counter, dict]:
    if args.pretrain_source == "tiny_local":
        return write_tiny_local_pretrain(output_dir)
    if args.pretrain_source == "existing":
        return write_existing_pretrain(args, output_dir)
    if args.pretrain_source == "downloaded":
        return DATA_HELPERS.collect_pretrain_to_splits(args, output_dir)
    raise ValueError(f"unsupported pretrain source {args.pretrain_source}")


def collect_anchor_examples(args: argparse.Namespace) -> tuple[list, Counter]:
    if args.anchor_examples <= 0:
        return [], Counter()
    v8 = load_module("prepare_tiny_chatbot_v8_direct_answer_corpus.py", "prepare_tiny_chatbot_v8_direct_answer_corpus")
    examples, counters = v8.generate_direct_sft_examples(args.anchor_examples, args.seed + 101)
    return examples, counters


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a generic real-chat SFT corpus for KeemenaLM v8/v9 style training.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS), default=[])
    parser.add_argument("--hf-dataset", action="append", default=[], help="Hugging Face dataset repo id, e.g. allenai/WildChat.")
    parser.add_argument("--hf-max-files", type=int, default=0, help="Limit data files per HF dataset. 0 means all.")
    parser.add_argument("--source-file", action="append", default=[], help="Local JSON/JSONL/JSONL.GZ/Parquet file to import.")
    parser.add_argument("--source-name", default="local_real_chat")
    parser.add_argument("--pretrain-source", choices=("downloaded", "tiny_local", "existing"), default="downloaded")
    parser.add_argument("--existing-pretrain-dir", default="", help="Existing pretrain split directory with training.txt/validation.txt/testing.txt. Used with --pretrain-source existing.")
    parser.add_argument("--pretrain-max-characters", type=int, default=200_000_000)
    parser.add_argument("--tinystories-train-files", type=int, default=4)
    parser.add_argument("--fineweb-files", type=int, default=8)
    parser.add_argument("--max-sft-examples", type=int, default=200_000)
    parser.add_argument("--max-examples-per-category", type=int, default=0)
    parser.add_argument("--anchor-examples", type=int, default=0)
    parser.add_argument("--max-assistant-turn-index", type=int, default=1)
    parser.add_argument("--min-user-words", type=int, default=1)
    parser.add_argument("--max-user-words", type=int, default=100)
    parser.add_argument("--min-assistant-words", type=int, default=1)
    parser.add_argument("--max-assistant-words", type=int, default=100)
    parser.add_argument("--max-user-chars", type=int, default=800)
    parser.add_argument("--max-assistant-chars", type=int, default=900)
    parser.add_argument("--max-prompt-chars", type=int, default=1400)
    parser.add_argument("--max-user-lines", type=int, default=4)
    parser.add_argument("--max-assistant-lines", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.96)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args(argv)

    if not args.preset and not args.hf_dataset and not args.source_file:
        parser.error("provide at least one --preset, --hf-dataset, or --source-file")
    for name in (
        "hf_max_files",
        "pretrain_max_characters",
        "tinystories_train_files",
        "fineweb_files",
        "max_sft_examples",
        "max_examples_per_category",
        "anchor_examples",
        "max_assistant_turn_index",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.pretrain_source == "existing" and not args.existing_pretrain_dir:
        parser.error("--existing-pretrain-dir is required when --pretrain-source existing")
    if not (0.0 < args.train_fraction < 1.0):
        parser.error("--train-fraction must be in (0, 1)")
    if not (0.0 <= args.validation_fraction < 1.0):
        parser.error("--validation-fraction must be in [0, 1)")
    if args.train_fraction + args.validation_fraction >= 1.0:
        parser.error("--train-fraction + --validation-fraction must be < 1")
    return args


def validate_local_source_files(source_file_args: list[str]) -> list[Path]:
    source_files = []
    for source_file_arg in source_file_args:
        source_file = Path(source_file_arg).resolve()
        if not source_file.exists():
            raise ValueError(f"--source-file does not exist: {source_file}")
        if not source_file.is_file():
            raise ValueError(f"--source-file must be a data file, not a directory: {source_file}")
        if not is_data_file(str(source_file)):
            raise ValueError(f"--source-file has unsupported extension: {source_file}")
        source_files.append(source_file)
    return source_files


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    source_files = validate_local_source_files(args.source_file)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_counters, pretrain_sources = write_pretrain(args, output_dir)
    hf_repos = [*expand_presets(args.preset), *args.hf_dataset]
    for repo_id in hf_repos:
        source_files.extend(hf_dataset_files(repo_id, output_dir, args.hf_max_files))

    all_examples = []
    source_counters = {}
    for source_file in source_files:
        source_group = source_group_for_path(source_file, args.source_name)
        examples, counters = collect_examples_from_file(source_file, source_group, args)
        all_examples.extend(examples)
        source_counters[str(source_file)] = dict(counters)

    anchor_examples, anchor_counters = collect_anchor_examples(args)
    all_examples.extend(anchor_examples)
    all_examples = cap_examples(deduplicate_examples(all_examples), args)
    if not all_examples:
        raise RuntimeError("real-chat importer did not produce any SFT examples")

    training, validation, testing = split_examples(all_examples, args.train_fraction, args.validation_fraction)
    write_sft_split(output_dir / "sft", "training", training)
    write_sft_split(output_dir / "sft", "validation", validation)
    write_sft_split(output_dir / "sft", "testing", testing)

    source_counts = Counter(example.source_group for example in all_examples)
    category_counts = Counter(example.category for example in all_examples)
    metadata = {
        "dataset_name": "tiny_chatbot_real_chat_corpus",
        "intended_scope": "Generic real-chat SFT corpus adapter for KeemenaLM. Converts common messages/conversations/prompt-response schemas to assistant-only target examples.",
        "sources": {
            "presets": args.preset,
            "hf_datasets": hf_repos,
            "source_files": [str(path) for path in source_files],
            "pretrain_source": args.pretrain_source,
            "pretrain_source_files": pretrain_sources,
        },
        "args": vars(args),
        "counts": {
            "pretrain": {
                "training": int(pretrain_counters["training_documents"]),
                "validation": int(pretrain_counters["validation_documents"]),
                "testing": int(pretrain_counters["testing_documents"]),
                "characters": int(pretrain_counters["pretrain_characters"]),
            },
            "sft": {
                "training": len(training),
                "validation": len(validation),
                "testing": len(testing),
                "total": len(all_examples),
                "source_groups": dict(source_counts),
                "categories": dict(category_counts),
            },
        },
        "counters": {
            "pretrain": dict(pretrain_counters),
            "sources": source_counters,
            "anchor": dict(anchor_counters),
        },
        "format_policy": {
            "input_schema": "messages/conversations/prompt-response rows",
            "output_schema": "prompt_text + target_text + assistant_text + chat_text JSONL",
            "loss_policy": "runner trains assistant target text only via existing clean SFT batching",
            "chat_template": "User:/Assistant:/<END_ASSISTANT>/<CHAT_END>",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)

    print(f"Prepared real-chat corpus at: {output_dir}")
    print(f"Pretrain docs train/val/test: {metadata['counts']['pretrain']['training']}/{metadata['counts']['pretrain']['validation']}/{metadata['counts']['pretrain']['testing']}")
    print(f"Pretrain chars: {metadata['counts']['pretrain']['characters']}")
    print(f"SFT examples train/val/test: {len(training)}/{len(validation)}/{len(testing)}")
    print(f"SFT source groups: {dict(source_counts)}")
    print(f"SFT categories: {dict(category_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

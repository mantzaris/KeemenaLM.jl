#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_v5_scratch_corpus"
DEFAULT_BASE_CLEAN_SFT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_clean_sft_corpus_v1"

TINYSTORIES_REPO = "roneneldan/TinyStories"
FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"
ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"

TINYSTORIES_TRAIN_PARQUETS = (
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
)
TINYSTORIES_VALID_PARQUET = "data/validation-00000-of-00001-869c898b519ad725.parquet"

END_ASSISTANT = "<END_ASSISTANT>"
CHAT_END = "<CHAT_END>"
WORD_RE = re.compile(r"[A-Za-z0-9']+")
NON_ENGLISH_RE = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

PRETRAIN_REJECT_PHRASES = (
    "javascript",
    "cookie policy",
    "privacy policy",
    "terms of service",
    "subscribe to",
    "click here",
    "all rights reserved",
    "function ",
    "import ",
    "public static",
)

SFT_REJECT_PHRASES = (
    "i do not have access",
    "i don't have access",
    "i do not have the capability",
    "i don't have the capability",
    "as an ai language model",
    "specific details",
    "specific information",
    "provided text",
    "given text",
    "the book",
    "the name",
    "comprehensive",
    "in conclusion",
    "references:",
)

SFT_KEEP_HINTS = (
    "rewrite",
    "rephrase",
    "sound kinder",
    "make it warmer",
    "plan",
    "ideas",
    "brainstorm",
    "suggest",
    "overwhelmed",
    "stressed",
    "anxious",
    "sad",
    "follow-up",
    "follow up",
    "message",
    "email",
    "thanks",
    "hello",
    "hi",
    "dinner",
    "weekend",
    "friend",
)


@dataclass(frozen=True)
class SFTExample:
    id: str
    dialogue_group: str
    source_group: str
    source_ref: str
    category: str
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


def paragraphize(text: str) -> list[str]:
    normalized = normalize_text(text)
    chunks = re.split(r"\n\s*\n", normalized)
    return [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks if chunk.strip()]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def looks_non_english(text: str) -> bool:
    return bool(NON_ENGLISH_RE.search(text))


def looks_code_or_table_heavy(text: str) -> bool:
    if "```" in text:
        return True
    if text.count("|") >= 6:
        return True
    markers = sum(text.count(marker) for marker in ("{", "}", ";", "</", "/>", "=>", "::"))
    return markers >= 8


def usable_pretrain_document(text: str) -> bool:
    words = word_count(text)
    if not (40 <= words <= 900):
        return False
    if URL_RE.search(text):
        return False
    if looks_non_english(text) or looks_code_or_table_heavy(text):
        return False
    if contains_any(text, PRETRAIN_REJECT_PHRASES):
        return False
    alpha_count = sum(character.isalpha() for character in text)
    if alpha_count / max(len(text), 1) < 0.55:
        return False
    return True


def first_string_column(parquet_file: pq.ParquetFile) -> str:
    names = parquet_file.schema_arrow.names
    for preferred in ("text", "story", "content", "document"):
        if preferred in names:
            return preferred
    for field in parquet_file.schema_arrow:
        if str(field.type) in ("string", "large_string"):
            return field.name
    raise ValueError("could not find a string column in parquet file")


def iter_parquet_texts(path: Path, batch_size: int = 1024) -> Iterable[str]:
    parquet_file = pq.ParquetFile(path)
    column = first_string_column(parquet_file)
    for batch in parquet_file.iter_batches(columns=[column], batch_size=batch_size):
        for row in batch.to_pylist():
            value = row.get(column)
            if value is None:
                continue
            yield str(value)


def download_tinystories(output_dir: Path, train_files: int) -> list[Path]:
    selected_train = list(TINYSTORIES_TRAIN_PARQUETS[:max(train_files, 0)])
    selected = selected_train + [TINYSTORIES_VALID_PARQUET, "README.md"]
    local_dir = output_dir / "cache" / "tinystories"
    snapshot_path = snapshot_download(
        repo_id=TINYSTORIES_REPO,
        repo_type="dataset",
        allow_patterns=selected,
        local_dir=local_dir,
    )
    return [Path(snapshot_path) / relative_path for relative_path in selected_train]


def fineweb_file_candidates(max_files: int) -> list[str]:
    if max_files <= 0:
        return []
    files = HfApi().list_repo_files(FINEWEB_EDU_REPO, repo_type="dataset")
    parquet_files = sorted(path for path in files if path.startswith("data/") and path.endswith(".parquet"))
    return parquet_files[:max_files]


def download_fineweb_edu(output_dir: Path, max_files: int) -> list[Path]:
    selected = fineweb_file_candidates(max_files)
    if not selected:
        return []
    local_dir = output_dir / "cache" / "fineweb_edu"
    snapshot_path = snapshot_download(
        repo_id=FINEWEB_EDU_REPO,
        repo_type="dataset",
        allow_patterns=selected + ["README.md"],
        local_dir=local_dir,
    )
    return [Path(snapshot_path) / relative_path for relative_path in selected]


def collect_pretrain_documents(args: argparse.Namespace, output_dir: Path) -> tuple[list[str], Counter, dict]:
    counters = Counter()
    source_files: dict[str, list[str]] = {"tinystories": [], "fineweb_edu": []}
    documents: list[str] = []
    total_characters = 0
    max_characters = args.pretrain_max_characters

    def maybe_add_document(text: str, source: str) -> bool:
        nonlocal total_characters
        for paragraph in paragraphize(text):
            counters[f"{source}_paragraphs_seen"] += 1
            if not usable_pretrain_document(paragraph):
                counters[f"{source}_paragraphs_rejected"] += 1
                continue
            if max_characters > 0 and total_characters + len(paragraph) > max_characters:
                return False
            documents.append(paragraph)
            total_characters += len(paragraph) + 2
            counters[f"{source}_paragraphs_kept"] += 1
        return max_characters <= 0 or total_characters < max_characters

    for path in download_tinystories(output_dir, args.tinystories_train_files):
        source_files["tinystories"].append(str(path))
        for text in iter_parquet_texts(path):
            if not maybe_add_document(text, "tinystories"):
                break
        if max_characters > 0 and total_characters >= max_characters:
            break

    if max_characters <= 0 or total_characters < max_characters:
        for path in download_fineweb_edu(output_dir, args.fineweb_files):
            source_files["fineweb_edu"].append(str(path))
            for text in iter_parquet_texts(path):
                if not maybe_add_document(text, "fineweb_edu"):
                    break
            if max_characters > 0 and total_characters >= max_characters:
                break

    counters["pretrain_documents"] = len(documents)
    counters["pretrain_characters"] = total_characters
    return documents, counters, source_files


def load_clean_sft_module():
    script_path = Path(__file__).with_name("prepare_tiny_chatbot_clean_sft_corpus.py")
    spec = importlib.util.spec_from_file_location("prepare_tiny_chatbot_clean_sft_corpus", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean_sft_row_to_example(row: dict) -> SFTExample | None:
    prompt_text = normalize_text(str(row.get("prompt_text", "")))
    assistant_text = normalize_text(str(row.get("assistant_text", "")))
    target_text = str(row.get("target_text", f" {assistant_text}\n{END_ASSISTANT}\n{CHAT_END}"))
    chat_text = str(row.get("chat_text", prompt_text + target_text))
    prompt_words = word_count(prompt_text)
    assistant_words = word_count(assistant_text)

    if not prompt_text or not assistant_text:
        return None
    if not (5 <= assistant_words <= 90):
        return None
    if prompt_words > 140:
        return None
    lowered_prompt = prompt_text.lower()
    lowered_answer = assistant_text.lower()
    if contains_any(lowered_answer, SFT_REJECT_PHRASES):
        return None
    if looks_non_english(prompt_text) or looks_non_english(assistant_text):
        return None
    if looks_code_or_table_heavy(prompt_text) or looks_code_or_table_heavy(assistant_text):
        return None
    if assistant_text.count("\n") > 5:
        return None
    if not (contains_any(lowered_prompt, SFT_KEEP_HINTS) or assistant_words <= 45):
        return None

    return SFTExample(
        id=str(row.get("id", sha1_hex(chat_text))),
        dialogue_group=str(row.get("dialogue_group", row.get("id", sha1_hex(chat_text)))),
        source_group="ultrachat_sft_v5_strict",
        source_ref=str(row.get("source_ref", ULTRACHAT_REPO)),
        category=str(row.get("category", "general_direct")),
        prompt_text=prompt_text,
        assistant_text=assistant_text,
        target_text=target_text,
        chat_text=chat_text,
        prompt_words=prompt_words,
        assistant_words=assistant_words,
    )


def read_existing_clean_sft(base_dir: Path) -> list[SFTExample]:
    examples: list[SFTExample] = []
    for split_name in ("training", "validation", "testing"):
        path = base_dir / f"{split_name}.jsonl"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                example = clean_sft_row_to_example(json.loads(line))
                if example is not None:
                    examples.append(example)
    return examples


def build_clean_sft_from_ultrachat(output_dir: Path) -> list[SFTExample]:
    helper = load_clean_sft_module()
    snapshot_path = helper.download_ultrachat_snapshot(output_dir)
    raw_examples, _ = helper.iter_examples(Path(snapshot_path))
    deduped = helper.deduplicate_examples(raw_examples)
    examples: list[SFTExample] = []
    for raw_example in deduped:
        row = asdict(raw_example)
        example = clean_sft_row_to_example(row)
        if example is not None:
            examples.append(example)
    return examples


def deduplicate_sft_examples(examples: Iterable[SFTExample]) -> list[SFTExample]:
    seen: dict[str, SFTExample] = {}
    for example in examples:
        key = re.sub(r"\s+", " ", example.chat_text.strip().lower())
        if key not in seen:
            seen[key] = example
    deduped = list(seen.values())
    deduped.sort(key=lambda example: sha1_hex(example.id))
    return deduped


def collect_sft_examples(args: argparse.Namespace, output_dir: Path) -> tuple[list[SFTExample], Counter]:
    counters = Counter()
    base_dir = Path(args.base_clean_sft_dir).resolve()
    if base_dir.is_dir():
        examples = read_existing_clean_sft(base_dir)
        counters["source_existing_clean_sft"] = 1
    else:
        raise FileNotFoundError(
            f"base clean SFT directory does not exist: {base_dir}. "
            "Run tools/prepare_tiny_chatbot_clean_sft_corpus.py first, or pass --base-clean-sft-dir."
        )

    counters["strict_sft_before_dedup"] = len(examples)
    examples = deduplicate_sft_examples(examples)
    if args.max_sft_examples > 0:
        examples = examples[:args.max_sft_examples]
    counters["strict_sft_examples"] = len(examples)
    return examples, counters


def split_items(items: list, train_fraction: float = 0.98, validation_fraction: float = 0.01):
    ordered = sorted(items, key=lambda item: sha1_hex(item if isinstance(item, str) else item.id))
    n = len(ordered)
    n_train = int(n * train_fraction)
    n_validation = int(n * validation_fraction)
    return ordered[:n_train], ordered[n_train:n_train + n_validation], ordered[n_train + n_validation:]


def write_text_split(root: Path, name: str, documents: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{name}.txt").open("w", encoding="utf-8") as handle:
        if documents:
            handle.write("\n\n".join(document.strip() for document in documents if document.strip()))
            handle.write("\n")


def write_sft_split(root: Path, name: str, examples: list[SFTExample]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False))
            handle.write("\n")
    with (root / f"{name}.txt").open("w", encoding="utf-8") as handle:
        if examples:
            handle.write("\n\n".join(example.chat_text for example in examples))
            handle.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare v5 scratch pretrain + chat SFT data for KeemenaLM.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-clean-sft-dir", default=str(DEFAULT_BASE_CLEAN_SFT_DIR))
    parser.add_argument("--pretrain-max-characters", type=int, default=200_000_000)
    parser.add_argument("--tinystories-train-files", type=int, default=4)
    parser.add_argument("--fineweb-files", type=int, default=1)
    parser.add_argument("--max-sft-examples", type=int, default=40_000)
    args = parser.parse_args(argv)
    if args.pretrain_max_characters < 0:
        parser.error("--pretrain-max-characters must be >= 0")
    if args.tinystories_train_files < 0:
        parser.error("--tinystories-train-files must be >= 0")
    if args.fineweb_files < 0:
        parser.error("--fineweb-files must be >= 0")
    if args.max_sft_examples < 0:
        parser.error("--max-sft-examples must be >= 0")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_documents, pretrain_counters, pretrain_sources = collect_pretrain_documents(args, output_dir)
    sft_examples, sft_counters = collect_sft_examples(args, output_dir)

    chat_lm_documents = [example.chat_text for example in sft_examples]
    pretrain_train, pretrain_validation, pretrain_testing = split_items(pretrain_documents)
    chat_train, chat_validation, chat_testing = split_items(chat_lm_documents, 0.9, 0.05)
    sft_train, sft_validation, sft_testing = split_items(sft_examples, 0.9, 0.05)

    write_text_split(output_dir / "pretrain", "training", pretrain_train)
    write_text_split(output_dir / "pretrain", "validation", pretrain_validation)
    write_text_split(output_dir / "pretrain", "testing", pretrain_testing)
    write_text_split(output_dir / "chat_lm", "training", chat_train)
    write_text_split(output_dir / "chat_lm", "validation", chat_validation)
    write_text_split(output_dir / "chat_lm", "testing", chat_testing)
    write_sft_split(output_dir / "sft", "training", sft_train)
    write_sft_split(output_dir / "sft", "validation", sft_validation)
    write_sft_split(output_dir / "sft", "testing", sft_testing)

    metadata = {
        "dataset_name": "tiny_chatbot_v5_scratch_corpus",
        "intended_scope": "Scratch curriculum: simple/prose LM pretraining, chat-format LM continuation, strict short-answer assistant-only SFT.",
        "sources": {
            "tinystories": TINYSTORIES_REPO,
            "fineweb_edu": FINEWEB_EDU_REPO,
            "ultrachat": ULTRACHAT_REPO,
            "pretrain_source_files": pretrain_sources,
            "base_clean_sft_dir": str(Path(args.base_clean_sft_dir).resolve()),
        },
        "args": vars(args),
        "counts": {
            "pretrain": {
                "training": len(pretrain_train),
                "validation": len(pretrain_validation),
                "testing": len(pretrain_testing),
                "characters": sum(len(document) + 2 for document in pretrain_documents),
            },
            "chat_lm": {
                "training": len(chat_train),
                "validation": len(chat_validation),
                "testing": len(chat_testing),
            },
            "sft": {
                "training": len(sft_train),
                "validation": len(sft_validation),
                "testing": len(sft_testing),
            },
        },
        "counters": {
            "pretrain": dict(pretrain_counters),
            "sft": dict(sft_counters),
        },
        "sft_policy": {
            "assistant_word_range": [5, 90],
            "prompt_max_words": 140,
            "reject_phrases": list(SFT_REJECT_PHRASES),
            "keep_hints": list(SFT_KEEP_HINTS),
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)

    print(f"Prepared v5 scratch corpus at: {output_dir}")
    print(f"Pretrain docs: {len(pretrain_documents)} chars: {metadata['counts']['pretrain']['characters']}")
    print(f"Chat LM examples: {len(chat_lm_documents)}")
    print(f"Strict SFT examples: {len(sft_examples)}")
    print(f"Splits pretrain train/val/test: {len(pretrain_train)}/{len(pretrain_validation)}/{len(pretrain_testing)}")
    print(f"Splits sft train/val/test: {len(sft_train)}/{len(sft_validation)}/{len(sft_testing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

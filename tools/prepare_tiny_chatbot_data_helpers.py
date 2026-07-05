#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download


TINYSTORIES_REPO = "roneneldan/TinyStories"
FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"

TINYSTORIES_TRAIN_PARQUETS = (
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
)

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


def sha1_int(text: str) -> int:
    return int(sha1_hex(text)[:12], 16)


def normalize_text(text: str) -> str:
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def paragraphize(text: str) -> list[str]:
    normalized = normalize_text(text)
    chunks = re.split(r"\n\s*\n", normalized)
    return [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks if chunk.strip()]


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
    if len(text) < 40 or len(text) > 4000:
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in PRETRAIN_REJECT_PHRASES):
        return False
    if URL_RE.search(text) or looks_non_english(text) or looks_code_or_table_heavy(text):
        return False
    if word_count(text) < 8:
        return False
    return True


def first_string_column(parquet_file: pq.ParquetFile) -> str:
    names = parquet_file.schema_arrow.names
    for field in parquet_file.schema_arrow:
        if str(field.type) in ("string", "large_string"):
            return field.name
    raise ValueError(f"could not find a string column in parquet file with columns {names}")


def iter_parquet_texts(path: Path, batch_size: int = 1024) -> Iterable[str]:
    parquet_file = pq.ParquetFile(path)
    column = first_string_column(parquet_file)
    for batch in parquet_file.iter_batches(columns=[column], batch_size=batch_size):
        values = batch.column(0).to_pylist()
        for value in values:
            if value is not None:
                yield normalize_text(str(value))


def download_tinystories(output_dir: Path, train_files: int) -> list[Path]:
    selected = list(TINYSTORIES_TRAIN_PARQUETS[:train_files])
    snapshot_path = snapshot_download(
        repo_id=TINYSTORIES_REPO,
        repo_type="dataset",
        allow_patterns=selected,
        local_dir=output_dir / "cache" / "tinystories",
    )
    return [Path(snapshot_path) / relative_path for relative_path in selected]


def fineweb_edu_files(max_files: int) -> list[str]:
    files = HfApi().list_repo_files(FINEWEB_EDU_REPO, repo_type="dataset")
    parquet_files = sorted(path for path in files if path.startswith("data/") and path.endswith(".parquet"))
    return parquet_files[:max_files]


def download_fineweb_edu(output_dir: Path, max_files: int) -> list[Path]:
    selected = fineweb_edu_files(max_files)
    snapshot_path = snapshot_download(
        repo_id=FINEWEB_EDU_REPO,
        repo_type="dataset",
        allow_patterns=selected,
        local_dir=output_dir / "cache" / "fineweb_edu",
    )
    return [Path(snapshot_path) / relative_path for relative_path in selected]


def split_name_for_text(text: str) -> str:
    bucket = sha1_int(text) % 1000
    if bucket < 5:
        return "validation"
    if bucket < 10:
        return "testing"
    return "training"


def open_split_writers(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    return {
        "training": (root / "training.txt").open("w", encoding="utf-8"),
        "validation": (root / "validation.txt").open("w", encoding="utf-8"),
        "testing": (root / "testing.txt").open("w", encoding="utf-8"),
    }


def close_writers(writers) -> None:
    for handle in writers.values():
        handle.close()


def write_split_document(writers, split_name: str, document: str) -> None:
    handle = writers[split_name]
    handle.write(document.strip())
    handle.write("\n\n")


def collect_pretrain_to_splits(args, output_dir: Path) -> tuple[Counter, dict]:
    counters = Counter()
    source_files: dict[str, list[str]] = {"tinystories": [], "fineweb_edu": []}
    writers = open_split_writers(output_dir / "pretrain")
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
            split_name = split_name_for_text(paragraph)
            write_split_document(writers, split_name, paragraph)
            total_characters += len(paragraph) + 2
            counters[f"{source}_paragraphs_kept"] += 1
            counters[f"{split_name}_documents"] += 1
        return max_characters <= 0 or total_characters < max_characters

    try:
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
    finally:
        close_writers(writers)

    counters["pretrain_documents"] = counters["training_documents"] + counters["validation_documents"] + counters["testing_documents"]
    counters["pretrain_characters"] = total_characters
    return counters, source_files


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


def write_text_split(root: Path, name: str, documents: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{name}.txt").open("w", encoding="utf-8") as handle:
        if documents:
            handle.write("\n\n".join(document.strip() for document in documents if document.strip()))
            handle.write("\n")


def deduplicate_sft_examples(examples: Iterable[SFTExample]) -> list[SFTExample]:
    seen: dict[str, SFTExample] = {}
    for example in examples:
        key = re.sub(r"\s+", " ", example.chat_text.strip().lower())
        if key not in seen:
            seen[key] = example
    deduped = list(seen.values())
    deduped.sort(key=lambda example: sha1_hex(example.id))
    return deduped


def link_or_copy_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        destination_path.unlink()
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)

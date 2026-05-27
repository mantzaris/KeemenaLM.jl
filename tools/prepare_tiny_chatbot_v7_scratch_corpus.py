#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_v7_scratch_corpus"
DEFAULT_BASE_CLEAN_SFT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_clean_sft_corpus_v1"


def load_v6_module():
    script_path = Path(__file__).with_name("prepare_tiny_chatbot_v6_scratch_corpus.py")
    spec = importlib.util.spec_from_file_location("prepare_tiny_chatbot_v6_scratch_corpus", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V6 = load_v6_module()
V5 = V6.V5


def sha1_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


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


def collect_pretrain_to_splits(args: argparse.Namespace, output_dir: Path) -> tuple[Counter, dict]:
    counters = Counter()
    source_files: dict[str, list[str]] = {"tinystories": [], "fineweb_edu": []}
    writers = open_split_writers(output_dir / "pretrain")
    total_characters = 0
    max_characters = args.pretrain_max_characters

    def maybe_add_document(text: str, source: str) -> bool:
        nonlocal total_characters
        for paragraph in V5.paragraphize(text):
            counters[f"{source}_paragraphs_seen"] += 1
            if not V5.usable_pretrain_document(paragraph):
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
        for path in V5.download_tinystories(output_dir, args.tinystories_train_files):
            source_files["tinystories"].append(str(path))
            for text in V5.iter_parquet_texts(path):
                if not maybe_add_document(text, "tinystories"):
                    break
            if max_characters > 0 and total_characters >= max_characters:
                break

        if max_characters <= 0 or total_characters < max_characters:
            for path in V5.download_fineweb_edu(output_dir, args.fineweb_files):
                source_files["fineweb_edu"].append(str(path))
                for text in V5.iter_parquet_texts(path):
                    if not maybe_add_document(text, "fineweb_edu"):
                        break
                if max_characters > 0 and total_characters >= max_characters:
                    break
    finally:
        close_writers(writers)

    counters["pretrain_documents"] = counters["training_documents"] + counters["validation_documents"] + counters["testing_documents"]
    counters["pretrain_characters"] = total_characters
    return counters, source_files


def split_examples(examples: list) -> tuple[list, list, list]:
    ordered = sorted(examples, key=lambda example: hashlib.sha1(example.id.encode("utf-8")).hexdigest())
    training = []
    validation = []
    testing = []
    for example in ordered:
        split_name = split_name_for_text(example.chat_text)
        if split_name == "validation":
            validation.append(example)
        elif split_name == "testing":
            testing.append(example)
        else:
            training.append(example)
    return training, validation, testing


def collect_v7_sft_examples(args: argparse.Namespace, output_dir: Path) -> tuple[list, Counter]:
    counters = Counter()
    synthetic_examples, synthetic_counters = V6.generate_synthetic_sft_examples(args.synthetic_sft_examples, args.seed)
    counters.update({f"synthetic_{key}": value for key, value in synthetic_counters.items()})

    external_examples, external_counters = V6.collect_external_sft(args, output_dir)
    counters.update({f"external_{key}": value for key, value in external_counters.items()})

    examples = V5.deduplicate_sft_examples([*synthetic_examples, *external_examples])
    counters["total_sft_examples"] = len(examples)
    counters["synthetic_requested"] = args.synthetic_sft_examples
    counters["external_requested"] = args.max_external_sft_examples
    return examples, counters


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare v7 scratch corpus with larger pretraining and smaller SFT steering data.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-clean-sft-dir", default=str(DEFAULT_BASE_CLEAN_SFT_DIR))
    parser.add_argument("--pretrain-max-characters", type=int, default=1_500_000_000)
    parser.add_argument("--tinystories-train-files", type=int, default=4)
    parser.add_argument("--fineweb-files", type=int, default=8)
    parser.add_argument("--synthetic-sft-examples", type=int, default=12_000)
    parser.add_argument("--max-external-sft-examples", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args(argv)
    for name in (
        "pretrain_max_characters",
        "tinystories_train_files",
        "fineweb_files",
        "synthetic_sft_examples",
        "max_external_sft_examples",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_counters, pretrain_sources = collect_pretrain_to_splits(args, output_dir)
    sft_examples, sft_counters = collect_v7_sft_examples(args, output_dir)
    chat_lm_documents = [example.chat_text for example in sft_examples]
    chat_train, chat_validation, chat_testing = V5.split_items(chat_lm_documents, 0.96, 0.02)
    sft_train, sft_validation, sft_testing = split_examples(sft_examples)

    V5.write_text_split(output_dir / "chat_lm", "training", chat_train)
    V5.write_text_split(output_dir / "chat_lm", "validation", chat_validation)
    V5.write_text_split(output_dir / "chat_lm", "testing", chat_testing)
    V5.write_sft_split(output_dir / "sft", "training", sft_train)
    V5.write_sft_split(output_dir / "sft", "validation", sft_validation)
    V5.write_sft_split(output_dir / "sft", "testing", sft_testing)

    source_counts = Counter(example.source_group for example in sft_examples)
    metadata = {
        "dataset_name": "tiny_chatbot_v7_scratch_corpus",
        "intended_scope": "Larger general pretraining plus small SFT steering; final training should use mixed replay instead of SFT-only grinding.",
        "sources": {
            "tinystories": V5.TINYSTORIES_REPO,
            "fineweb_edu": V5.FINEWEB_EDU_REPO,
            "ultrachat": V5.ULTRACHAT_REPO,
            "pretrain_source_files": pretrain_sources,
            "base_clean_sft_dir": str(Path(args.base_clean_sft_dir).resolve()),
        },
        "args": vars(args),
        "counts": {
            "pretrain": {
                "training": int(pretrain_counters["training_documents"]),
                "validation": int(pretrain_counters["validation_documents"]),
                "testing": int(pretrain_counters["testing_documents"]),
                "characters": int(pretrain_counters["pretrain_characters"]),
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
                "source_groups": dict(source_counts),
            },
        },
        "counters": {
            "pretrain": dict(pretrain_counters),
            "sft": dict(sft_counters),
        },
        "v7_policy": {
            "pretrain": "streamed split writer; bigger general corpus than v6",
            "sft": "small steering set; not intended for isolated long SFT",
            "recommended_final_stage": "mixed replay 70% pretrain, 20% chat LM, 10% assistant-only SFT with early stopping",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)

    print(f"Prepared v7 scratch corpus at: {output_dir}")
    print(f"Pretrain docs train/val/test: {metadata['counts']['pretrain']['training']}/{metadata['counts']['pretrain']['validation']}/{metadata['counts']['pretrain']['testing']}")
    print(f"Pretrain chars: {metadata['counts']['pretrain']['characters']}")
    print(f"SFT examples train/val/test: {len(sft_train)}/{len(sft_validation)}/{len(sft_testing)}")
    print(f"SFT source groups: {dict(source_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

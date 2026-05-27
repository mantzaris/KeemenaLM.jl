#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_v6_scratch_corpus"
DEFAULT_BASE_CLEAN_SFT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_clean_sft_corpus_v1"

EXTERNAL_PROMPT_REJECT_PHRASES = (
    "according to",
    "based on the passage",
    "based on the text",
    "given the passage",
    "given the text",
    "provided passage",
    "provided text",
    "read the passage",
    "the article",
    "the book",
    "the document",
    "the following passage",
    "the following text",
    "the statement about",
    "what is the name",
    "what was the name",
    "who is",
    "who was",
)

EXTERNAL_KEEP_HINTS = (
    "brainstorm",
    "dinner",
    "email",
    "friend",
    "hello",
    "help me plan",
    "ideas",
    "make it",
    "message",
    "plan",
    "rephrase",
    "rewrite",
    "suggest",
    "weekend",
)


def load_v5_module():
    script_path = Path(__file__).with_name("prepare_tiny_chatbot_v5_scratch_corpus.py")
    spec = importlib.util.spec_from_file_location("prepare_tiny_chatbot_v5_scratch_corpus", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V5 = load_v5_module()


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return V5.normalize_text(text)


def word_count(text: str) -> int:
    return V5.word_count(text)


def contains_any(text: str, phrases) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def make_example(category: str, user_text: str, assistant_text: str, source_index: int):
    user_text = normalize_text(user_text)
    assistant_text = normalize_text(assistant_text)
    prompt_text = f"User: {user_text}\nAssistant:"
    target_text = f" {assistant_text}\n{V5.END_ASSISTANT}\n{V5.CHAT_END}"
    chat_text = f"{prompt_text} {assistant_text}\n{V5.END_ASSISTANT}\n{V5.CHAT_END}"
    example_id = f"synthetic_v6_{category}_{sha1_hex(chat_text)[:24]}_{source_index}"
    return V5.SFTExample(
        id=example_id,
        dialogue_group=example_id,
        source_group="synthetic_v6_clean_assistant",
        source_ref="tools/prepare_tiny_chatbot_v6_scratch_corpus.py",
        category=category,
        prompt_text=prompt_text,
        assistant_text=assistant_text,
        target_text=target_text,
        chat_text=chat_text,
        prompt_words=word_count(prompt_text),
        assistant_words=word_count(assistant_text),
    )


def clean_generated_example(example) -> bool:
    if not example.prompt_text or not example.assistant_text:
        return False
    if not (4 <= example.assistant_words <= 120):
        return False
    if example.prompt_words > 140:
        return False
    lowered = example.assistant_text.lower()
    if contains_any(lowered, V5.SFT_REJECT_PHRASES):
        return False
    if V5.looks_non_english(example.prompt_text) or V5.looks_non_english(example.assistant_text):
        return False
    if V5.looks_code_or_table_heavy(example.prompt_text) or V5.looks_code_or_table_heavy(example.assistant_text):
        return False
    return True


def generated_answer_for_rewrite(message: str, tone: str, rng: random.Random) -> str:
    openers = {
        "warmer": "Thanks for the update. I appreciate the work you have already put into this. Could we adjust the timing a bit so it is easier to finish well?",
        "shorter": "Thanks for the update. Could we adjust the timing so this is easier to finish well?",
        "clearer": "Thanks for the update. I want to make the timing easier to manage, so could we adjust the plan and confirm the next step?",
        "more polite": "Thank you for the update. Would it be possible to adjust the timing so we can finish this carefully?",
        "more direct": "Thanks for the update. I need us to adjust the timing and agree on the next step.",
        "friendlier": "Thanks for the update. I appreciate it. Could we tweak the timing so this feels easier to handle?",
    }
    base = openers.get(tone, openers["clearer"])
    if "late" in message.lower():
        return base + " I would rather reset expectations now than rush and miss something important."
    if "confused" in message.lower():
        return base + " I am not fully clear on one part yet, so a quick clarification would help."
    if rng.random() < 0.35:
        return base + " Please let me know what works best."
    return base


def generated_plan(goal: str, timebox: str, constraint: str) -> str:
    return (
        f"Here is a simple {timebox} plan: first, pick the smallest useful version of {goal}. "
        f"Next, spend one focused block on it and ignore anything that is not required. "
        f"Then check the result, fix the obvious issue, and stop. Keep the plan {constraint}."
    )


def generated_ideas(topic: str, constraint: str, rng: random.Random) -> str:
    ideas = [
        f"Try a low-effort version of {topic}.",
        f"Pair {topic} with something familiar so it is easier to start.",
        f"Set a short timer and treat it as an experiment.",
        f"Invite one person if company would make it more fun.",
        f"Prepare the simplest materials ahead of time.",
    ]
    rng.shuffle(ideas)
    return f"Here are a few ideas that keep it {constraint}: " + " ".join(ideas[:4])


def generated_support(situation: str, next_step: str) -> str:
    return (
        f"That sounds like a lot to carry. Start by making the situation smaller: write down only the next visible action, "
        f"then do {next_step}. You do not need to solve the whole thing at once."
    )


def generated_message(recipient: str, situation: str, tone: str) -> str:
    return (
        f"Hi, I wanted to let you know about {situation}. I appreciate your patience, and I want to handle it clearly. "
        f"Could we choose the next step that works best for both of us? Thanks."
        if tone != "direct"
        else f"Hi, I need to update you about {situation}. Please let me know the best next step so we can keep this moving."
    )


def generated_explanation(concept: str, audience: str) -> str:
    return (
        f"{concept} means breaking a bigger idea into smaller parts that are easier to reason about. "
        f"For {audience}, the key is to connect it to something familiar, give one example, and avoid extra detail until it is needed."
    )


def generated_decision(choice_a: str, choice_b: str, criterion: str) -> str:
    return (
        f"Use {criterion} as the deciding rule. Choose {choice_a} if it improves that rule right now; choose {choice_b} "
        f"if it reduces risk or gives you more time. If both look close, pick the option that is easier to reverse."
    )


def generate_synthetic_sft_examples(count: int, seed: int) -> tuple[list, Counter]:
    rng = random.Random(seed)
    counters = Counter()
    examples = []
    seen = set()

    messages = [
        "I might be late to the meeting because another task is taking longer than expected.",
        "I am confused about the next step and need someone to explain it again.",
        "The deadline is tight and I do not want to promise more than I can do.",
        "I need to cancel tonight because I am exhausted.",
        "I disagree with the plan but want to say it respectfully.",
        "I forgot to reply earlier and want to apologize without overexplaining.",
        "I need more information before I can make a decision.",
        "The draft is too long and I want it to sound simpler.",
    ]
    tones = ["warmer", "shorter", "clearer", "more polite", "more direct", "friendlier"]
    goals = [
        "clean my room", "study for an exam", "prepare dinner", "write a short report", "plan a weekend",
        "organize my files", "start exercising", "reply to emails", "finish a small project", "make a birthday plan",
        "prepare for a call", "compare two options", "make a grocery list", "practice a presentation",
    ]
    timeboxes = ["15-minute", "30-minute", "one-hour", "evening", "weekend morning"]
    constraints = ["cheap", "calm", "simple", "low-pressure", "easy to pause", "not too ambitious"]
    idea_topics = [
        "a quiet weekend", "a simple dinner", "a birthday surprise", "a rainy afternoon", "a study break",
        "a small home project", "a friend hangout", "a quick lunch", "a relaxing evening", "a no-spend day",
    ]
    situations = [
        "I feel overwhelmed by everything I need to do",
        "I am anxious about a conversation",
        "I have too many tasks and no clear starting point",
        "I feel stuck after making a mistake",
        "I am tired but still need to make progress",
        "I keep putting off a small task",
    ]
    next_steps = [
        "one five-minute cleanup", "one short note", "one easy phone reminder", "one small list",
        "one message asking for clarification", "one quiet break before continuing",
    ]
    recipients = ["my manager", "a friend", "my roommate", "a teammate", "a teacher", "a client"]
    message_situations = [
        "a delayed reply", "a schedule change", "a missed deadline", "needing clarification",
        "changing plans", "asking for more time", "following up after a meeting",
    ]
    concepts = [
        "prioritization", "a tradeoff", "a budget", "a habit", "a summary", "a deadline",
        "a rough draft", "feedback", "a backup plan", "scope",
    ]
    audiences = ["a beginner", "a busy friend", "a student", "a teammate", "someone new to the topic"]
    decisions = [
        ("cook at home", "order takeout"), ("start now", "rest first"), ("send a short reply", "wait for more details"),
        ("choose the simpler plan", "try the ambitious plan"), ("buy the basic option", "save for the better option"),
    ]
    criteria = ["energy", "time", "cost", "risk", "reversibility", "clarity"]
    greetings = [
        ("hello", "Hi. What would you like help with today?"),
        ("hi there", "Hi. I can help with planning, rewriting, ideas, or sorting through a problem."),
        ("can you help me?", "Yes. Tell me what you are working on and what kind of help you want."),
        ("I need help but I am not sure where to start.", "Start with the rough version. What is the situation, and what outcome would be useful?"),
    ]
    context_notes = [
        "I want something practical.",
        "I want a calm answer.",
        "I do not want it to sound dramatic.",
        "I only have a little energy.",
        "I want the first step to be easy.",
        "I am trying to keep this simple.",
        "I want it to feel friendly.",
        "I want to avoid overthinking it.",
        "I want something I can do today.",
        "I want the answer to be concrete.",
        "I want it to be low-pressure.",
        "I want to make a decision without spiraling.",
        "I want a useful rough version.",
        "I need help getting unstuck.",
        "I want it to be clear but not harsh.",
        "I want the smallest reasonable next step.",
        "I need a version that sounds natural.",
        "I want to keep the tone steady.",
        "I want something easy to follow.",
        "I want to make progress without making it huge.",
        "I want to avoid sounding cold.",
        "I want to keep it brief.",
        "I need a simple starting point.",
        "I want the answer to stay focused.",
    ]
    style_requests = [
        "Keep it short.",
        "Use plain language.",
        "Give me a concrete next step.",
        "Avoid a long explanation.",
        "Make it sound natural.",
        "Use a warm tone.",
        "Keep it direct.",
        "Make it easy to scan.",
        "Give me a useful first draft.",
        "Avoid generic filler.",
        "Make it practical.",
        "Use a supportive tone.",
        "Keep it under four sentences.",
        "Give me a simple version.",
        "Focus on what to do first.",
        "Make it clear and kind.",
    ]
    assistant_closers = [
        "Start with the easiest useful step.",
        "A small clear step is enough to begin.",
        "Keep the plan smaller than your first instinct.",
        "The goal is progress, not a perfect version.",
        "Make the next action visible and specific.",
        "If it still feels big, cut the first step in half.",
        "Use the version that is easiest to actually do.",
        "You can refine it after the first pass.",
        "Keep it simple enough that you will start.",
        "Choose the option that reduces friction.",
        "It is fine for the first version to be rough.",
        "Stop once the next step is clear.",
    ]

    categories = ["greeting", "rewrite", "plan", "ideas", "supportive", "message", "explain", "decision"]
    attempts = 0
    max_attempts = max(10_000, count * 8)

    def pick(items, index: int, stride: int = 1):
        return items[(index // stride) % len(items)]

    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        category = categories[(attempts - 1) % len(categories)]
        variation = (attempts - 1) // len(categories)
        context_note = pick(context_notes, variation)
        style_request = pick(style_requests, variation, len(context_notes))
        assistant_closer = pick(assistant_closers, variation, len(context_notes) * len(style_requests))

        if category == "greeting":
            greeting_user, greeting_assistant = pick(greetings, variation)
            goal = pick(goals, variation, len(greetings))
            user = f"{greeting_user} I am trying to {goal}. {style_request}"
            assistant = f"{greeting_assistant} Tell me the rough situation and what outcome would help most. {assistant_closer}"
        elif category == "rewrite":
            message = pick(messages, variation)
            tone = pick(tones, variation, len(messages))
            user = f"Rewrite this to sound {tone}: {message} {context_note} {style_request}"
            assistant = f"{generated_answer_for_rewrite(message, tone, rng)} {assistant_closer}"
        elif category == "plan":
            goal = pick(goals, variation)
            timebox = pick(timeboxes, variation, len(goals))
            constraint = pick(constraints, variation, len(goals) * len(timeboxes))
            user = f"Help me make a {timebox} plan to {goal}. Keep it {constraint}. {context_note} {style_request}"
            assistant = f"{generated_plan(goal, timebox, constraint)} {assistant_closer}"
        elif category == "ideas":
            topic = pick(idea_topics, variation)
            constraint = pick(constraints, variation, len(idea_topics))
            user = f"Give me a few ideas for {topic}. Keep them {constraint}. {context_note} {style_request}"
            assistant = f"{generated_ideas(topic, constraint, rng)} {assistant_closer}"
        elif category == "supportive":
            situation = pick(situations, variation)
            next_step = pick(next_steps, variation, len(situations))
            user = f"{situation}. What should I do first? {context_note} {style_request}"
            assistant = f"{generated_support(situation, next_step)} {assistant_closer}"
        elif category == "message":
            recipient = pick(recipients, variation)
            situation = pick(message_situations, variation, len(recipients))
            tone = pick(["polite", "direct", "warm"], variation, len(recipients) * len(message_situations))
            user = f"Draft a {tone} message to {recipient} about {situation}. {context_note} {style_request}"
            assistant = f"{generated_message(recipient, situation, tone)} {assistant_closer}"
        elif category == "explain":
            concept = pick(concepts, variation)
            audience = pick(audiences, variation, len(concepts))
            user = f"Explain {concept} to {audience} in simple words. {context_note} {style_request}"
            assistant = f"{generated_explanation(concept, audience)} {assistant_closer}"
        else:
            choice_a, choice_b = pick(decisions, variation)
            criterion = pick(criteria, variation, len(decisions))
            user = f"Should I {choice_a} or {choice_b}? I care most about {criterion}. {context_note} {style_request}"
            assistant = f"{generated_decision(choice_a, choice_b, criterion)} {assistant_closer}"

        example = make_example(category, user, assistant, attempts)
        key = re.sub(r"\s+", " ", example.chat_text.strip().lower())
        if key in seen or not clean_generated_example(example):
            continue
        seen.add(key)
        examples.append(example)
        counters[f"synthetic_{category}"] += 1

    if len(examples) < count:
        raise RuntimeError(f"only generated {len(examples)} synthetic examples after {attempts} attempts")
    counters["synthetic_examples"] = len(examples)
    return examples, counters


def external_example_is_v6_clean(example) -> bool:
    lowered_prompt = example.prompt_text.lower()
    lowered_answer = example.assistant_text.lower()
    if contains_any(lowered_prompt, EXTERNAL_PROMPT_REJECT_PHRASES):
        return False
    if contains_any(lowered_answer, V5.SFT_REJECT_PHRASES):
        return False
    if not contains_any(lowered_prompt, EXTERNAL_KEEP_HINTS):
        return False
    if example.prompt_words > 90:
        return False
    if not (5 <= example.assistant_words <= 80):
        return False
    return True


def collect_external_sft(args: argparse.Namespace, output_dir: Path) -> tuple[list, Counter]:
    counters = Counter()
    if args.max_external_sft_examples <= 0:
        return [], counters
    helper_args = argparse.Namespace(**vars(args))
    helper_args.max_sft_examples = args.max_external_sft_examples * 4
    examples, source_counters = V5.collect_sft_examples(helper_args, output_dir)
    counters.update({f"source_{key}": value for key, value in source_counters.items()})
    filtered = [example for example in examples if external_example_is_v6_clean(example)]
    filtered = V5.deduplicate_sft_examples(filtered)
    filtered = filtered[: args.max_external_sft_examples]
    counters["external_before_v6_filter"] = len(examples)
    counters["external_after_v6_filter"] = len(filtered)
    return filtered, counters


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare v6 scratch pretrain + clean assistant data for KeemenaLM.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-clean-sft-dir", default=str(DEFAULT_BASE_CLEAN_SFT_DIR))
    parser.add_argument("--pretrain-max-characters", type=int, default=750_000_000)
    parser.add_argument("--tinystories-train-files", type=int, default=4)
    parser.add_argument("--fineweb-files", type=int, default=4)
    parser.add_argument("--synthetic-sft-examples", type=int, default=180_000)
    parser.add_argument("--max-external-sft-examples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260522)
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

    pretrain_documents, pretrain_counters, pretrain_sources = V5.collect_pretrain_documents(args, output_dir)
    synthetic_examples, synthetic_counters = generate_synthetic_sft_examples(args.synthetic_sft_examples, args.seed)
    external_examples, external_counters = collect_external_sft(args, output_dir)
    sft_examples = V5.deduplicate_sft_examples([*synthetic_examples, *external_examples])

    chat_lm_documents = [example.chat_text for example in sft_examples]
    pretrain_train, pretrain_validation, pretrain_testing = V5.split_items(pretrain_documents, 0.99, 0.005)
    chat_train, chat_validation, chat_testing = V5.split_items(chat_lm_documents, 0.96, 0.02)
    sft_train, sft_validation, sft_testing = V5.split_items(sft_examples, 0.96, 0.02)

    V5.write_text_split(output_dir / "pretrain", "training", pretrain_train)
    V5.write_text_split(output_dir / "pretrain", "validation", pretrain_validation)
    V5.write_text_split(output_dir / "pretrain", "testing", pretrain_testing)
    V5.write_text_split(output_dir / "chat_lm", "training", chat_train)
    V5.write_text_split(output_dir / "chat_lm", "validation", chat_validation)
    V5.write_text_split(output_dir / "chat_lm", "testing", chat_testing)
    V5.write_sft_split(output_dir / "sft", "training", sft_train)
    V5.write_sft_split(output_dir / "sft", "validation", sft_validation)
    V5.write_sft_split(output_dir / "sft", "testing", sft_testing)

    source_counts = Counter(example.source_group for example in sft_examples)
    metadata = {
        "dataset_name": "tiny_chatbot_v6_scratch_corpus",
        "intended_scope": "Larger scratch curriculum: larger prose LM pretraining, chat-format LM continuation, and synthetic-dominant clean assistant SFT.",
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
                "source_groups": dict(source_counts),
            },
        },
        "counters": {
            "pretrain": dict(pretrain_counters),
            "synthetic_sft": dict(synthetic_counters),
            "external_sft": dict(external_counters),
        },
        "sft_policy": {
            "dominant_source": "synthetic_v6_clean_assistant",
            "external_prompt_reject_phrases": list(EXTERNAL_PROMPT_REJECT_PHRASES),
            "external_keep_hints": list(EXTERNAL_KEEP_HINTS),
            "base_reject_phrases": list(V5.SFT_REJECT_PHRASES),
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)

    print(f"Prepared v6 scratch corpus at: {output_dir}")
    print(f"Pretrain docs: {len(pretrain_documents)} chars: {metadata['counts']['pretrain']['characters']}")
    print(f"Synthetic SFT examples: {len(synthetic_examples)}")
    print(f"External SFT examples kept: {len(external_examples)}")
    print(f"Total clean SFT examples: {len(sft_examples)}")
    print(f"Splits pretrain train/val/test: {len(pretrain_train)}/{len(pretrain_validation)}/{len(pretrain_testing)}")
    print(f"Splits sft train/val/test: {len(sft_train)}/{len(sft_validation)}/{len(sft_testing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

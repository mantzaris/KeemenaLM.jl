#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import prepare_tiny_chatbot_data_helpers as DATA_HELPERS


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "tiny_chatbot_v8_direct_answer_corpus"


CHAT_END = DATA_HELPERS.CHAT_END
END_ASSISTANT = DATA_HELPERS.END_ASSISTANT


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def make_example(category: str, user_text: str, assistant_text: str, source_index: int, prior_turns: list[tuple[str, str]] | None = None):
    user_text = normalize_text(user_text)
    assistant_text = normalize_text(assistant_text)
    rendered = []
    if prior_turns:
        for user_turn, assistant_turn in prior_turns:
            rendered.append(f"User: {normalize_text(user_turn)}")
            rendered.append(f"Assistant: {normalize_text(assistant_turn)}")
            rendered.append(END_ASSISTANT)
            rendered.append(CHAT_END)
    rendered.append(f"User: {user_text}")
    rendered.append("Assistant:")
    prompt_text = "\n".join(rendered)
    target_text = f" {assistant_text}\n{END_ASSISTANT}\n{CHAT_END}"
    chat_text = prompt_text + target_text
    example_id = f"direct_v8_{category}_{sha1_hex(chat_text)[:24]}_{source_index}"
    return DATA_HELPERS.SFTExample(
        id=example_id,
        dialogue_group=example_id,
        source_group="synthetic_v8_direct_answer",
        source_ref="tools/prepare_tiny_chatbot_v8_direct_answer_corpus.py",
        category=category,
        prompt_text=prompt_text,
        assistant_text=assistant_text,
        target_text=target_text,
        chat_text=chat_text,
        prompt_words=word_count(prompt_text),
        assistant_words=word_count(assistant_text),
    )


def add_example(examples: list, seen: set[str], counters: Counter, category: str, user: str, assistant: str, source_index: int, prior_turns: list[tuple[str, str]] | None = None) -> None:
    example = make_example(category, user, assistant, source_index, prior_turns)
    key = re.sub(r"\s+", " ", example.chat_text.strip().lower())
    if key in seen:
        return
    if not clean_direct_example(example):
        counters[f"rejected_{category}"] += 1
        return
    seen.add(key)
    examples.append(example)
    counters[category] += 1


def clean_direct_example(example) -> bool:
    if not example.prompt_text or not example.assistant_text:
        return False
    if example.assistant_words < 1 or example.assistant_words > 90:
        return False
    if example.prompt_words > 180:
        return False
    lowered = example.assistant_text.lower()
    reject_phrases = (
        "as an ai language model",
        "according to the passage",
        "provided text",
        "[name]",
        "[address]",
        "user:",
        "assistant:",
        "system:",
    )
    if any(phrase in lowered for phrase in reject_phrases):
        return False
    if DATA_HELPERS.looks_non_english(example.prompt_text) or DATA_HELPERS.looks_non_english(example.assistant_text):
        return False
    if DATA_HELPERS.looks_code_or_table_heavy(example.prompt_text) or DATA_HELPERS.looks_code_or_table_heavy(example.assistant_text):
        return False
    return True


def generate_direct_sft_examples(count: int, seed: int) -> tuple[list, Counter]:
    rng = random.Random(seed)
    counters = Counter()
    examples: list = []
    seen: set[str] = set()
    source_index = 0

    greetings = [
        ("hello", "Hi. What would you like help with today?"),
        ("hi", "Hi. I can help with a question, a rewrite, a plan, or a simple explanation."),
        ("can you help me?", "Yes. Tell me what you want to do and what kind of answer would help."),
        ("good morning", "Good morning. What can I help you figure out?"),
        ("hey there", "Hi. Send me the task or question and I will help."),
    ]
    arithmetic_prompt_forms = [
        "what is {expression}?",
        "what is {expression}? {style}",
        "What is {expression}? {style}",
        "Please answer directly: {expression}. {style}",
        "Can you calculate {expression}? {style}",
        "Give only the answer: {expression}. {style}",
        "Practice item {index}: what is {expression}? {style}",
    ]

    def arithmetic_problem(index: int) -> tuple[str, str]:
        if index % 8 == 0:
            return "1 plus 1", "2"
        operation = index % 4
        base = index // 4
        if operation == 0:
            left = (base * 37) % 100 + 1
            right = (base * 19) % 100 + 1
            return f"{left} plus {right}", str(left + right)
        if operation == 1:
            left = (base * 43) % 120 + 20
            right = min(left, (base * 17) % 80 + 1)
            return f"{left} minus {right}", str(left - right)
        if operation == 2:
            left = base % 15 + 1
            right = (base * 5) % 15 + 1
            return f"{left} times {right}", str(left * right)
        divisor = base % 12 + 2
        quotient = (base * 7) % 30 + 1
        dividend = divisor * quotient
        return f"{dividend} divided by {divisor}", str(quotient)
    color_facts = [
        ("green", True), ("green", True), ("green", True), ("blue", True), ("red", True),
        ("yellow", True), ("purple", True), ("orange", True), ("black", True), ("white", True),
        ("pink", True), ("brown", True), ("gray", True), ("teal", True), ("cyan", True),
        ("table", False), ("music", False), ("seven", False), ("Tuesday", False), ("France", False),
        ("Japan", False), ("Paris", False), ("democracy", False), ("dinner", False), ("notebook", False),
    ]
    yes_no_facts = [
        ("Is the sky usually blue on a clear day?", "Yes. On a clear day, the sky usually looks blue."),
        ("Can fish live in water?", "Yes. Fish live in water."),
        ("Is ice hot?", "No. Ice is cold."),
        ("Do birds usually have wings?", "Yes. Birds usually have wings."),
        ("Is a square a shape?", "Yes. A square is a shape."),
        ("Is a minute longer than an hour?", "No. An hour is longer than a minute."),
        ("Is water a liquid at normal room temperature?", "Yes. Water is usually a liquid at normal room temperature."),
        ("Is the USA a democracy or a dictatorship?", "The USA is generally described as a democracy and a constitutional republic."),
        ("I live in the USA. Is the USA a democracy or a dictatorship?", "The USA is generally described as a democracy and a constitutional republic."),
    ]
    capitals = [
        ("France", "Paris"), ("France", "Paris"), ("France", "Paris"), ("France", "Paris"),
        ("Japan", "Tokyo"), ("Canada", "Ottawa"), ("Italy", "Rome"), ("Germany", "Berlin"),
        ("Spain", "Madrid"), ("Mexico", "Mexico City"), ("Brazil", "Brasilia"), ("Egypt", "Cairo"),
        ("Australia", "Canberra"), ("United Kingdom", "London"), ("Ireland", "Dublin"),
        ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Portugal", "Lisbon"),
        ("Greece", "Athens"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
        ("Peru", "Lima"), ("India", "New Delhi"), ("South Korea", "Seoul"),
        ("Thailand", "Bangkok"), ("Kenya", "Nairobi"), ("Morocco", "Rabat"),
    ]
    common_facts = [
        ("What planet do humans live on?", "Humans live on Earth."),
        ("How many days are in a week?", "There are seven days in a week."),
        ("What do bees make?", "Bees make honey."),
        ("What gas do people breathe in to stay alive?", "People need oxygen to stay alive."),
        ("What do you call frozen water?", "Frozen water is called ice."),
        ("What animal says meow?", "A cat says meow."),
        ("What is the opposite of hot?", "The opposite of hot is cold."),
    ]
    names = ["Alex", "Alex", "Alex", "Alex", "Maya", "Sam", "Jordan", "Taylor", "Riley", "Morgan", "Casey", "Jamie", "Avery"]
    places = ["the USA", "Canada", "Mexico", "France", "Japan", "Brazil"]
    favorite_colors = ["green", "blue", "red", "purple", "yellow"]
    rewrites = [
        ("You forgot again.", "Could you please check this when you have a chance?"),
        ("This is wrong.", "I think this needs another look."),
        ("Send it now.", "Could you send it when you are ready?"),
        ("I need that today.", "Could you send that today if possible?"),
        ("Your message was confusing.", "Could you clarify one part of your message for me?"),
        ("You never answer me.", "Could you let me know when you have time to reply?"),
        ("This makes no sense.", "Could you explain this part a little more clearly?"),
        ("Fix this before lunch.", "Could you please look at this before lunch if you have time?"),
        ("I hate this plan.", "I have some concerns about this plan and would like to discuss them."),
        ("You missed the point.", "I think we may be looking at different parts of the issue."),
        ("That idea is bad.", "I am not sure that idea will work well as written."),
        ("You need to redo it.", "Could you please revise this when you have a chance?"),
        ("Stop ignoring this.", "Could you please take a look at this soon?"),
        ("Your draft is messy.", "The draft may be easier to follow with a little more structure."),
        ("I already told you this.", "I may not have explained this clearly before, so I will restate it."),
    ]
    planning_tasks = [
        ("clean my room", "Start with trash, then clear one surface, then put clothes away."),
        ("make a simple dinner", "Pick one protein, one easy side, and one vegetable. Keep it simple."),
        ("study for a quiz", "Review the main notes, practice a few questions, then rest for a short break."),
        ("plan a calm evening", "Choose dinner, one small chore, and one relaxing activity. Do not overpack it."),
        ("write a short email", "Write the main point first, add one detail, then end with the next step."),
    ]
    dinner_pairs = [
        ("pasta with tomato sauce", "rice with eggs and vegetables"),
        ("soup and toast", "bean tacos"),
        ("grilled cheese and salad", "stir-fried rice"),
        ("baked potato with toppings", "omelet with toast"),
    ]
    uncertainty_questions = [
        "What is the exact population of my town today?",
        "What did my teacher say after class yesterday?",
        "What is inside my closed notebook?",
    ]
    everyday_tasks = [
        "write an email", "plan dinner", "clean my desk", "study for a quiz", "choose a movie",
        "make a grocery list", "reply to a friend", "organize notes", "plan a short walk", "pack a bag",
        "start a small project", "prepare for a call", "sort laundry", "make a budget", "pick a lunch",
        "write a thank-you note", "plan a quiet weekend", "review homework", "set up a calendar", "choose a gift",
    ]
    short_style_requests = [
        "Answer briefly.", "Keep it simple.", "Use one sentence.", "Be direct.", "Use plain words.",
        "Keep it practical.", "Avoid extra detail.", "Make it clear.", "Keep it calm.", "Answer like a helpful assistant.",
    ]
    planning_constraints = ["quick", "cheap", "calm", "easy to pause", "low-effort", "simple", "realistic"]
    planning_timeboxes = ["10-minute", "15-minute", "30-minute", "one-hour", "evening", "weekend morning"]
    foods = [
        "pasta", "rice", "eggs", "soup", "tacos", "sandwiches", "salad", "beans",
        "toast", "potatoes", "noodles", "stir-fry", "quesadillas", "oatmeal", "chili",
        "wraps", "roasted vegetables", "lentils", "curry", "grain bowls",
    ]
    rewrite_recipients = ["a friend", "a teammate", "a manager", "a roommate", "a client", "a teacher"]
    rewrite_styles = [
        "Keep it one sentence.",
        "Keep the meaning the same.",
        "Make it calm.",
        "Make it polite.",
        "Use plain words.",
        "Avoid sounding angry.",
        "Make it professional.",
        "Make it friendly.",
        "Keep it short.",
        "Make it easier to receive.",
        "Use a softer tone.",
        "Do not add extra details.",
    ]

    categories = [
        "greeting",
        "arithmetic", "arithmetic", "arithmetic", "arithmetic",
        "color", "color", "color",
        "yes_no_fact", "yes_no_fact",
        "capital", "capital", "capital", "capital",
        "common_fact", "common_fact",
        "name_recall", "name_recall", "name_recall", "name_recall",
        "place_recall", "preference_recall",
        "rewrite", "rewrite",
        "planning", "dinner_ideas", "uncertainty",
    ]
    category_variations = Counter()
    attempts = 0
    max_attempts = max(count * 20, 10_000)
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        category = categories[(attempts - 1) % len(categories)]
        category_variations[category] += 1
        variation = category_variations[category] - 1
        source_index += 1

        if category == "greeting":
            greeting_user, greeting_assistant = greetings[variation % len(greetings)]
            task = everyday_tasks[(variation // len(greetings)) % len(everyday_tasks)]
            style = short_style_requests[(variation // (len(greetings) * len(everyday_tasks))) % len(short_style_requests)]
            user = f"{greeting_user}. I need help to {task}. {style}"
            assistant = greeting_assistant
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "arithmetic":
            expression, answer = arithmetic_problem(variation)
            style = short_style_requests[(variation // 7) % len(short_style_requests)]
            form = arithmetic_prompt_forms[(variation // 4) % len(arithmetic_prompt_forms)]
            user = form.format(expression=expression, style=style, index=variation)
            if form.startswith("Give only") or form.startswith("Please answer"):
                assistant = answer
            else:
                assistant = f"The answer is {answer}."
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "color":
            item, is_color = color_facts[variation % len(color_facts)]
            style = short_style_requests[(variation // len(color_facts)) % len(short_style_requests)]
            prompt_forms = [
                f"Is {item} a color? {style}",
                f"is {item} a color? {style}",
                f"Please answer yes or no: is {item} a color? {style}",
                f"I am checking basic facts. Is {item} a color? {style}",
                f"Fact check {variation}: Is {item} a color? {style}",
            ]
            user = prompt_forms[(variation // (len(color_facts) * len(short_style_requests))) % len(prompt_forms)]
            assistant = f"Yes. {item.capitalize()} is a color." if is_color else f"No. {item.capitalize()} is not a color."
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "yes_no_fact":
            if variation < len(yes_no_facts) * len(short_style_requests):
                user, assistant = yes_no_facts[variation % len(yes_no_facts)]
                user = f"{user} {short_style_requests[(variation // len(yes_no_facts)) % len(short_style_requests)]}"
            else:
                number = variation - len(yes_no_facts) * len(short_style_requests)
                user = f"Is {number} an even number? Answer yes or no."
                assistant = f"{'Yes' if number % 2 == 0 else 'No'}. {number} is {'even' if number % 2 == 0 else 'odd'}."
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "capital":
            country, capital = ("France", "Paris") if variation % 4 == 0 else capitals[variation % len(capitals)]
            style = short_style_requests[(variation // len(capitals)) % len(short_style_requests)]
            forms = [
                f"What is the capital of {country}? {style}",
                f"what is the capital of {country}? {style}",
                f"Tell me the capital of {country}. {style}",
                f"Quick fact check: capital of {country}? {style}",
                f"Capital drill {variation}: what is the capital of {country}? {style}",
            ]
            user = forms[(variation // (len(capitals) * len(short_style_requests))) % len(forms)]
            assistant = f"The capital of {country} is {capital}."
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "common_fact":
            if variation < len(common_facts) * len(short_style_requests) * 4:
                user, assistant = common_facts[variation % len(common_facts)]
                style = short_style_requests[(variation // len(common_facts)) % len(short_style_requests)]
                form_index = (variation // (len(common_facts) * len(short_style_requests))) % 4
                prefix = "" if form_index == 0 else f"Fact drill {variation}: "
                user = f"{prefix}{user} {style}"
            else:
                number = variation - len(common_facts) * len(short_style_requests) * 4
                left = (number * 37) % 1000
                right = (number * 71 + 13) % 1000
                user = f"Which number is larger, {left} or {right}?"
                assistant = f"{max(left, right)} is larger."
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "name_recall":
            name = "Alex" if variation % 4 == 0 else names[variation % len(names)]
            note_number = variation // len(names)
            prior_forms = [
                (f"my name is {name.lower()}.", f"Nice to meet you, {name}."),
                (f"my name is {name.lower()}. The memory label is {note_number}.", f"Nice to meet you, {name}."),
                (f"My name is {name}.", f"Nice to meet you, {name}."),
                (f"My name is {name}. The memory label is {note_number}.", f"Nice to meet you, {name}."),
                (f"Please remember that my name is {name}.", f"Got it. Your name is {name}."),
                (f"Please remember that my name is {name}. The memory label is {note_number}.", f"Got it. Your name is {name}."),
                (f"My name is {name}. My note number is {note_number}.", f"Nice to meet you, {name}."),
                (f"For this chat, my name is {name}.", f"Understood. Your name is {name}."),
                (f"For this chat, my name is {name}. The memory label is {note_number}.", f"Understood. Your name is {name}."),
            ]
            prior = [prior_forms[variation % len(prior_forms)]]
            user_forms = ["what is my name?", "What is my name?", "Can you tell me my name?", "Please remind me of my name."]
            user = user_forms[(variation // len(prior_forms)) % len(user_forms)]
            add_example(examples, seen, counters, category, user, f"Your name is {name}.", source_index, prior)
        elif category == "place_recall":
            place = places[variation % len(places)]
            note_number = variation // len(places)
            prior = [(f"I live in {place}. My note number is {note_number}.", f"Got it. You live in {place}.")]
            add_example(examples, seen, counters, category, "Where do I live?", f"You live in {place}.", source_index, prior)
        elif category == "preference_recall":
            color = favorite_colors[variation % len(favorite_colors)]
            note_number = variation // len(favorite_colors)
            prior = [(f"My favorite color is {color}. My note number is {note_number}.", f"Got it. Your favorite color is {color}.")]
            add_example(examples, seen, counters, category, "What is my favorite color?", f"Your favorite color is {color}.", source_index, prior)
        elif category == "rewrite":
            rough, kind = rewrites[variation % len(rewrites)]
            recipient = rewrite_recipients[(variation // len(rewrites)) % len(rewrite_recipients)]
            style = rewrite_styles[(variation // (len(rewrites) * len(rewrite_recipients))) % len(rewrite_styles)]
            forms = [
                (f"rewrite this kindly for {recipient}: {rough} {style}", kind),
                (f"Rewrite this kindly for {recipient}: {rough} {style}", kind),
                (f"Make this sound polite for {recipient}: {rough} {style}", kind),
                (f"Give me a gentler version for {recipient}: {rough} {style}", kind),
            ]
            user, assistant = forms[(variation // (len(rewrites) * len(rewrite_recipients) * len(rewrite_styles))) % len(forms)]
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "planning":
            base_task, base_plan = planning_tasks[variation % len(planning_tasks)]
            if variation < len(planning_tasks):
                task = base_task
            else:
                task = everyday_tasks[variation % len(everyday_tasks)]
            timebox = planning_timeboxes[(variation // len(everyday_tasks)) % len(planning_timeboxes)]
            constraint = planning_constraints[(variation // (len(everyday_tasks) * len(planning_timeboxes))) % len(planning_constraints)]
            user = f"Give me a {timebox} plan to {task}. Keep it {constraint}."
            assistant = base_plan if task == base_task else f"Start with the smallest useful step, spend one {timebox} block on it, then check what remains. Keep it {constraint}."
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "dinner_ideas":
            first = foods[variation % len(foods)]
            second_index = (variation // len(foods) + 3) % len(foods)
            if foods[second_index] == first:
                second_index = (second_index + 1) % len(foods)
            second = foods[second_index]
            style = short_style_requests[(variation // (len(foods) * len(foods))) % len(short_style_requests)]
            user = f"Give me two simple dinner ideas. {style}"
            assistant = f"Two simple dinner ideas are {first} and {second}."
            add_example(examples, seen, counters, category, user, assistant, source_index)
        elif category == "uncertainty":
            base_question = uncertainty_questions[variation % len(uncertainty_questions)]
            label = variation // len(uncertainty_questions)
            user = f"{base_question} The private note label is {label}."
            assistant = "I do not know that from the information here. If you give me more details, I can help reason about it."
            add_example(examples, seen, counters, category, user, assistant, source_index)

        if attempts % 97 == 0:
            rng.shuffle(categories)

    if len(examples) < count:
        raise RuntimeError(f"only generated {len(examples)} direct examples after {attempts} attempts")
    counters["total_direct_examples"] = len(examples)
    return examples, counters


def tiny_local_pretrain_documents() -> list[str]:
    return [
        "A helpful assistant answers the user's question directly. It gives one clear answer and then stops.",
        "Simple arithmetic uses exact answers. One plus one is two. Two plus two is four. Three plus four is seven.",
        "Green, blue, red, yellow, and purple are colors. A table is not a color. A number is not a color.",
        "Paris is the capital of France. Tokyo is the capital of Japan. Ottawa is the capital of Canada. Rome is the capital of Italy.",
        "A short conversation can include facts from the user. If the user says their name is Alex, the assistant can remember Alex in the next turn.",
        "Good chatbot replies avoid writing fake User turns. They answer as the assistant and stop at the end of the answer.",
        "A simple plan should be short, practical, and easy to follow. The first step should be clear.",
        "Kind rewrites keep the meaning but make the message more polite and easier to receive.",
        "The United States is generally described as a democracy and a constitutional republic. It is not normally described as a dictatorship.",
        "When the assistant does not know a private fact, it should say that it does not know and ask for more information.",
    ] * 20


def write_tiny_local_pretrain(output_dir: Path) -> tuple[Counter, dict]:
    counters = Counter()
    writers = DATA_HELPERS.open_split_writers(output_dir / "pretrain")
    total_characters = 0
    try:
        for index, document in enumerate(tiny_local_pretrain_documents()):
            split_name = "training"
            if index % 20 == 0:
                split_name = "validation"
            elif index % 20 == 1:
                split_name = "testing"
            DATA_HELPERS.write_split_document(writers, split_name, document)
            counters[f"{split_name}_documents"] += 1
            total_characters += len(document) + 2
    finally:
        DATA_HELPERS.close_writers(writers)
    counters["pretrain_documents"] = counters["training_documents"] + counters["validation_documents"] + counters["testing_documents"]
    counters["pretrain_characters"] = total_characters
    return counters, {"tiny_local": ["built_in_v8_smoke_documents"]}


def split_examples(examples: list) -> tuple[list, list, list]:
    ordered = sorted(examples, key=lambda example: sha1_hex(example.id))
    training = []
    validation = []
    testing = []
    for example in ordered:
        bucket = int(hashlib.sha1(example.chat_text.encode("utf-8")).hexdigest()[:12], 16) % 1000
        if bucket < 20:
            validation.append(example)
        elif bucket < 40:
            testing.append(example)
        else:
            training.append(example)
    if not validation and len(ordered) >= 3:
        validation.append(training.pop())
    if not testing and len(ordered) >= 3:
        testing.append(training.pop())
    return training, validation, testing


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare v8 direct-answer scratch chatbot corpus for KeemenaLM.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pretrain-source", choices=("downloaded", "tiny_local"), default="downloaded")
    parser.add_argument("--pretrain-max-characters", type=int, default=1_500_000_000)
    parser.add_argument("--tinystories-train-files", type=int, default=4)
    parser.add_argument("--fineweb-files", type=int, default=8)
    parser.add_argument("--direct-sft-examples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args(argv)
    for name in ("pretrain_max_characters", "tinystories_train_files", "fineweb_files", "direct_sft_examples"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.direct_sft_examples <= 0:
        parser.error("--direct-sft-examples must be > 0")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pretrain_source == "tiny_local":
        pretrain_counters, pretrain_sources = write_tiny_local_pretrain(output_dir)
    else:
        pretrain_counters, pretrain_sources = DATA_HELPERS.collect_pretrain_to_splits(args, output_dir)

    direct_examples, direct_counters = generate_direct_sft_examples(args.direct_sft_examples, args.seed)
    sft_train, sft_validation, sft_testing = split_examples(direct_examples)
    DATA_HELPERS.write_sft_split(output_dir / "sft", "training", sft_train)
    DATA_HELPERS.write_sft_split(output_dir / "sft", "validation", sft_validation)
    DATA_HELPERS.write_sft_split(output_dir / "sft", "testing", sft_testing)

    category_counts = Counter(example.category for example in direct_examples)
    metadata = {
        "dataset_name": "tiny_chatbot_v8_direct_answer_corpus",
        "intended_scope": "Scratch chatbot corpus with general pretraining plus direct-answer assistant-only final targets. No raw chat-LM continuation stage and no repeated synthetic template closers.",
        "sources": {
            "pretrain_source": args.pretrain_source,
            "tinystories": DATA_HELPERS.TINYSTORIES_REPO,
            "fineweb_edu": DATA_HELPERS.FINEWEB_EDU_REPO,
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
                "training": len(sft_train),
                "validation": len(sft_validation),
                "testing": len(sft_testing),
                "source_groups": {"synthetic_v8_direct_answer": len(direct_examples)},
                "categories": dict(category_counts),
            },
        },
        "counters": {
            "pretrain": dict(pretrain_counters),
            "direct_sft": dict(direct_counters),
        },
        "v8_policy": {
            "final_objective": "assistant_only direct-answer SFT with general pretrain replay",
            "forbidden_major_stage": "raw chat transcript continuation / chat_lm all_tokens",
            "behavior_gate": [
                "fake role marker rejection",
                "repetition-loop rejection",
                "arithmetic/color/capital/name-recall checks",
            ],
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)

    print(f"Prepared v8 direct-answer corpus at: {output_dir}")
    print(f"Pretrain docs train/val/test: {metadata['counts']['pretrain']['training']}/{metadata['counts']['pretrain']['validation']}/{metadata['counts']['pretrain']['testing']}")
    print(f"Pretrain chars: {metadata['counts']['pretrain']['characters']}")
    print(f"SFT examples train/val/test: {len(sft_train)}/{len(sft_validation)}/{len(sft_testing)}")
    print(f"SFT categories: {dict(category_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

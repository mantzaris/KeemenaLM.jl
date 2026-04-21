using Downloads
using JSON3
using SHA

struct ChatExample
    id::String
    dialogue_group::String
    source_group::String
    source_ref::String
    category::String
    turn_count::Int
    chat_text::String
end

const END_ASSISTANT = "<END_ASSISTANT>"
const CHAT_END = "<CHAT_END>"
const DEFAULT_OUTPUT_DIRECTORY = joinpath(@__DIR__, "..", "tmp", "tiny_chatbot_real_corpus_v2")
const OASST1_READY_MESSAGES_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz"
const OASST1_LICENSE = "Apache-2.0"
const TARGET_EXTERNAL_SINGLE_TURN_MAX = 14000
const TARGET_EXTERNAL_MULTI_TURN_MAX = 2200
const MAX_ACCEPTED_OASST_RANK = 4

function repository_root()::String
    return normpath(joinpath(@__DIR__, ".."))
end

function main(arguments)
    output_directory = isempty(arguments) ? DEFAULT_OUTPUT_DIRECTORY : abspath(arguments[1])
    mkpath(output_directory)

    local_examples, local_metadata = build_local_curated_core()
    external_examples, external_status = build_external_oasst1_subset(output_directory)

    all_examples = vcat(local_examples, external_examples)
    unique_examples = deduplicate_examples(all_examples)
    training, validation, testing = split_examples(unique_examples)

    metadata = build_metadata(
        unique_examples,
        training,
        validation,
        testing,
        local_metadata,
        external_status,
    )

    write_dataset(output_directory, training, validation, testing, metadata)

    println("Prepared dataset at: ", output_directory)
    println("Total examples: ", length(unique_examples))
    println("Training/validation/testing: ", length(training), "/", length(validation), "/", length(testing))
    println("Local curated examples: ", count(example -> example.source_group == "local_curated_core", unique_examples))
    println("External OASST1 single-turn examples: ", count(example -> example.source_group == "oasst1_filtered_single_turn", unique_examples))
    println("External OASST1 multi-turn examples: ", count(example -> example.source_group == "oasst1_filtered_multi_turn", unique_examples))
    println("External fetch status: ", external_status["status"])
end

function build_local_curated_core()
    examples = ChatExample[]

    local_source_refs = [
        "authored local conversational core inside tools/prepare_tiny_chatbot_demo_dataset.jl",
        joinpath(repository_root(), "docs", "src", "index.md"),
        joinpath(repository_root(), "notes", "official_models.md"),
        normpath(joinpath(repository_root(), "..", "KeemenaPreprocessing.jl", "docs", "src", "index.md")),
        normpath(joinpath(repository_root(), "..", "KeemenaPreprocessing.jl", "docs", "src", "guides", "quickstart.md")),
        normpath(joinpath(repository_root(), "..", "KeemenaPreprocessing.jl", "docs", "src", "guides", "offsets.md")),
        normpath(joinpath(repository_root(), "..", "KeemenaSubwords.jl", "docs", "src", "index.md")),
        normpath(joinpath(repository_root(), "..", "KeemenaSubwords.jl", "docs", "src", "loading.md")),
        normpath(joinpath(repository_root(), "..", "KeemenaSubwords.jl", "docs", "src", "integration.md")),
    ]

    append!(examples, one_turn_examples("greet", "greeting", "local_curated_core", local_source_refs[1], [
        ("Hi.", "Hi. What would you like help with?"),
        ("Hello there.", "Hello. What are you working on right now?"),
        ("Hey, are you around?", "Yes. I'm here and ready to help."),
        ("Good morning.", "Good morning. What would make today easier?"),
        ("Good evening.", "Good evening. What do you need a hand with?"),
        ("How are you?", "I'm ready to help. What's on your mind?"),
        ("Can we chat for a minute?", "Sure. What would you like to talk through?"),
        ("I just want a quick answer.", "No problem. Ask the quick question and I'll keep it short."),
        ("I don't know where to start.", "That's okay. Tell me the situation in one sentence and we'll start there."),
        ("Can you help me think clearly?", "Yes. Give me the problem, and I'll break it down simply."),
        ("Hi, I need a little help.", "Of course. What kind of help do you need?"),
        ("Hey assistant.", "Hey. What can I do for you?"),
        ("Are you busy?", "I'm available. Go ahead."),
        ("Can you keep this simple?", "Yes. I'll keep the answer short and plain."),
        ("I'd like a calm answer.", "Sure. Tell me the question and I'll keep the tone calm."),
        ("Can you help without making it complicated?", "Yes. I'll keep it practical."),
        ("I'm feeling stuck.", "Let's make it smaller. What is the first thing that's blocking you?"),
        ("Can we do one step at a time?", "Absolutely. Tell me the task, and we'll take it in small steps."),
        ("I'm low on energy today.", "Understood. We can aim for the smallest useful next step."),
        ("Can you be direct?", "Yes. Ask the question and I'll answer directly."),
        ("I need a friendly answer.", "Sure. What's the question?"),
        ("Please don't give me a lecture.", "Understood. I'll keep it brief and useful."),
        ("Can you just point me in the right direction?", "Yes. Tell me what you're deciding between."),
        ("Hi, can you help me sort something out?", "Yes. Give me the short version, and we'll sort it out."),
    ]))

    append!(examples, one_turn_examples("explain", "simple_explanation", "local_curated_core", local_source_refs[1], [
        ("What is a habit tracker?", "A habit tracker is a simple way to notice whether you're doing a small routine consistently over time."),
        ("What does 'good enough' mean in practice?", "It usually means finishing the useful version first instead of chasing a perfect version too early."),
        ("What is a fallback plan?", "A fallback plan is the simpler backup you can use if the main plan becomes too hard or too slow."),
        ("What is a soft deadline?", "A soft deadline is a target date you aim for without treating it as a hard failure point."),
        ("What does 'scope creep' mean?", "Scope creep means a task grows beyond the original goal because extra ideas keep getting added."),
        ("What is a rough draft?", "A rough draft is the first usable version that helps you think, revise, and improve later."),
        ("What is a bottleneck?", "A bottleneck is the part of a process that limits how fast everything else can move."),
        ("What is a simple routine?", "A simple routine is a repeated sequence that reduces decision-making for everyday tasks."),
        ("What does prioritizing really mean?", "It means deciding what matters most first, not trying to do everything at once."),
        ("What is overthinking?", "Overthinking is spending too much energy circling decisions instead of testing a small next step."),
        ("What is a working note?", "A working note is a quick note you make to think clearly, not a polished final document."),
        ("What is a tiny experiment?", "A tiny experiment is a small test that helps you learn before making a bigger commitment."),
        ("What does being organized actually mean?", "It means your important information and tasks are easy to find and easy to act on."),
        ("What is a plain-language answer?", "A plain-language answer uses simple words and avoids unnecessary jargon."),
        ("What is a checklist good for?", "A checklist helps you remember repeatable steps without keeping them all in your head."),
        ("What is a decision rule?", "A decision rule is a short guideline you use to make similar choices more consistently."),
        ("What is a small win?", "A small win is a modest step that makes progress visible and helps momentum build."),
        ("What is a quick review?", "A quick review is a short look back at what worked, what didn't, and what to change next."),
        ("What does 'be specific' mean?", "It means using concrete details so the next action is obvious."),
        ("What is a gentle reminder?", "A gentle reminder nudges attention back to something important without sounding harsh."),
        ("What is a realistic plan?", "A realistic plan matches the time, energy, and tools you actually have."),
        ("What is a mental reset?", "A mental reset is a short pause that helps you stop spiraling and return to the next useful step."),
        ("What is a follow-up question for?", "A follow-up question narrows the problem so the answer can fit the real situation."),
        ("What is a support tool?", "A support tool is anything that makes a task easier, clearer, or more repeatable."),
    ]))

    append!(examples, one_turn_examples("task", "everyday_help", "local_curated_core", local_source_refs[1], [
        ("Can you help me start a to-do list for today?", "Yes. Start with three items: one important task, one quick win, and one thing that will make tomorrow easier."),
        ("How do I clean up a messy task list?", "Group similar tasks, drop low-value items, and choose just a few for today."),
        ("How should I begin a hard task?", "Make the first step small enough that you can do it without needing extra motivation."),
        ("Can you help me make a short plan for tonight?", "Yes. Pick one useful task, one basic reset task, and one relaxing activity."),
        ("How do I stop procrastinating for ten minutes?", "Choose a tiny starting action, set a timer, and focus only on beginning."),
        ("What should I do when everything feels urgent?", "Pause, sort by real consequences, and pick the next one thing that actually needs movement."),
        ("Can you help me write a gentle reminder to myself?", "Sure. Try: 'Do one useful thing now, and let that be enough for this moment.'"),
        ("How do I plan a low-stress morning?", "Decide your first task tonight, lay out what you need, and keep the morning choice count low."),
        ("How do I make a boring task easier?", "Shrink it, add a timer, and define what 'done enough' means before you start."),
        ("Can you help me reset after a bad work session?", "Yes. Take a short break, write down what went wrong, and restart with one smaller objective."),
        ("How do I keep track of small errands?", "Put them in one visible list and batch them by location or time if you can."),
        ("Can you help me organize notes from today?", "Yes. Pull out the decisions, the open questions, and the next actions."),
        ("What should I do before ending the workday?", "Wrap up one loose task, note the next step for tomorrow, and close your tabs or notes cleanly."),
        ("How do I ask for help without rambling?", "State the problem, say what you've tried, and ask one specific question."),
        ("How can I make a weekend plan without overbooking?", "Pick one priority, one social or fun activity, and leave open space around them."),
        ("Can you help me choose between two good options?", "Yes. Compare them on effort, upside, and how reversible the choice is."),
        ("How do I recover when I miss a routine?", "Restart at the next opportunity and avoid turning one miss into a longer break."),
        ("Can you help me break a project into pieces?", "Sure. Define the outcome, split it into stages, and turn the first stage into a concrete action list."),
        ("How do I prepare for a short meeting?", "Write the goal, the top questions, and the one decision you want by the end."),
        ("Can you help me make a simple shopping plan?", "Yes. Group the list by area, note the must-haves first, and avoid browsing without a plan."),
        ("How do I keep a decision from dragging on?", "Set a deadline, define what information is enough, and choose once that bar is met."),
        ("What should I do if I feel scattered?", "Write down everything pulling at you, then circle only the next useful thing."),
    ]))

    append!(examples, one_turn_examples("rewrite", "rewrite_clarify", "local_curated_core", local_source_refs[1], [
        ("Can you rewrite this to sound kinder: 'You forgot to send the file again.'", "Sure. Try: 'I think the file may have been missed. Could you send it when you have a moment?'"),
        ("Can you rewrite this to sound clearer: 'Let's maybe do the thing soon.'", "Sure. Try: 'Let's do this tomorrow afternoon if that works for you.'"),
        ("Can you make this shorter: 'I am reaching out because I wanted to follow up regarding the notes from yesterday.'", "Yes. Try: 'I'm following up on yesterday's notes.'"),
        ("Can you make this less stiff: 'Please advise regarding your availability.'", "Yes. Try: 'Let me know when you're available.'"),
        ("Can you rewrite this so it sounds calm: 'This is getting out of hand.'", "Sure. Try: 'I think this is getting harder to manage than we expected.'"),
        ("Can you make this friendlier: 'I need this today.'", "Try: 'Could you send this today if possible?'"),
        ("Can you simplify this sentence: 'The primary objective is to establish alignment on next steps.'", "Yes. Try: 'The goal is to agree on the next steps.'"),
        ("Can you rewrite this note so it sounds more direct: 'It might be useful to perhaps review the draft again.'", "Sure. Try: 'Please review the draft one more time.'"),
        ("Can you make this more polite: 'That doesn't make sense.'", "Try: 'I may be missing something. Could you walk me through that part?'"),
        ("Can you rewrite this to sound more confident: 'I was kind of thinking maybe we could try this.'", "Sure. Try: 'I think this is the best option to try next.'"),
        ("Can you make this sound warmer: 'Received. Thanks.'", "Yes. Try: 'Got it, thanks for sending that over.'"),
        ("Can you make this sentence easier to read: 'The schedule remains subject to revision pending confirmation.'", "Try: 'The schedule may change until we get final confirmation.'"),
        ("Can you rewrite this for plain language: 'We need to reduce friction in the workflow.'", "Sure. Try: 'We need to make the workflow easier to use.'"),
        ("Can you rewrite this so it doesn't sound passive-aggressive: 'As mentioned earlier, this was already covered.'", "Try: 'We talked about this earlier, but I'm happy to restate the key point.'"),
        ("Can you turn this into a quick message: 'I won't be able to attend due to a scheduling conflict.'", "Yes. Try: 'I can't make it because of a scheduling conflict.'"),
        ("Can you make this less wordy: 'At this point in time, I do not believe that is necessary.'", "Try: 'I don't think that's necessary right now.'"),
        ("Can you rewrite this so it sounds more helpful: 'Read the instructions.'", "Try: 'The instructions should cover it, but I can also help if anything is unclear.'"),
        ("Can you make this sound more reassuring: 'There are still issues.'", "Try: 'There are still a few issues, but they look manageable.'"),
        ("Can you make this more natural: 'Your response would be appreciated at your earliest convenience.'", "Try: 'I'd appreciate a response when you have a chance.'"),
        ("Can you rewrite this to sound less abrupt: 'Send the updated version.'", "Try: 'Could you send the updated version when it's ready?'"),
    ]))

    append!(examples, one_turn_examples("brainstorm", "light_brainstorming", "local_curated_core", local_source_refs[1], [
        ("Can you give me three cozy birthday ideas for a rainy day?", "Sure. You could host a movie-and-dessert night, do a tea-and-board-game evening, or set up a simple at-home brunch."),
        ("Can you brainstorm a few names for a tiny study group?", "Yes. Try names like Quiet Hour, Steady Work Club, Focus Circle, or Little Study Lab."),
        ("Can you suggest a few low-pressure hobbies to try?", "Sure. Try sketching, casual baking, short walks with photos, journaling, or simple houseplant care."),
        ("Can you give me a few ideas for a thoughtful thank-you gesture?", "Yes. You could write a short note, bring a favorite snack, or mention one specific thing you appreciated."),
        ("Can you suggest some weekend reset ideas?", "Try a room reset, a grocery restock, a simple meal prep block, and one quiet hour without notifications."),
        ("Can you brainstorm easy indoor date ideas?", "Sure. Cook one simple meal together, do a movie theme night, or try a tiny trivia game at home."),
        ("Can you give me a few ideas for a low-budget gift?", "Yes. A handwritten note, a framed photo, a small snack bundle, or a playlist with a short message can work well."),
        ("Can you help me think of a calm evening routine?", "Try dimming the lights, doing one quick cleanup task, making tomorrow's first decision, and ending with a quiet activity."),
        ("Can you suggest some simple ways to celebrate a small win?", "Sure. Take a short walk, tell a friend, mark it in a notebook, or give yourself a relaxed hour afterward."),
        ("Can you give me a few ideas for a screen-free break?", "Yes. Stretch, tidy one small area, make tea, sit outside, or write a quick paper note."),
        ("Can you brainstorm some friendly icebreakers for a small group?", "Try asking about a recent small win, a comfort food, a useful app, or a hobby people are curious about."),
        ("Can you suggest some tiny rewards for finishing work?", "A short walk, a favorite snack, a guilt-free episode of something easy, or twenty minutes with no tasks can work well."),
        ("Can you help me think of simple journaling prompts?", "Sure. Try: 'What felt heavy today?', 'What helped?', and 'What do I want tomorrow to feel like?'"),
        ("Can you suggest some low-effort lunch ideas?", "Try wraps, rice bowls, pasta with vegetables, soup and toast, or leftovers with one fresh add-on."),
        ("Can you brainstorm a few ways to make reading more consistent?", "Yes. Keep one book visible, read for ten minutes, pair reading with tea, and stop while you still want more."),
        ("Can you suggest some tiny acts of kindness?", "Send a thoughtful message, hold a door, leave a clear note, or thank someone for a specific helpful thing."),
        ("Can you give me a few ways to make a room feel calmer?", "Reduce clutter in one zone, soften the lighting, clear surfaces, and keep one comforting object visible."),
        ("Can you brainstorm ideas for a simple personal reset day?", "Sleep a little more, clean one area, eat something steady, go outside, and limit decisions."),
        ("Can you suggest a few easy conversation starters?", "Ask what someone is looking forward to, what they've enjoyed lately, or what made their week easier."),
        ("Can you help me think of names for a tiny chatbot project?", "Sure. Try names like Little Chat, Quiet GPT, Pocket Assistant, Tiny Talker, or Plain Bot."),
        ("Can you suggest some gentle goals for a stressful week?", "Aim to protect sleep, do the most important task earlier, eat regularly, and lower the number of optional commitments."),
        ("Can you brainstorm some fun rainy-day solo activities?", "Try baking, sorting photos, making a playlist, reading somewhere cozy, or doing a short creative project."),
        ("Can you suggest a few ideas for a simple morning reset?", "Open a window, drink water, put your phone down for a minute, and decide the first task before checking messages."),
        ("Can you give me a few low-stakes creative prompts?", "Write a letter to your future self, sketch your desk, describe a made-up cafe, or invent a tiny daily ritual."),
    ]))

    append!(examples, one_turn_examples("plan", "planning", "local_curated_core", local_source_refs[1], [
        ("Can you help me plan a focused one-hour work block?", "Yes. Pick one clear outcome, work for fifty minutes, and use the last ten to close loose ends and note the next step."),
        ("How should I plan a short study session?", "Choose one topic, one outcome, and one review step so the session has a clear shape."),
        ("Can you help me plan an easy Sunday reset?", "Sure. Do one tidy task, one food prep task, one calendar check, and then stop."),
        ("How do I plan a simple evening after a long day?", "Keep it basic: eat something, do one small reset task, and choose one relaxing activity."),
        ("Can you help me plan a gentle start to tomorrow?", "Set out what you need tonight, choose the first task now, and keep the morning low-friction."),
        ("How do I plan around low motivation?", "Use smaller tasks, shorter blocks, and fewer commitments than you think you 'should' handle."),
        ("Can you help me plan a catch-up afternoon?", "Start with the one thing that reduces the most pressure, then do quick wins only if you still have energy."),
        ("How should I plan errands without wasting time?", "Group stops by area, put the must-have stop first, and leave some buffer instead of overpacking the route."),
        ("Can you help me plan a clean handoff note?", "List what changed, what still matters, and the exact next action someone should take."),
        ("How do I plan a realistic reading goal?", "Base it on minutes or pages you can actually sustain, not on an ideal version of yourself."),
        ("Can you help me plan a quiet morning off?", "Choose one thing that feels nourishing, one thing that feels useful, and leave room for slowness."),
        ("How do I plan a small personal project?", "Start with a tiny version you can finish, then let later versions grow from that."),
        ("Can you help me plan a short writing session?", "Decide whether the goal is drafting, editing, or outlining, and don't try to do all three at once."),
        ("How should I plan a stressful week?", "Protect sleep, reduce optional tasks, and decide what can be postponed before the week gets crowded."),
        ("Can you help me plan a tidy desk reset?", "Clear trash first, group loose papers, return tools to one place, and stop when the space is usable."),
        ("How do I plan a quick home reset before guests arrive?", "Focus on the entry, bathroom, visible surfaces, and a few comfort details like light or music."),
        ("Can you help me plan my first day with a new habit?", "Make the first version easy, attach it to an existing cue, and define success very modestly."),
        ("How should I plan a short break so it actually helps?", "Choose something that changes your state a bit, like movement, water, air, or silence."),
        ("Can you help me plan a work restart after vacation?", "Review what changed, pick the top priority, and ignore the urge to handle everything at once."),
        ("How do I plan a simple weekly review?", "Look at what moved forward, what got stuck, and what must happen next week."),
        ("Can you help me plan a short conversation with someone about a problem?", "Start with the goal, name the issue clearly, and ask for one concrete next step before the conversation ends."),
        ("How should I plan a calm bedtime routine?", "Lower stimulation, do one tiny reset task, and decide tomorrow's first move before bed."),
    ]))

    append!(examples, one_turn_examples("limits", "limitations_and_refusals", "local_curated_core", local_source_refs[1], [
        ("Can you decide my whole life for me?", "No. I can help you think it through, but the decision should still be yours."),
        ("Can you guarantee this plan will work?", "No. I can suggest a reasonable plan, but I can't guarantee the outcome."),
        ("Can you tell me what someone else is secretly thinking?", "No. I can help you interpret the situation carefully, but I can't know another person's private thoughts."),
        ("Can you make me stop feeling anxious right now?", "I can't make feelings disappear, but I can help you slow things down and choose a calming next step."),
        ("Can you write something cruel for me?", "No. I can help you say the hard thing clearly without making it cruel."),
        ("Can you help me trick someone?", "No. I can help you communicate honestly or set a boundary instead."),
        ("Can you give me a perfect answer with no uncertainty?", "No. I can give a useful answer, but some uncertainty is normal."),
        ("Can you do my work for me and hide it?", "No. I can help you outline it, simplify it, or get started honestly."),
        ("Can you tell me how to hurt someone?", "No. I can help with de-escalation, safety, or getting support instead."),
        ("Can you help me with something illegal?", "No. I can help with a safe or lawful alternative."),
        ("Can you fix every problem in one answer?", "No. I can help with the next useful step, which is often the real way progress starts."),
        ("Can you pretend to know facts you aren't sure about?", "No. It's better for me to be clear about uncertainty than to sound confident and be wrong."),
        ("Can you give medical certainty from one sentence?", "No. I can suggest general next steps, but not a diagnosis or certainty."),
        ("Can you answer even if my request is unclear?", "I can try, but if the request is vague I'll usually do better by asking a clarifying question first."),
        ("Can you help me write a manipulative apology?", "No. I can help you write a sincere apology that takes responsibility instead."),
        ("Can you tell me exactly what to do without context?", "Not reliably. A short clarifying detail usually makes the advice much better."),
    ]))

    append!(examples, one_turn_examples("smalltask", "small_task_help", "local_curated_core", local_source_refs[1], [
        ("Can you suggest a subject line for a friendly follow-up email?", "Sure. Try: 'Quick follow-up', 'Checking in', or 'Following up on our note'."),
        ("Can you help me title a short note about next steps?", "Yes. Try: 'Next Steps', 'Current Plan', or 'What Happens Next'."),
        ("Can you give me a simple agenda for a 15-minute check-in?", "Sure. Start with status, then blockers, then the next action and owner."),
        ("Can you help me write a short thank-you text?", "Yes. Try: 'Thanks again for the help today. I really appreciated it.'"),
        ("Can you help me ask for a small extension politely?", "Sure. Try: 'Would it be possible to have a little more time on this? I want to do it carefully.'"),
        ("Can you make this reminder sound gentle: 'Don't forget the form.'", "Try: 'Quick reminder about the form when you have a moment.'"),
        ("Can you help me turn this into a checklist?", "Yes. Send the steps, and I can turn them into a short checklist."),
        ("Can you suggest a clearer way to ask for feedback?", "Try: 'Could you give me feedback on clarity and structure?'"),
        ("Can you help me phrase a small boundary?", "Sure. Try: 'I can't take that on today, but I can look at it tomorrow.'"),
        ("Can you help me write a short intro message?", "Yes. Keep it to who you are, why you're reaching out, and one clear ask."),
        ("Can you suggest a calm response when I'm running late?", "Try: 'I'm running a little behind. I'll be there as soon as I can.'"),
        ("Can you give me a short way to close a note politely?", "Sure. Try: 'Thanks', 'Appreciate it', or 'Take care'."),
    ]))

    append!(examples, one_turn_examples("tech", "keemena_flavor", "local_curated_core", local_source_refs[2], [
        ("What is KeemenaLM.jl in simple terms?", "KeemenaLM.jl is a small Julia project for GPT-style language-model experiments, training, saving, loading, and generation demos."),
        ("What is a bundle in KeemenaLM.jl?", "A bundle is the saved model package that keeps the model config and weights together so you can reload them later."),
        ("What is a checkpoint for?", "A checkpoint lets you save training progress so you can resume later or inspect the model state at that point."),
        ("What is the difference between a checkpoint and a bundle?", "A checkpoint is mainly for training state, while a bundle is the export-style artifact you use for loading and generation."),
        ("Can KeemenaLM.jl load remote models automatically?", "Not as a general feature. The current path is still much more local and explicit than a broad remote-model workflow."),
        ("What is the official model flow in simple terms?", "It is the path for resolving a known model source and loading it in a controlled way instead of guessing from arbitrary files."),
        ("What does chat! mean in KeemenaLM.jl?", "It is the chat-oriented generation path, but it is still only as good as the model and training recipe behind it."),
        ("What does KeemenaPreprocessing.jl do?", "It prepares text for tokenizer or model workflows, with attention to offsets, alignments, and reproducible preprocessing steps."),
        ("Why do offsets matter?", "Offsets help you map tokenized text back to the original text, which matters for alignment and downstream tooling."),
        ("What does streaming preprocessing mean here?", "It means processing text in a way that can scale through larger inputs without treating everything as one giant in-memory blob."),
        ("What is KeemenaSubwords.jl for?", "It handles tokenizer loading, tokenizer formats, normalization details, and training-ready outputs for subword workflows."),
        ("What tokenizer formats does KeemenaSubwords.jl support?", "It supports several common tokenizer formats so you can load or inspect tokenizers in a more uniform Julia workflow."),
        ("What does integration mean across the Keemena packages?", "It means preprocessing, tokenizer handling, and model workflows are meant to connect cleanly instead of living as separate one-off scripts."),
        ("Why would someone use a tiny char-level path first?", "Because it is simple, explicit, and useful for proving the training and export pipeline before chasing model quality."),
        ("Why is a tiny chatbot still weak even if training works?", "Because a working pipeline is not the same thing as a strong model. Data quality, tokenizer choice, and model capacity still matter."),
        ("What does boundary-aware chat training mean?", "It means the training stream includes explicit markers so the model can see where one chat example ends and the next begins."),
        ("Why would a docs-shaped dataset hurt chat quality?", "Because the model learns the tone and structure it sees. If the data reads like notes or manuals, the replies often do too."),
        ("What is the point of a tiny demo dataset?", "It gives you a small, inspectable way to test the full training, export, load, and generation path without pretending it is production data."),
        ("Why keep the local curated core visible in the final dataset?", "Because it anchors the assistant style you actually want instead of letting the external supplement define the whole tone."),
        ("Why keep the dataset small enough to inspect?", "Because manual inspection is one of the fastest ways to catch bad tone, bad supervision, and noisy examples in a tiny chatbot project."),
        ("What does tokenizer persistence still being separate mean?", "It means tokenizer artifacts may still live outside the model bundle, so you need to keep track of both pieces explicitly."),
        ("Why is a tiny GPT-style chatbot demo different from a docs assistant?", "The main goal shifts from accurate package QA to recognizable conversational behavior and short, helpful back-and-forth replies."),
        ("What is a reasonable non-goal for this tiny chatbot?", "It should not pretend to be a production assistant or a web-scale knowledge model."),
        ("Why use explicit chat markers like <END_ASSISTANT> and <CHAT_END>?", "They give the model a cleaner signal for answer endings and example boundaries than plain newlines alone."),
    ]))

    append!(examples, multi_turn_examples("clarify", "clarifying_dialogue", "local_curated_core", local_source_refs[1], [
        [
            ("User", "I want to have a better week."),
            ("Assistant", "Sure. What would make the week feel better: less stress, more focus, or more rest?"),
            ("User", "Less stress and more focus."),
            ("Assistant", "Then start by cutting one optional task, protecting your first work block, and deciding tomorrow's first step tonight."),
        ],
        [
            ("User", "I need help planning tonight."),
            ("Assistant", "What kind of night do you need most: productive, restful, or a little of both?"),
            ("User", "A little of both."),
            ("Assistant", "Do one useful task first, then switch fully into a calm activity so the evening has a clear end point."),
        ],
        [
            ("User", "I want to write something but I keep avoiding it."),
            ("Assistant", "Is the problem starting, deciding what to say, or worrying that it won't be good?"),
            ("User", "Mostly starting."),
            ("Assistant", "Then give yourself a ten-minute draft window and write the roughest version first."),
        ],
        [
            ("User", "Can you help me choose what to do first?"),
            ("Assistant", "Yes. Is the pressure coming from urgency, importance, or mental clutter?"),
            ("User", "Mostly urgency."),
            ("Assistant", "Start with the task that has the closest real consequence, then reassess instead of planning the whole day at once."),
        ],
        [
            ("User", "I want a better routine."),
            ("Assistant", "What feels more unstable right now: mornings, work blocks, or evenings?"),
            ("User", "Mornings."),
            ("Assistant", "Then fix just the first fifteen minutes of the morning before trying to redesign the whole day."),
        ],
        [
            ("User", "I'm not sure how to ask for help."),
            ("Assistant", "Do you need emotional support, practical advice, or help with a task?"),
            ("User", "Practical advice."),
            ("Assistant", "Then describe the problem briefly, say what you've tried, and ask one clear question."),
        ],
        [
            ("User", "Can you help me clean up my notes?"),
            ("Assistant", "Are the notes mostly ideas, decisions, or tasks?"),
            ("User", "Mostly ideas and tasks."),
            ("Assistant", "Split them into two lists right away: ideas to revisit later and actions you can actually do next."),
        ],
        [
            ("User", "I need a calmer answer to a tense message."),
            ("Assistant", "Do you want to set a boundary, solve the problem, or just lower the temperature first?"),
            ("User", "Lower the temperature first."),
            ("Assistant", "Then acknowledge the concern, keep the wording simple, and avoid trying to settle every detail in one reply."),
        ],
        [
            ("User", "I want to plan a small project."),
            ("Assistant", "Is the biggest problem the idea itself or turning it into steps?"),
            ("User", "Turning it into steps."),
            ("Assistant", "Define the smallest finished version first, then list only the steps needed to reach that version."),
        ],
        [
            ("User", "I feel behind."),
            ("Assistant", "Behind on one important thing, or behind on everything at once?"),
            ("User", "Everything at once."),
            ("Assistant", "Then don't try to catch up globally. Pick the one area where movement will reduce the most pressure."),
        ],
        [
            ("User", "Can you help me make a short reading plan?"),
            ("Assistant", "Is the goal consistency, speed, or understanding?"),
            ("User", "Consistency."),
            ("Assistant", "Read for a small fixed time each day and stop while it still feels easy enough to repeat tomorrow."),
        ],
        [
            ("User", "I want to use a chatbot project as a demo."),
            ("Assistant", "Do you want to show technical correctness or recognizable conversational behavior?"),
            ("User", "Recognizable conversational behavior."),
            ("Assistant", "Then the dataset needs to teach short, direct replies more than package-specific factual coverage."),
        ],
        [
            ("User", "I want a tiny assistant that feels helpful."),
            ("Assistant", "Should it sound more like a friendly helper or more like a formal support bot?"),
            ("User", "A friendly helper."),
            ("Assistant", "Then keep the answers plain, short, and warm instead of sounding like documentation."),
        ],
        [
            ("User", "I'm choosing between more data cleanup and another width sweep."),
            ("Assistant", "What failed more clearly in the last run: the model capacity or the answer style?"),
            ("User", "The answer style."),
            ("Assistant", "Then improve the supervision signal first. A wider model won't fix low-quality reply targets."),
        ],
        [
            ("User", "I need a simple explanation, not a research talk."),
            ("Assistant", "Got it. What's the topic?"),
            ("User", "Why training data shape matters."),
            ("Assistant", "Because the model tends to imitate the structure and tone it sees most often during training."),
        ],
        [
            ("User", "Can you help me prepare for a short demo?"),
            ("Assistant", "Is the demo about capability, stability, or polish?"),
            ("User", "Capability and stability."),
            ("Assistant", "Then show a few reliable interactions and make the artifact path explicit instead of trying to impress with breadth."),
        ],
        [
            ("User", "I need help naming a small Julia chatbot demo."),
            ("Assistant", "Do you want the name to sound technical, friendly, or playful?"),
            ("User", "Friendly."),
            ("Assistant", "Then use a short, approachable name and avoid heavy platform jargon."),
        ],
        [
            ("User", "Can you help me recover after a bad draft?"),
            ("Assistant", "Do you want help revising it or help resetting emotionally first?"),
            ("User", "Revising it."),
            ("Assistant", "Then keep only the core idea, rewrite the opening, and treat the first draft as raw material instead of failure."),
        ],
    ]))

    append!(examples, multi_turn_examples("chatflow", "clarifying_dialogue", "local_curated_core", local_source_refs[1], [
        [
            ("User", "I need help sending a message, but I don't want it to sound cold."),
            ("Assistant", "Is the goal to sound warm, clear, or apologetic?"),
            ("User", "Warm and clear."),
            ("Assistant", "Then keep it short, name the request directly, and add one human sentence like 'I appreciate your time.'"),
        ],
        [
            ("User", "Can you help me plan tomorrow?"),
            ("Assistant", "Do you want a productive plan or a low-stress plan?"),
            ("User", "Low-stress, but still useful."),
            ("Assistant", "Pick one important task for the morning, one admin task for later, and leave the evening mostly open."),
        ],
        [
            ("User", "I don't know how to answer this question yet."),
            ("Assistant", "Do you need help thinking or help wording the reply?"),
            ("User", "Wording the reply."),
            ("Assistant", "Give a short honest status update, say what you do know, and name when you can follow up."),
        ],
        [
            ("User", "I want the chatbot demo to feel a little more natural."),
            ("Assistant", "Do you mean more conversational wording or better turn-taking?"),
            ("User", "Mostly better turn-taking."),
            ("Assistant", "Then the training data needs more short back-and-forth exchanges, not just isolated request-and-answer pairs."),
        ],
        [
            ("User", "I'm trying to reset after a messy day."),
            ("Assistant", "Do you want a practical reset or an emotional reset first?"),
            ("User", "Practical first."),
            ("Assistant", "Clear one surface, decide tomorrow's first step, and stop after those two things."),
        ],
        [
            ("User", "Can you help me ask a clearer question?"),
            ("Assistant", "Is the problem too broad, too vague, or too detailed?"),
            ("User", "Too broad."),
            ("Assistant", "Narrow it to one decision, one blocker, or one missing fact so the answer has somewhere to land."),
        ],
        [
            ("User", "I need help saying no without sounding rude."),
            ("Assistant", "Do you want a soft no or a firm no?"),
            ("User", "A soft no."),
            ("Assistant", "Thank them, decline plainly, and if useful offer a small alternative instead of a long explanation."),
        ],
        [
            ("User", "I need to explain a technical thing simply."),
            ("Assistant", "Is the listener technical or non-technical?"),
            ("User", "Non-technical."),
            ("Assistant", "Start with what the tool does for them, then mention one or two details only if they help the main point."),
        ],
        [
            ("User", "Can you help me plan a tiny project demo?"),
            ("Assistant", "Should the demo show polish or show progress?"),
            ("User", "Show progress."),
            ("Assistant", "Then show one clear before-and-after example and explain the workflow in plain language."),
        ],
        [
            ("User", "I want to feel less overwhelmed right now."),
            ("Assistant", "Would it help more to shrink the task or to choose the next step for you?"),
            ("User", "Shrink the task."),
            ("Assistant", "Make the task so small that you can begin it in under five minutes, then stop deciding and start moving."),
        ],
        [
            ("User", "Can you help me phrase a follow-up after no response?"),
            ("Assistant", "Do you want it to sound gentle or more direct?"),
            ("User", "Gentle."),
            ("Assistant", "Try a short check-in that repeats the request once and gives the other person room to reply when convenient."),
        ],
        [
            ("User", "How do I make an answer sound less robotic?"),
            ("Assistant", "Do you want it shorter, warmer, or more specific?"),
            ("User", "Warmer and more specific."),
            ("Assistant", "Use plain words, mention the user's actual situation, and avoid repeating generic helper phrases."),
        ],
        [
            ("User", "I need to reply to a message, but I feel defensive."),
            ("Assistant", "Do you want to calm the tone first or still make your point clearly?"),
            ("User", "Calm the tone first."),
            ("Assistant", "Acknowledge the concern, slow the wording down, and save the more detailed argument for later if needed."),
        ],
        [
            ("User", "Can you help me stop rambling when I ask for help?"),
            ("Assistant", "Is the harder part knowing the question or knowing what context to include?"),
            ("User", "Knowing what context to include."),
            ("Assistant", "Keep just three parts: the problem, what you tried, and the one thing you want help with."),
        ],
        [
            ("User", "I want a quick way to reset after getting distracted."),
            ("Assistant", "Would you rather restart the task or restart your attention first?"),
            ("User", "Restart my attention first."),
            ("Assistant", "Stand up, take one slow breath, and write the next visible action before you sit back down."),
        ],
        [
            ("User", "Can you help me make a small plan for this afternoon?"),
            ("Assistant", "Should it feel productive, restful, or balanced?"),
            ("User", "Balanced."),
            ("Assistant", "Pick one task that matters, one task that clears clutter, and one short break you can actually enjoy."),
        ],
        [
            ("User", "I need help saying what I mean without sounding harsh."),
            ("Assistant", "Is the message more about a request or a boundary?"),
            ("User", "A boundary."),
            ("Assistant", "State the limit plainly, avoid extra blame, and keep the focus on what you can or cannot do next."),
        ],
        [
            ("User", "I want the chatbot answers to feel less stiff."),
            ("Assistant", "Do you want warmer tone, shorter wording, or better turn-taking?"),
            ("User", "Shorter wording and better turn-taking."),
            ("Assistant", "Then the training examples should end sooner, use fewer lecture-style answers, and include more short follow-up turns."),
        ],
        [
            ("User", "Can you help me recover after I miss a deadline?"),
            ("Assistant", "Do you need a repair message or a recovery plan first?"),
            ("User", "A repair message first."),
            ("Assistant", "Acknowledge the miss clearly, say what changed, and give one realistic next date instead of vague promises."),
        ],
        [
            ("User", "I have too many ideas and not enough focus."),
            ("Assistant", "Do you need help choosing one idea or parking the rest?"),
            ("User", "Choosing one."),
            ("Assistant", "Pick the idea with the clearest next step and write the other ideas down so they stop competing for attention."),
        ],
        [
            ("User", "Can you help me phrase a softer correction?"),
            ("Assistant", "Do you want it to sound collaborative or just less sharp?"),
            ("User", "Collaborative."),
            ("Assistant", "Use wording like 'I think we may want to adjust this part' instead of sounding like you're issuing a verdict."),
        ],
        [
            ("User", "I want a calmer way to think about tomorrow."),
            ("Assistant", "Do you need a full plan or just a reassuring first step?"),
            ("User", "Just a reassuring first step."),
            ("Assistant", "Choose tomorrow's first task now and let the rest stay undecided until you have more energy."),
        ],
        [
            ("User", "Can you help me turn a rough idea into a question?"),
            ("Assistant", "Is the rough idea about a problem, a decision, or a plan?"),
            ("User", "A decision."),
            ("Assistant", "Then frame it around the tradeoff: ask what you gain, what you lose, and what makes the choice reversible."),
        ],
        [
            ("User", "I want the assistant to feel a bit more human."),
            ("Assistant", "Do you mean warmer wording or more natural back-and-forth?"),
            ("User", "More natural back-and-forth."),
            ("Assistant", "Then give it more short conversations where the second answer responds to a follow-up instead of just ending after one reply."),
        ],
    ]))

    metadata = Dict(
        "source_inputs" => local_source_refs,
        "counts_by_category" => count_by_category(examples),
        "total_examples" => length(examples),
    )

    return examples, metadata
end

function one_turn_examples(prefix::String, category::String, source_group::String, source_ref::String, pairs::Vector{Tuple{String,String}})
    examples = ChatExample[]
    for (index, (question, answer)) in enumerate(pairs)
        turns = [("User", question), ("Assistant", answer)]
        push!(examples, build_example("local_$(prefix)_$(lpad(index, 3, '0'))", source_group, source_ref, category, turns))
    end
    return examples
end

function multi_turn_examples(prefix::String, category::String, source_group::String, source_ref::String, dialogues::Vector{Vector{Tuple{String,String}}})
    examples = ChatExample[]
    for (index, turns) in enumerate(dialogues)
        push!(examples, build_example("local_$(prefix)_$(lpad(index, 3, '0'))", source_group, source_ref, category, turns))
    end
    return examples
end

function build_example(id::String, source_group::String, source_ref::String, category::String, turns::Vector{Tuple{String,String}})
    rendered_turns = String[]
    for (role, text) in turns
        normalized_text = normalize_chat_text(text)
        push!(rendered_turns, string(role, ": ", normalized_text))
    end
    if first(turns)[1] != "User" || last(turns)[1] != "Assistant"
        error("Each chat example must start with User and end with Assistant.")
    end
    chat_text = join(vcat(rendered_turns, END_ASSISTANT, CHAT_END), "\n")
    assistant_turns = count(turn -> turn[1] == "Assistant", turns)
    return ChatExample(id, id, source_group, source_ref, category, assistant_turns, chat_text)
end

function normalize_chat_text(text::AbstractString)::String
    cleaned = replace(text, "\r\n" => "\n", "\r" => "\n")
    cleaned = replace(cleaned, r"\s+" => " ")
    cleaned = strip(cleaned)
    endswith(cleaned, ('!', '.', '?')) || (cleaned *= ".")
    return cleaned
end

function build_external_oasst1_subset(output_directory::String)
    cache_directory = joinpath(output_directory, "cache")
    mkpath(cache_directory)
    gz_path = joinpath(cache_directory, "oasst1_ready_messages.jsonl.gz")

    try
        source_gz_path = ensure_oasst1_cache!(gz_path)
        examples, counts = parse_and_filter_oasst1(source_gz_path)
        status = Dict(
            "status" => "ok",
            "source" => "OpenAssistant/oasst1",
            "license" => OASST1_LICENSE,
            "url" => OASST1_READY_MESSAGES_URL,
            "cached_file" => source_gz_path,
            "selected_examples" => length(examples),
            "selected_single_turn_examples" => counts["single_turn"],
            "selected_multi_turn_examples" => counts["multi_turn"],
        )
        return examples, status
    catch error
        status = Dict(
            "status" => "failed",
            "source" => "OpenAssistant/oasst1",
            "license" => OASST1_LICENSE,
            "url" => OASST1_READY_MESSAGES_URL,
            "error" => sprint(showerror, error),
        )
        return ChatExample[], status
    end
end

function ensure_oasst1_cache!(preferred_path::String)::String
    isfile(preferred_path) && return preferred_path

    fallback_paths = [
        joinpath(repository_root(), "tmp", "tiny_chatbot_demo_dataset_v1", "cache", "oasst1_ready_messages.jsonl.gz"),
        joinpath(repository_root(), "tmp", "tiny_chatbot_demo_dataset_v2", "cache", "oasst1_ready_messages.jsonl.gz"),
        joinpath(repository_root(), "tmp", "tiny_chatbot_demo_dataset_v3", "cache", "oasst1_ready_messages.jsonl.gz"),
    ]
    for path in fallback_paths
        isfile(path) && return path
    end

    Downloads.download(OASST1_READY_MESSAGES_URL, preferred_path)
    return preferred_path
end

function parse_and_filter_oasst1(gz_path::String)
    records = Dict{String,Dict{String,Any}}()

    open(`gzip -dc $gz_path`, "r") do io
        for line in eachline(io)
            isempty(strip(line)) && continue
            object = JSON3.read(line)
            message_id = string(get(object, :message_id, ""))
            isempty(message_id) && continue
            records[message_id] = Dict(
                "message_id" => message_id,
                "parent_id" => string_or_empty(get(object, :parent_id, nothing)),
                "text" => string(get(object, :text, "")),
                "role" => lowercase(string(get(object, :role, ""))),
                "lang" => lowercase(string(get(object, :lang, ""))),
                "review_result" => bool_or_false(get(object, :review_result, false)),
                "deleted" => bool_or_false(get(object, :deleted, false)),
                "rank" => int_or_missing(get(object, :rank, nothing)),
                "synthetic" => bool_or_false(get(object, :synthetic, false)),
                "message_tree_id" => string_or_empty(get(object, :message_tree_id, nothing)),
                "labels" => extract_label_names(get(object, :labels, Any[])),
            )
        end
    end

    single_turn_candidates = ChatExample[]
    multi_turn_candidates = ChatExample[]

    for record in values(records)
        record["role"] == "assistant" || continue
        valid_external_record(record) || continue

        parent = parent_record(records, record)
        parent === nothing && continue
        prompt = normalize_external_text(parent["text"])
        answer = normalize_external_text(record["text"])
        single_turn_ok = external_pair_usable(prompt, answer)
        multi_turn_start_ok = external_conversation_start_usable(prompt, answer)
        (single_turn_ok || multi_turn_start_ok) || continue

        id = "oasst1_single_" * record["message_id"]
        tree_id = isempty(record["message_tree_id"]) ? id : "oasst1_tree_" * record["message_tree_id"]
        if single_turn_ok
            push!(single_turn_candidates, ChatExample(
                id,
                tree_id,
                "oasst1_filtered_single_turn",
                "OpenAssistant/oasst1:2023-04-12_oasst_ready.messages.jsonl.gz",
                "general_assistant",
                1,
                join([
                    "User: " * prompt,
                    "Assistant: " * answer,
                    END_ASSISTANT,
                    CHAT_END,
                ], "\n"),
            ))
        end

        multi_turn_start_ok || continue
        follow_up_user = best_child(records, record, "prompter")
        follow_up_user === nothing && continue
        second_assistant = best_child(records, follow_up_user, "assistant")
        second_assistant === nothing && continue
        valid_external_record(second_assistant) || continue

        follow_up_prompt = normalize_external_text(follow_up_user["text"])
        follow_up_answer = normalize_external_text(second_assistant["text"])
        external_follow_up_usable(follow_up_prompt, follow_up_answer) || continue

        follow_up_id = "oasst1_multi_" * follow_up_user["message_id"] * "_" * second_assistant["message_id"]
        push!(multi_turn_candidates, ChatExample(
            follow_up_id,
            tree_id,
            "oasst1_filtered_multi_turn",
            "OpenAssistant/oasst1:2023-04-12_oasst_ready.messages.jsonl.gz",
            "general_assistant",
            2,
            join([
                "User: " * prompt,
                "Assistant: " * answer,
                "User: " * follow_up_prompt,
                "Assistant: " * follow_up_answer,
                END_ASSISTANT,
                CHAT_END,
            ], "\n"),
        ))
    end

    sort!(single_turn_candidates; by = example -> sha1_hex(example.id))
    sort!(multi_turn_candidates; by = example -> sha1_hex(example.id))

    selected_single_turn = first(single_turn_candidates, min(TARGET_EXTERNAL_SINGLE_TURN_MAX, length(single_turn_candidates)))
    selected_multi_turn = first(multi_turn_candidates, min(TARGET_EXTERNAL_MULTI_TURN_MAX, length(multi_turn_candidates)))
    return vcat(selected_single_turn, selected_multi_turn), Dict(
        "single_turn" => length(selected_single_turn),
        "multi_turn" => length(selected_multi_turn),
    )
end

function extract_label_names(raw_labels)::Vector{String}
    names = String[]
    for label in raw_labels
        name = if label isa Pair
            first(label) == :name ? string(last(label)) : string(first(label))
        else
            try
                string(get(label, :name, ""))
            catch
                ""
            end
        end
        isempty(name) || push!(names, lowercase(name))
    end
    return unique(names)
end

function normalize_external_text(text::AbstractString)::String
    cleaned = replace(text, "\r\n" => "\n", "\r" => "\n")
    cleaned = replace(cleaned, r"\s+" => " ")
    cleaned = strip(cleaned)
    cleaned = replace(cleaned, "``" => "")
    cleaned = replace(cleaned, '`' => "")
    return cleaned
end

function valid_external_record(record::Dict{String,Any})::Bool
    record["lang"] == "en" || return false
    record["review_result"] || return false
    record["deleted"] && return false
    record["synthetic"] && return false
    rank = record["rank"]
    (rank === missing || rank <= MAX_ACCEPTED_OASST_RANK) || return false
    return true
end

function parent_record(records::Dict{String,Dict{String,Any}}, record::Dict{String,Any})
    parent_id = record["parent_id"]
    isempty(parent_id) && return nothing
    haskey(records, parent_id) || return nothing
    parent = records[parent_id]
    parent["role"] == "prompter" || return nothing
    valid_external_record(parent) || return nothing
    return parent
end

function best_child(records::Dict{String,Dict{String,Any}}, record::Dict{String,Any}, role::String)
    candidates = Dict{String,Any}[]
    for candidate in values(records)
        candidate["parent_id"] == record["message_id"] || continue
        candidate["role"] == role || continue
        valid_external_record(candidate) || continue
        push!(candidates, candidate)
    end
    isempty(candidates) && return nothing
    sort!(candidates; by = candidate -> (
        candidate["rank"] === missing ? 0 : Int(candidate["rank"]),
        sha1_hex(candidate["message_id"]),
    ))
    return first(candidates)
end

function external_pair_usable(prompt::String, answer::String)::Bool
    return external_conversation_start_usable(prompt, answer)
end

function external_conversation_start_usable(prompt::String, answer::String)::Bool
    prompt_words = length(split(prompt))
    answer_words = length(split(answer))
    prompt_chars = length(prompt)
    answer_chars = length(answer)

    1 <= prompt_words <= 48 || return false
    3 <= answer_words <= 120 || return false
    4 <= prompt_chars <= 320 || return false
    12 <= answer_chars <= 520 || return false

    occursin('\n', prompt) && return false
    count(==('\n'), answer) > 2 && return false
    count(==(':'), answer) > 4 && return false
    occursin("http://", prompt) && return false
    occursin("https://", prompt) && return false
    occursin("http://", answer) && return false
    occursin("https://", answer) && return false
    occursin("```", answer) && return false
    occursin("<|", answer) && return false
    occursin('♪', prompt) && return false
    occursin('♪', answer) && return false
    startswith(lowercase(answer), "as an ai") && return false
    startswith(lowercase(answer), "openassistant") && return false

    combined = lowercase(prompt * " " * answer)
    lowered_prompt = lowercase(prompt)
    startswith(lowered_prompt, "very thorough article") && return false
    startswith(lowered_prompt, "good explanation") && return false
    startswith(lowered_prompt, "which do you think is more likely") && return false
    startswith(lowered_prompt, "what are some themes in") && return false
    startswith(lowered_prompt, "write some words that rhyme") && return false
    startswith(lowered_prompt, "who are you") && return false
    startswith(lowered_prompt, "what is your name") && return false

    reject_keywords = [
        "roleplay", "pretend you are", "write a story", "short story", "fanfiction",
        "erotic", "sexual", "sex scene", "nsfw", "nude", "porn",
        "suicide", "self-harm", "kill myself", "overdose",
        "make a bomb", "bomb-making", "weapon", "stab", "poison", "murder",
        "malware", "phishing", "ransomware", "exploit", "steal passwords",
        "meth", "cocaine", "heroin", "drug recipe",
        "hate speech", "racial slur", "ethnic cleansing",
        "jailbreak", "ignore previous instructions",
        "citation", "citations", "research paper", "bibliography", "essay", "thesis",
        "according to", "historically", "in economics", "in philosophy",
        "political", "election", "legal advice", "diagnosis", "medical advice",
        "world cup", "capital of", "population of", "what year", "who won",
        "spongebob", "my liege", "your eminence", "pheasant and ale",
        "mermaid", "wabbit", "duck season",
        "css3", "browser version", "priority queue",
        "java", "c#", "c ++", "diablo", "elden ring",
        "windows defender", "mcaffe", "mcafee",
        "limerick", "rhyme with",
        "the force", "skywalker", "jedi",
    ]
    any(keyword -> occursin(keyword, combined), reject_keywords) && return false

    technical_keywords = [
        "python", "javascript", "sql", "regex", "docker", "kubernetes",
        "proof of", "theorem", "derivative of", "c++", "java code",
    ]
    any(keyword -> occursin(keyword, combined), technical_keywords) && return false

    weird_markers = [
        "###", "====", "lorem ipsum", "[[", "]]", "{", "}", "1.", "2.", "3.",
    ]
    count(marker -> occursin(marker, answer), weird_markers) >= 3 && return false

    return true
end

function external_follow_up_usable(prompt::String, answer::String)::Bool
    prompt_words = length(split(prompt))
    answer_words = length(split(answer))
    prompt_chars = length(prompt)
    answer_chars = length(answer)

    1 <= prompt_words <= 32 || return false
    2 <= answer_words <= 88 || return false
    3 <= prompt_chars <= 220 || return false
    8 <= answer_chars <= 380 || return false
    occursin('\n', prompt) && return false
    count(==('\n'), answer) > 2 && return false
    count(==(':'), answer) > 4 && return false
    occursin("http://", prompt) && return false
    occursin("https://", prompt) && return false
    occursin("http://", answer) && return false
    occursin("https://", answer) && return false
    occursin("```", answer) && return false
    occursin("<|", answer) && return false
    occursin('♪', prompt) && return false
    occursin('♪', answer) && return false

    combined = lowercase(prompt * " " * answer)
    follow_up_rejects = [
        "another question", "new topic", "different topic",
        "tell me a joke", "riddle", "quiz", "trivia",
        "world cup", "capital of", "who won", "diablo", "elden ring",
        "java", "css3", "browser", "priority queue",
        "windows defender", "mcafee",
        "limerick", "rhyme with", "fever", "temperature",
        "open assistant", "general knowledge", "the force", "skywalker", "jedi",
    ]
    any(keyword -> occursin(keyword, combined), follow_up_rejects) && return false
    return true
end


function deduplicate_examples(examples::Vector{ChatExample})
    seen = Dict{String,ChatExample}()
    for example in examples
        key = lowercase(strip(example.chat_text))
        haskey(seen, key) && continue
        seen[key] = example
    end
    deduped = collect(values(seen))
    sort!(deduped; by = example -> sha1_hex(example.id))
    return deduped
end

function split_examples(examples::Vector{ChatExample})
    ordered = sort(copy(examples); by = example -> sha1_hex(example.dialogue_group))
    n = length(ordered)
    n_train = floor(Int, 0.8 * n)
    n_validation = floor(Int, 0.1 * n)
    n_test = n - n_train - n_validation
    training = ordered[1:n_train]
    validation = ordered[(n_train + 1):(n_train + n_validation)]
    testing = ordered[(n_train + n_validation + 1):(n_train + n_validation + n_test)]
    return training, validation, testing
end

function build_metadata(examples, training, validation, testing, local_metadata, external_status)
    counts_by_source = Dict(
        "local_curated_core" => count(example -> example.source_group == "local_curated_core", examples),
        "oasst1_filtered_single_turn" => count(example -> example.source_group == "oasst1_filtered_single_turn", examples),
        "oasst1_filtered_multi_turn" => count(example -> example.source_group == "oasst1_filtered_multi_turn", examples),
    )

    return Dict(
        "dataset_name" => "tiny_chatbot_real_corpus_v2",
        "intended_scope" => "Expanded real downloaded conversational corpus for a small GPT-style chatbot in Julia. Ordinary conversational help, light planning, clarification, rewrite support, harmless practical requests, and a small amount of Keemena flavor as a minor anchor rather than the main behavior.",
        "persona" => "Small, polite, plainspoken assistant with short, direct, helpful replies.",
        "format_policy" => Dict(
            "rendered_chat_format" => "User: ...\\nAssistant: ...\\n<END_ASSISTANT>\\n<CHAT_END>",
            "assistant_end_marker" => END_ASSISTANT,
            "chat_end_marker" => CHAT_END,
            "multi_turn_policy" => "Short multi-turn dialogues are kept coherent in the same explicit chat format and grouped together by dialogue id for splitting.",
        ),
        "split_policy" => Dict(
            "method" => "Deterministic stable SHA1 ordering by dialogue_group, then fixed 80/10/10 split.",
            "train_fraction" => 0.8,
            "validation_fraction" => 0.1,
            "test_fraction" => 0.1,
        ),
        "local_source_inputs" => local_metadata["source_inputs"],
        "external_source_inputs" => [
            Dict(
                "name" => "OpenAssistant/oasst1",
                "file" => "2023-04-12_oasst_ready.messages.jsonl.gz",
                "license" => OASST1_LICENSE,
                "url" => OASST1_READY_MESSAGES_URL,
                "status" => external_status["status"],
            ),
        ],
        "external_filtering_rules" => [
            "English only",
            "review_result=true, not deleted, not synthetic, and accept top reviewed alternatives up to rank 4 to leave the tiny regime while staying in a reviewed band",
            "extract both single-turn and short multi-turn assistant dialogues from OASST trees",
            "single-turn slice uses a moderately widened conversation-start filter; multi-turn slice uses a softer follow-up filter so turn-taking remains a material share",
            "short to medium prompt and answer lengths with low formatting noise, widened slightly from v1 to expand ordinary assistant coverage",
            "reject roleplay-heavy, fiction-heavy, sexual content, self-harm, violence, illegal-help, hateful content, policy-heavy shapes, citation-heavy shapes, and code-helpdesk-heavy shapes",
            "reject trivia-heavy, browser/version lookup, theatrical/fandom, and other oddball examples surfaced during inspection",
            "reject URL-heavy, overly long, or messy examples",
            "keep OASST1 dominant and keep the local anchor small",
        ],
        "dataset_counts" => Dict(
            "total" => length(examples),
            "training" => length(training),
            "validation" => length(validation),
            "testing" => length(testing),
        ),
        "counts_by_source_group" => counts_by_source,
        "counts_by_category" => count_by_category(examples),
        "multi_turn_count" => count(example -> example.turn_count > 1, examples),
        "average_turns_per_example" => round(sum(example.turn_count for example in examples) / max(length(examples), 1); digits = 2),
        "quality_bar" => [
            "no docs dumps",
            "no duplicate questions with conflicting answers in the local curated core",
            "assistant replies should sound like replies, not notes",
            "local curated core must remain visibly present",
        ],
        "external_fetch_status" => external_status,
    )
end

function write_dataset(output_directory, training, validation, testing, metadata)
    mkpath(output_directory)
    write_split(output_directory, "training", training)
    write_split(output_directory, "validation", validation)
    write_split(output_directory, "testing", testing)
    open(joinpath(output_directory, "metadata.json"), "w") do io
        write(io, JSON3.write(metadata; allow_inf = false))
    end
end

function write_split(output_directory, name, examples)
    open(joinpath(output_directory, string(name, ".txt")), "w") do io
        write(io, join([example.chat_text for example in examples], "\n\n"))
        isempty(examples) || write(io, "\n")
    end

    open(joinpath(output_directory, string(name, ".jsonl")), "w") do io
        for example in examples
            row = Dict(
                "id" => example.id,
                "dialogue_group" => example.dialogue_group,
                "source_group" => example.source_group,
                "source_ref" => example.source_ref,
                "category" => example.category,
                "turn_count" => example.turn_count,
                "chat_text" => example.chat_text,
            )
            write(io, JSON3.write(row))
            write(io, "\n")
        end
    end
end

function count_by_category(examples)::Dict{String,Int}
    counts = Dict{String,Int}()
    for example in examples
        counts[example.category] = get(counts, example.category, 0) + 1
    end
    return Dict(sort(collect(counts); by = first))
end

function sha1_hex(text::AbstractString)::String
    return bytes2hex(sha1(String(text)))
end

string_or_empty(value) = value === nothing ? "" : string(value)
bool_or_false(value) = value === nothing ? false : Bool(value)
int_or_missing(value) = value === nothing ? missing : Int(value)

main(ARGS)

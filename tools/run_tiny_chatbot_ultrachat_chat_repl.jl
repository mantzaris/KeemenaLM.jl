#!/usr/bin/env julia

using KeemenaLM
using KeemenaSubwords

const DEFAULT_RUN_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_ultrachat_subword_candidate_run_v1")

Base.@kwdef struct ChatReplSettings
    run_dir::String = DEFAULT_RUN_DIR
    bundle_dir::String = ""
    tokenizer_bundle_dir::String = ""
    max_new_tokens::Int = 160
    temperature::Float64 = 0.7
    top_k::Int = 40
    top_p::Float64 = 0.95
    seed::Union{Nothing,Int} = 20260419
    system_prompt::String = ""
end

KeemenaLM.Core.tokenizer_encode(tokenizer::KeemenaSubwords.AbstractSubwordTokenizer, text::AbstractString) =
    KeemenaSubwords.encode(tokenizer, text; add_special_tokens = false)

KeemenaLM.Core.tokenizer_decode(
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer,
    token_ids::AbstractVector{<:Integer},
) = KeemenaSubwords.decode(tokenizer, Int[Int(token_id) for token_id in token_ids])

function main(args)
    settings = parse_args(args)
    run_chat_repl(settings)
end

function parse_args(args)::ChatReplSettings
    run_dir = DEFAULT_RUN_DIR
    bundle_dir = ""
    tokenizer_bundle_dir = ""
    max_new_tokens = 160
    temperature = 0.7
    top_k = 40
    top_p = 0.95
    seed = 20260419
    system_prompt = ""

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_usage()
            exit(0)
        elseif argument == "--run-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --run-dir")
            run_dir = abspath(args[argument_index])
        elseif argument == "--bundle-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --bundle-dir")
            bundle_dir = abspath(args[argument_index])
        elseif argument == "--tokenizer-bundle-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --tokenizer-bundle-dir")
            tokenizer_bundle_dir = abspath(args[argument_index])
        elseif argument == "--max-new-tokens"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --max-new-tokens")
            max_new_tokens = parse(Int, args[argument_index])
        elseif argument == "--temperature"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --temperature")
            temperature = parse(Float64, args[argument_index])
        elseif argument == "--top-k"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --top-k")
            top_k = parse(Int, args[argument_index])
        elseif argument == "--top-p"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --top-p")
            top_p = parse(Float64, args[argument_index])
        elseif argument == "--seed"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --seed")
            seed = parse(Int, args[argument_index])
        elseif argument == "--no-seed"
            seed = nothing
        elseif argument == "--system-prompt"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --system-prompt")
            system_prompt = String(args[argument_index])
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end

        argument_index += 1
    end

    max_new_tokens >= 0 || throw(ArgumentError("--max-new-tokens must be >= 0"))
    temperature >= 0.0 || throw(ArgumentError("--temperature must be >= 0.0"))
    top_k >= 0 || throw(ArgumentError("--top-k must be >= 0"))
    0.0 < top_p <= 1.0 || throw(ArgumentError("--top-p must be in (0.0, 1.0]"))

    return ChatReplSettings(
        run_dir = run_dir,
        bundle_dir = bundle_dir,
        tokenizer_bundle_dir = tokenizer_bundle_dir,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_k = top_k,
        top_p = top_p,
        seed = seed,
        system_prompt = system_prompt,
    )
end

function print_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_ultrachat_chat_repl.jl [options]

Options:
  --run-dir DIR                 Candidate run directory. Defaults to tmp/tiny_chatbot_ultrachat_subword_candidate_run_v1.
  --bundle-dir DIR              Model bundle directory. Defaults to RUN_DIR/bundle.
  --tokenizer-bundle-dir DIR    KeemenaSubwords tokenizer bundle. Defaults to RUN_DIR/tokenizer_bundle.
  --max-new-tokens N            Tokens per reply. Defaults to 160.
  --temperature X               Sampling temperature. Use 0.0 for greedy. Defaults to 0.7.
  --top-k N                     Sampling top-k. Defaults to 40.
  --top-p X                     Sampling top-p. Defaults to 0.95.
  --seed N                      Deterministic sampling seed. Defaults to 20260419.
  --no-seed                     Use a time-based sampling seed.
  --system-prompt TEXT          Optional system prompt. Empty by default because the corpus did not train on system turns.
""")
end

function run_chat_repl(settings::ChatReplSettings)
    run_dir = abspath(settings.run_dir)
    bundle_dir = isempty(settings.bundle_dir) ? joinpath(run_dir, "bundle") : abspath(settings.bundle_dir)
    tokenizer_bundle_dir = isempty(settings.tokenizer_bundle_dir) ? joinpath(run_dir, "tokenizer_bundle") : abspath(settings.tokenizer_bundle_dir)

    isdir(bundle_dir) || throw(ArgumentError("bundle directory does not exist: $(bundle_dir)"))
    isdir(tokenizer_bundle_dir) || throw(ArgumentError("tokenizer bundle directory does not exist: $(tokenizer_bundle_dir)"))

    model = load_model(bundle_dir; backend = :flux)
    tokenizer = KeemenaSubwords.load_training_bundle(tokenizer_bundle_dir)
    generation_config = GenerationConfig(
        max_new_tokens = settings.max_new_tokens,
        temperature = settings.temperature,
        top_k = settings.top_k,
        top_p = settings.top_p,
        seed = settings.seed,
    )
    session = ChatSession(
        model,
        tokenizer,
        nothing;
        system_prompt = settings.system_prompt,
        generation_config = generation_config,
    )

    println("Loaded UltraChat candidate bundle: ", bundle_dir)
    println("Loaded tokenizer bundle: ", tokenizer_bundle_dir)
    println("Type /exit or /quit to leave the REPL.")
    chat_repl(session)
    return session
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

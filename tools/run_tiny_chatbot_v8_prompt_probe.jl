#!/usr/bin/env julia

using KeemenaLM
using KeemenaSubwords

const DEFAULT_RUN_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v9_broad_336m_run")
const V8_PROBE_STOP_SEQUENCES = String[
    "<END_ASSISTANT>",
    "<CHAT_END>",
    "\nUser:",
    "\nAssistant:",
    "\nSystem:",
]

Base.@kwdef struct V8PromptProbeSettings
    run_dir::String = DEFAULT_RUN_DIR
    model_key::String = ""
    bundle_dir::String = ""
    tokenizer_bundle_dir::String = ""
    prompts::Vector{String} = String[]
    max_new_tokens::Int = 120
    temperature::Float64 = 0.0
    top_k::Int = 0
    top_p::Float64 = 1.0
    seed::Union{Nothing,Int} = 20260419
    device::Symbol = :auto
end

KeemenaLM.Core.tokenizer_encode(tokenizer::KeemenaSubwords.AbstractSubwordTokenizer, text::AbstractString) =
    KeemenaSubwords.encode(tokenizer, text; add_special_tokens = false)

KeemenaLM.Core.tokenizer_decode(
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer,
    token_ids::AbstractVector{<:Integer},
) = KeemenaSubwords.decode(tokenizer, Int[Int(token_id) for token_id in token_ids])

function main(args)
    settings = parse_args(args)
    run_probe(settings)
end

function parse_args(args)::V8PromptProbeSettings
    run_dir = DEFAULT_RUN_DIR
    model_key = ""
    bundle_dir = ""
    tokenizer_bundle_dir = ""
    prompts = String[]
    max_new_tokens = 120
    temperature = 0.0
    top_k = 0
    top_p = 1.0
    seed = 20260419
    device = :auto

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_usage()
            exit(0)
        elseif argument == "--model-key"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --model-key")
            model_key = String(args[argument_index])
        elseif argument == "--run-dir"
            argument_index += 1
            run_dir = abspath(args[argument_index])
        elseif argument == "--bundle-dir"
            argument_index += 1
            bundle_dir = abspath(args[argument_index])
        elseif argument == "--tokenizer-bundle-dir"
            argument_index += 1
            tokenizer_bundle_dir = abspath(args[argument_index])
        elseif argument == "--prompt"
            argument_index += 1
            push!(prompts, String(args[argument_index]))
        elseif argument == "--max-new-tokens"
            argument_index += 1
            max_new_tokens = parse(Int, args[argument_index])
        elseif argument == "--temperature"
            argument_index += 1
            temperature = parse(Float64, args[argument_index])
        elseif argument == "--top-k"
            argument_index += 1
            top_k = parse(Int, args[argument_index])
        elseif argument == "--top-p"
            argument_index += 1
            top_p = parse(Float64, args[argument_index])
        elseif argument == "--seed"
            argument_index += 1
            seed = parse(Int, args[argument_index])
        elseif argument == "--no-seed"
            seed = nothing
        elseif argument == "--device"
            argument_index += 1
            device = parse_device(args[argument_index])
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end
        argument_index += 1
    end

    if isempty(prompts)
        prompts = String[
            "hello",
            "what is 1 plus 1?",
            "what is 1+1?",
            "what is 2 plus 2?",
            "what is 2+2?",
            "is green a color?",
            "is blue a color?",
            "what is the capital of France?",
            "what is the capital of Japan?",
            "my name is Alex. what is my name?",
            "rewrite this kindly: You forgot again.",
            "give me two simple dinner ideas.",
        ]
    end

    return V8PromptProbeSettings(
        run_dir = run_dir,
        model_key = model_key,
        bundle_dir = bundle_dir,
        tokenizer_bundle_dir = tokenizer_bundle_dir,
        prompts = prompts,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_k = top_k,
        top_p = top_p,
        seed = seed,
        device = device,
    )
end

function parse_device(argument::AbstractString)::Symbol
    device = Symbol(lowercase(argument))
    device in (:cpu, :gpu, :auto) || throw(ArgumentError("--device must be cpu, gpu, or auto"))
    return device
end

function print_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_prompt_probe.jl [options]

Options:
  --run-dir DIR                 Candidate run directory. Defaults to the current v9 broad run.
  --model-key KEY              Official model artifact key, e.g. tiny-chatbot-v9-broad-336m.
  --bundle-dir DIR              Model bundle directory. Defaults to RUN_DIR/best_behavior_bundle if present, else RUN_DIR/bundle.
  --tokenizer-bundle-dir DIR    KeemenaSubwords tokenizer bundle. Defaults to RUN_DIR/tokenizer_bundle.
  --prompt TEXT                 Prompt to test. Can be repeated. Defaults to a small probe suite.
  --max-new-tokens N            Tokens per reply. Defaults to 120.
  --temperature X               Sampling temperature. Defaults to 0.0.
  --top-k N                     Sampling top-k. Defaults to 0.
  --top-p X                     Sampling top-p. Defaults to 1.0.
  --device cpu|gpu|auto         Inference device. Defaults to auto.
""")
end

function run_probe(settings::V8PromptProbeSettings)
    if !isempty(settings.model_key)
        run_dir = resolve_model_artifact(settings.model_key)
        bundle_dir = isempty(settings.bundle_dir) ? resolve_bundle(settings.model_key) : abspath(settings.bundle_dir)
        tokenizer_bundle_dir = isempty(settings.tokenizer_bundle_dir) ? resolve_tokenizer_bundle(settings.model_key) : abspath(settings.tokenizer_bundle_dir)
    else
        run_dir = abspath(settings.run_dir)
        default_best_bundle_dir = joinpath(run_dir, "best_behavior_bundle")
        default_bundle_dir = isdir(default_best_bundle_dir) ? default_best_bundle_dir : joinpath(run_dir, "bundle")
        bundle_dir = isempty(settings.bundle_dir) ? default_bundle_dir : abspath(settings.bundle_dir)
        tokenizer_bundle_dir = isempty(settings.tokenizer_bundle_dir) ? joinpath(run_dir, "tokenizer_bundle") : abspath(settings.tokenizer_bundle_dir)
    end

    isdir(bundle_dir) || throw(ArgumentError("bundle directory does not exist: $(bundle_dir)"))
    isdir(tokenizer_bundle_dir) || throw(ArgumentError("tokenizer bundle directory does not exist: $(tokenizer_bundle_dir)"))

    model = load_model(bundle_dir; backend = :flux)
    model = KeemenaLM.FluxBackend.move_model_to_device(model; device = settings.device)
    tokenizer = KeemenaSubwords.load_training_bundle(tokenizer_bundle_dir)
    generation_config = GenerationConfig(
        max_new_tokens = settings.max_new_tokens,
        temperature = settings.temperature,
        top_k = settings.top_k,
        top_p = settings.top_p,
        seed = settings.seed,
        stop_sequences = copy(V8_PROBE_STOP_SEQUENCES),
    )

    println("bundle_dir: ", bundle_dir)
    println("tokenizer_bundle_dir: ", tokenizer_bundle_dir)
    println("device: ", settings.device)
    for prompt in settings.prompts
        prompt_text = "User: " * prompt * "\nAssistant:"
        completion = KeemenaLM.Core.generate_completion(
            model,
            tokenizer,
            nothing,
            prompt_text;
            generation_config = generation_config,
        )
        println("---")
        println("user> ", prompt)
        println("assistant_raw> ", repr(completion))
        println("assistant> ", pretty_completion(completion))
    end
end

function pretty_completion(completion::AbstractString)::String
    cleaned = strip(replace(String(completion), r"\s+" => " "))
    for punctuation in (".", ",", "!", "?", ":", ";")
        cleaned = replace(cleaned, " " * punctuation => punctuation)
    end
    return cleaned
end

function command_allows_gpu_device(args)::Bool
    device = :auto
    argument_index = 1
    while argument_index <= length(args)
        if args[argument_index] == "--device"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --device")
            device = parse_device(args[argument_index])
        end
        argument_index += 1
    end
    return device !== :cpu
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !any(argument -> argument in ("--help", "-h"), ARGS)
        command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_gpu()
    end
    Base.invokelatest(main, ARGS)
end

#!/usr/bin/env julia

using JSON3
using KeemenaLM
using KeemenaSubwords

const DEFAULT_RUN_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_ultrachat_subword_candidate_run_v1")
const DEFAULT_OUTPUT_BASENAME = "sample_outputs_decoding_eval"
const CHAT_DECODING_STOP_SEQUENCES = String[
    "<END_ASSISTANT>",
    "<CHAT_END>",
    "\nUser:",
    "\nAssistant:",
    "\nSystem:",
]

Base.@kwdef struct DecodingEvalSettings
    run_dir::String = DEFAULT_RUN_DIR
    bundle_dir::String = ""
    tokenizer_bundle_dir::String = ""
    prompts_path::String = ""
    output_prefix::String = ""
    max_new_tokens::Int = 200
    prompt_limit::Int = 0
    seed::Int = 20260419
    sampling_seed::Int = 20260420
    sampling_temperature::Float64 = 0.7
    sampling_top_k::Int = 40
    sampling_top_p::Float64 = 0.95
    device::Symbol = :cpu
end

KeemenaLM.Core.tokenizer_encode(tokenizer::KeemenaSubwords.AbstractSubwordTokenizer, text::AbstractString) =
    KeemenaSubwords.encode(tokenizer, text; add_special_tokens = false)

KeemenaLM.Core.tokenizer_decode(
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer,
    token_ids::AbstractVector{<:Integer},
) = KeemenaSubwords.decode(tokenizer, Int[Int(token_id) for token_id in token_ids])

function main(args)
    settings = parse_args(args)
    payload, text_path, json_path = run_decoding_eval(settings)

    println("decoding evaluation completed")
    println("run_dir: ", payload["run_dir"])
    println("bundle_dir: ", payload["bundle_dir"])
    println("tokenizer_bundle_dir: ", payload["tokenizer_bundle_dir"])
    println("device: ", payload["device"])
    println("prompts: ", length(payload["prompts"]))
    println("text_report: ", text_path)
    println("json_report: ", json_path)
end

function parse_args(args)::DecodingEvalSettings
    run_dir = DEFAULT_RUN_DIR
    bundle_dir = ""
    tokenizer_bundle_dir = ""
    prompts_path = ""
    output_prefix = ""
    max_new_tokens = 200
    prompt_limit = 0
    seed = 20260419
    sampling_seed = 20260420
    sampling_temperature = 0.7
    sampling_top_k = 40
    sampling_top_p = 0.95
    device = :cpu

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
        elseif argument == "--prompts-path"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --prompts-path")
            prompts_path = abspath(args[argument_index])
        elseif argument == "--output-prefix"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --output-prefix")
            output_prefix = abspath(args[argument_index])
        elseif argument == "--max-new-tokens"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --max-new-tokens")
            max_new_tokens = parse(Int, args[argument_index])
        elseif argument == "--prompt-limit"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --prompt-limit")
            prompt_limit = parse(Int, args[argument_index])
        elseif argument == "--seed"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --seed")
            seed = parse(Int, args[argument_index])
        elseif argument == "--sampling-seed"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --sampling-seed")
            sampling_seed = parse(Int, args[argument_index])
        elseif argument == "--sampling-temperature"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --sampling-temperature")
            sampling_temperature = parse(Float64, args[argument_index])
        elseif argument == "--sampling-top-k"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --sampling-top-k")
            sampling_top_k = parse(Int, args[argument_index])
        elseif argument == "--sampling-top-p"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --sampling-top-p")
            sampling_top_p = parse(Float64, args[argument_index])
        elseif argument == "--device"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --device")
            device = parse_device(args[argument_index])
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end

        argument_index += 1
    end

    max_new_tokens >= 0 || throw(ArgumentError("--max-new-tokens must be >= 0"))
    prompt_limit >= 0 || throw(ArgumentError("--prompt-limit must be >= 0"))
    sampling_top_k >= 0 || throw(ArgumentError("--sampling-top-k must be >= 0"))
    0.0 < sampling_top_p <= 1.0 || throw(ArgumentError("--sampling-top-p must be in (0.0, 1.0]"))
    sampling_temperature >= 0.0 || throw(ArgumentError("--sampling-temperature must be >= 0.0"))

    return DecodingEvalSettings(
        run_dir = run_dir,
        bundle_dir = bundle_dir,
        tokenizer_bundle_dir = tokenizer_bundle_dir,
        prompts_path = prompts_path,
        output_prefix = output_prefix,
        max_new_tokens = max_new_tokens,
        prompt_limit = prompt_limit,
        seed = seed,
        sampling_seed = sampling_seed,
        sampling_temperature = sampling_temperature,
        sampling_top_k = sampling_top_k,
        sampling_top_p = sampling_top_p,
        device = device,
    )
end

function parse_device(argument::AbstractString)::Symbol
    device = Symbol(lowercase(argument))
    device in (:cpu, :gpu, :auto) ||
        throw(ArgumentError("--device must be cpu, gpu, or auto"))
    return device
end

function print_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_ultrachat_decoding_eval.jl [options]

Options:
  --run-dir DIR                 Candidate run directory. Defaults to tmp/tiny_chatbot_ultrachat_subword_candidate_run_v1.
  --bundle-dir DIR              Model bundle directory. Defaults to RUN_DIR/bundle.
  --tokenizer-bundle-dir DIR    KeemenaSubwords tokenizer bundle. Defaults to RUN_DIR/tokenizer_bundle.
  --prompts-path PATH           Prompt JSON/TXT file. Defaults to RUN_DIR/evaluation_prompts.json.
  --output-prefix PATH          Output prefix. Defaults to RUN_DIR/sample_outputs_decoding_eval.
  --max-new-tokens N            Tokens per sample per mode. Defaults to 200.
  --prompt-limit N              Limit prompts for smoke tests. 0 means all prompts.
  --seed N                      Greedy-mode seed. Defaults to 20260419.
  --sampling-seed N             Sampling-mode seed. Defaults to 20260420.
  --sampling-temperature X      Sampling temperature. Defaults to 0.7.
  --sampling-top-k N            Sampling top-k. Defaults to 40.
  --sampling-top-p X            Sampling top-p. Defaults to 0.95.
  --device cpu|gpu|auto         Inference device. Defaults to cpu.
""")
end

function run_decoding_eval(settings::DecodingEvalSettings)
    run_dir = abspath(settings.run_dir)
    bundle_dir = isempty(settings.bundle_dir) ? joinpath(run_dir, "bundle") : abspath(settings.bundle_dir)
    tokenizer_bundle_dir = isempty(settings.tokenizer_bundle_dir) ? joinpath(run_dir, "tokenizer_bundle") : abspath(settings.tokenizer_bundle_dir)
    prompts_path = isempty(settings.prompts_path) ? default_prompts_path(run_dir) : abspath(settings.prompts_path)
    output_prefix = isempty(settings.output_prefix) ? joinpath(run_dir, DEFAULT_OUTPUT_BASENAME) : abspath(settings.output_prefix)

    isdir(bundle_dir) || throw(ArgumentError("bundle directory does not exist: $(bundle_dir)"))
    isdir(tokenizer_bundle_dir) || throw(ArgumentError("tokenizer bundle directory does not exist: $(tokenizer_bundle_dir)"))
    isfile(prompts_path) || throw(ArgumentError("prompts file does not exist: $(prompts_path)"))

    model = load_model(bundle_dir; backend = :flux)
    model = KeemenaLM.FluxBackend.move_model_to_device(model; device = settings.device)
    tokenizer = KeemenaSubwords.load_training_bundle(tokenizer_bundle_dir)
    prompts = read_prompts(prompts_path)
    if settings.prompt_limit > 0
        prompts = prompts[1:min(settings.prompt_limit, length(prompts))]
    end

    modes = decoding_modes(settings)
    results = generate_decoding_results(model, tokenizer, prompts, modes)
    payload = Dict(
        "run_dir" => run_dir,
        "bundle_dir" => bundle_dir,
        "tokenizer_bundle_dir" => tokenizer_bundle_dir,
        "device" => String(settings.device),
        "prompts_path" => prompts_path,
        "prompts" => prompts,
        "stop_sequences" => CHAT_DECODING_STOP_SEQUENCES,
        "results" => results,
    )

    text_path = output_prefix * ".txt"
    json_path = output_prefix * ".json"
    mkpath(dirname(text_path))
    write_text_report(text_path, payload)
    write_json(json_path, payload)
    return payload, text_path, json_path
end

function default_prompts_path(run_dir::AbstractString)::String
    json_path = joinpath(run_dir, "evaluation_prompts.json")
    isfile(json_path) && return json_path

    text_path = joinpath(run_dir, "evaluation_prompts.txt")
    isfile(text_path) && return text_path

    return json_path
end

function read_prompts(path::AbstractString)::Vector{String}
    extension = lowercase(splitext(path)[2])
    if extension == ".json"
        prompt_data = JSON3.read(read(path, String))
        if hasproperty(prompt_data, :prompts)
            prompts = [String(prompt) for prompt in prompt_data.prompts]
        else
            prompts = try
                [String(prompt) for prompt in prompt_data]
            catch
                throw(ArgumentError("JSON prompts file must be an array or an object with a prompts field: $(path)"))
            end
        end
    else
        prompts = read_prompt_paragraphs(path)
    end

    prompts = strip.(prompts)
    filter!(prompt -> !isempty(prompt), prompts)
    isempty(prompts) && throw(ArgumentError("prompts file did not contain any prompts: $(path)"))
    return prompts
end

function read_prompt_paragraphs(path::AbstractString)::Vector{String}
    prompts = String[]
    buffer = IOBuffer()

    open(path, "r") do io
        while !eof(io)
            line = readline(io; keep = true)
            if isempty(strip(line))
                prompt = strip(String(take!(buffer)))
                isempty(prompt) || push!(prompts, prompt)
            else
                write(buffer, line)
            end
        end
    end

    prompt = strip(String(take!(buffer)))
    isempty(prompt) || push!(prompts, prompt)
    return prompts
end

function decoding_modes(settings::DecodingEvalSettings)
    return [
        (
            name = "greedy_fixed",
            description = "Current fixed-length greedy behavior with no chat stop sequences.",
            generation_config = GenerationConfig(
                max_new_tokens = settings.max_new_tokens,
                temperature = 0.0,
                seed = settings.seed,
            ),
        ),
        (
            name = "greedy_stop_controlled",
            description = "Greedy decoding with explicit chat marker and turn-boundary stop sequences.",
            generation_config = GenerationConfig(
                max_new_tokens = settings.max_new_tokens,
                temperature = 0.0,
                seed = settings.seed,
                stop_sequences = copy(CHAT_DECODING_STOP_SEQUENCES),
            ),
        ),
        (
            name = "mild_sampling_stop_controlled",
            description = "Mild sampling with the same explicit chat stop sequences.",
            generation_config = GenerationConfig(
                max_new_tokens = settings.max_new_tokens,
                temperature = settings.sampling_temperature,
                top_k = settings.sampling_top_k,
                top_p = settings.sampling_top_p,
                seed = settings.sampling_seed,
                stop_sequences = copy(CHAT_DECODING_STOP_SEQUENCES),
            ),
        ),
    ]
end

function generate_decoding_results(model, tokenizer, prompts::Vector{String}, modes)
    results = Dict{String,Any}[]
    for mode in modes
        samples = Dict{String,Any}[]
        for (prompt_index, prompt) in enumerate(prompts)
            _, completion = KeemenaLM.Core.generate_prompt_completion(
                model,
                tokenizer,
                nothing,
                prompt;
                generation_config = mode.generation_config,
            )
            push!(
                samples,
                Dict(
                    "prompt_index" => prompt_index,
                    "prompt" => prompt,
                    "completion" => completion,
                    "output" => prompt * completion,
                    "completion_characters" => length(completion),
                ),
            )
        end

        push!(
            results,
            Dict(
                "mode" => mode.name,
                "description" => mode.description,
                "generation_config" => generation_config_dict(mode.generation_config),
                "samples" => samples,
            ),
        )
    end

    return results
end

function generation_config_dict(config::GenerationConfig)
    return Dict(
        "max_new_tokens" => config.max_new_tokens,
        "temperature" => config.temperature,
        "top_k" => config.top_k,
        "top_p" => config.top_p,
        "seed" => config.seed,
        "eos_token_id" => config.eos_token_id,
        "stop_sequences" => config.stop_sequences,
    )
end

function write_text_report(path::AbstractString, payload)
    open(path, "w") do io
        println(io, "# Tiny Chatbot Decoding Evaluation")
        println(io)
        println(io, "run_dir: ", payload["run_dir"])
        println(io, "bundle_dir: ", payload["bundle_dir"])
        println(io, "tokenizer_bundle_dir: ", payload["tokenizer_bundle_dir"])
        println(io, "device: ", payload["device"])
        println(io, "prompts_path: ", payload["prompts_path"])
        println(io, "stop_sequences: ", repr(payload["stop_sequences"]))
        println(io)

        for result in payload["results"]
            println(io, "## ", result["mode"])
            println(io, result["description"])
            println(io, "generation_config: ", result["generation_config"])
            println(io)

            for sample in result["samples"]
                println(io, "prompt_index: ", sample["prompt_index"])
                println(io, "prompt>")
                println(io, sample["prompt"])
                println(io, "completion>")
                println(io, sample["completion"])
                println(io, "output>")
                println(io, sample["output"])
                println(io)
            end
        end
    end

    return path
end

function write_json(path::AbstractString, payload)
    open(path, "w") do io
        JSON3.write(io, payload)
    end
    return path
end

function command_allows_gpu_device(args)::Bool
    device = :cpu
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
        command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    end
    Base.invokelatest(main, ARGS)
end

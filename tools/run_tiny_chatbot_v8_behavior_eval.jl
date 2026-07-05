#!/usr/bin/env julia

using JSON3
using KeemenaLM
using KeemenaSubwords

const V8_BEHAVIOR_DEFAULT_RUN_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v9_broad_336m_run")
const V8_BEHAVIOR_STOP_SEQUENCES = String[
    "<END_ASSISTANT>",
    "<CHAT_END>",
    "\nUser:",
    "\nAssistant:",
    "\nSystem:",
]

Base.@kwdef struct V8BehaviorEvalSettings
    run_dir::String = V8_BEHAVIOR_DEFAULT_RUN_DIR
    model_key::String = ""
    bundle_dir::String = ""
    tokenizer_bundle_dir::String = ""
    output_path::String = ""
    max_new_tokens::Int = 100
    prompt_limit::Int = 0
    seed::Int = 20260615
    device::Symbol = :cpu
end

KeemenaLM.Core.tokenizer_encode(tokenizer::KeemenaSubwords.AbstractSubwordTokenizer, text::AbstractString) =
    KeemenaSubwords.encode(tokenizer, text; add_special_tokens = false)

KeemenaLM.Core.tokenizer_decode(
    tokenizer::KeemenaSubwords.AbstractSubwordTokenizer,
    token_ids::AbstractVector{<:Integer},
) = KeemenaSubwords.decode(tokenizer, Int[Int(token_id) for token_id in token_ids])

function main(args)
    settings = v8_parse_behavior_eval_args(args)
    payload = run_v8_behavior_eval(settings)
    println("v8 behavior evaluation completed")
    println("run_dir: ", payload["run_dir"])
    println("bundle_dir: ", payload["bundle_dir"])
    println("tokenizer_bundle_dir: ", payload["tokenizer_bundle_dir"])
    println("passed: ", payload["summary"]["passed"])
    println("pass_rate: ", payload["summary"]["pass_rate"])
    println("failed_case_ids: ", payload["summary"]["failed_case_ids"])
    println("output_path: ", payload["output_path"])
end

function v8_parse_behavior_eval_args(args)::V8BehaviorEvalSettings
    run_dir = V8_BEHAVIOR_DEFAULT_RUN_DIR
    model_key = ""
    bundle_dir = ""
    tokenizer_bundle_dir = ""
    output_path = ""
    max_new_tokens = 100
    prompt_limit = 0
    seed = 20260615
    device = :cpu

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_v8_behavior_usage()
            exit(0)
        elseif argument == "--model-key"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --model-key")
            model_key = String(args[argument_index])
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
        elseif argument == "--output-path"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --output-path")
            output_path = abspath(args[argument_index])
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
        elseif argument == "--device"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --device")
            device = v8_parse_device(args[argument_index])
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end
        argument_index += 1
    end

    max_new_tokens >= 0 || throw(ArgumentError("--max-new-tokens must be >= 0"))
    prompt_limit >= 0 || throw(ArgumentError("--prompt-limit must be >= 0"))

    return V8BehaviorEvalSettings(
        run_dir = run_dir,
        model_key = model_key,
        bundle_dir = bundle_dir,
        tokenizer_bundle_dir = tokenizer_bundle_dir,
        output_path = output_path,
        max_new_tokens = max_new_tokens,
        prompt_limit = prompt_limit,
        seed = seed,
        device = device,
    )
end

function print_v8_behavior_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_behavior_eval.jl [options]

Options:
  --run-dir DIR                 Candidate run directory. Defaults to tmp/tiny_chatbot_v9_broad_336m_run.
  --model-key KEY              Official model artifact key, e.g. tiny-chatbot-v9-broad-336m.
  --bundle-dir DIR              Model bundle directory. Defaults to RUN_DIR/bundle.
  --tokenizer-bundle-dir DIR    KeemenaSubwords tokenizer bundle. Defaults to RUN_DIR/tokenizer_bundle.
  --output-path PATH            JSON report path. Defaults to RUN_DIR/behavior_eval.json.
  --max-new-tokens N            Tokens per behavior prompt. Defaults to 100.
  --prompt-limit N              Limit behavior prompts for smoke tests. 0 means all prompts.
  --seed N                      Greedy generation seed. Defaults to 20260615.
  --device cpu|gpu|auto         Inference device. Defaults to cpu.
""")
end

function v8_parse_device(argument::AbstractString)::Symbol
    device = Symbol(lowercase(strip(argument)))
    device in (:cpu, :gpu, :auto) || throw(ArgumentError("device must be cpu, gpu, or auto"))
    return device
end

function run_v8_behavior_eval(settings::V8BehaviorEvalSettings)
    if !isempty(settings.model_key)
        run_dir = resolve_model_artifact(settings.model_key)
        bundle_dir = isempty(settings.bundle_dir) ? resolve_bundle(settings.model_key) : abspath(settings.bundle_dir)
        tokenizer_bundle_dir = isempty(settings.tokenizer_bundle_dir) ? resolve_tokenizer_bundle(settings.model_key) : abspath(settings.tokenizer_bundle_dir)
        default_output_path = joinpath(pwd(), "tmp", string(settings.model_key, "_behavior_eval.json"))
    else
        run_dir = abspath(settings.run_dir)
        bundle_dir = isempty(settings.bundle_dir) ? joinpath(run_dir, "bundle") : abspath(settings.bundle_dir)
        tokenizer_bundle_dir = isempty(settings.tokenizer_bundle_dir) ? joinpath(run_dir, "tokenizer_bundle") : abspath(settings.tokenizer_bundle_dir)
        default_output_path = joinpath(run_dir, "behavior_eval.json")
    end
    output_path = isempty(settings.output_path) ? default_output_path : abspath(settings.output_path)

    isdir(bundle_dir) || throw(ArgumentError("bundle directory does not exist: $(bundle_dir)"))
    isdir(tokenizer_bundle_dir) || throw(ArgumentError("tokenizer bundle directory does not exist: $(tokenizer_bundle_dir)"))

    model = load_model(bundle_dir; backend = :flux)
    model = KeemenaLM.FluxBackend.move_model_to_device(model; device = settings.device)
    tokenizer = KeemenaSubwords.load_training_bundle(tokenizer_bundle_dir)
    cases = KeemenaLM.Core.chatbot_behavior_cases()
    if settings.prompt_limit > 0
        cases = cases[1:min(settings.prompt_limit, length(cases))]
    end

    report = v8_score_model_behavior(
        model,
        tokenizer;
        cases = cases,
        max_new_tokens = settings.max_new_tokens,
        seed = settings.seed,
    )
    payload = merge(report, Dict(
        "run_dir" => run_dir,
        "bundle_dir" => bundle_dir,
        "tokenizer_bundle_dir" => tokenizer_bundle_dir,
        "device" => String(settings.device),
        "output_path" => output_path,
    ))
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        JSON3.write(io, payload)
    end
    return payload
end

function v8_score_model_behavior(
    model,
    tokenizer;
    cases = KeemenaLM.Core.chatbot_behavior_cases(),
    max_new_tokens::Int = 100,
    seed::Int = 20260615,
)
    generation_config = GenerationConfig(
        max_new_tokens = max_new_tokens,
        temperature = 0.0,
        seed = seed,
        stop_sequences = copy(V8_BEHAVIOR_STOP_SEQUENCES),
    )
    completions = Dict{String,String}()
    results = Dict{String,Any}[]

    for case in cases
        case_id = String(getfield(case, :id))
        _, completion = KeemenaLM.Core.generate_prompt_completion(
            model,
            tokenizer,
            nothing,
            String(getfield(case, :prompt));
            generation_config = generation_config,
        )
        completions[case_id] = completion
        score = KeemenaLM.Core.score_chatbot_behavior_completion(case, completion)
        push!(
            results,
            Dict(
                "case" => v8_behavior_case_dict(case),
                "completion" => completion,
                "score" => score,
            ),
        )
    end

    summary = KeemenaLM.Core.score_chatbot_behavior_suite(completions; cases = cases)
    return Dict(
        "summary" => summary,
        "results" => results,
        "stop_sequences" => V8_BEHAVIOR_STOP_SEQUENCES,
        "generation_config" => Dict(
            "max_new_tokens" => max_new_tokens,
            "temperature" => 0.0,
            "seed" => seed,
            "stop_sequences" => V8_BEHAVIOR_STOP_SEQUENCES,
        ),
    )
end

function v8_behavior_case_dict(case)
    return Dict(
        "id" => String(getfield(case, :id)),
        "prompt" => String(getfield(case, :prompt)),
        "required_any" => String[String(value) for value in getfield(case, :required_any)],
        "required_all" => String[String(value) for value in getfield(case, :required_all)],
        "forbidden" => String[String(value) for value in getfield(case, :forbidden)],
        "max_completion_characters" => Int(getfield(case, :max_completion_characters)),
    )
end

function v8_behavior_command_allows_gpu(args)::Bool
    device = :cpu
    argument_index = 1
    while argument_index <= length(args)
        if args[argument_index] == "--device"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --device")
            device = v8_parse_device(args[argument_index])
        end
        argument_index += 1
    end
    return device !== :cpu
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !any(argument -> argument in ("--help", "-h"), ARGS)
        v8_behavior_command_allows_gpu(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    end
    Base.invokelatest(main, ARGS)
end

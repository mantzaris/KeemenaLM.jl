#!/usr/bin/env julia

using KeemenaLM
using KeemenaSubwords

const DEFAULT_RUN_DIR = joinpath(pwd(), "tmp", "tiny_chatbot_v9_broad_336m_run")

const V8_CHAT_STOP_SEQUENCES = String[
    "<END_ASSISTANT>",
    "<CHAT_END>",
    "\nUser:",
    "\nAssistant:",
    "\nSystem:",
]

const V8ChatMessage = NamedTuple{(:role, :content), Tuple{String, String}}

Base.@kwdef mutable struct V8ChatReplSession
    model::KeemenaLM.Core.AbstractCausalLM
    tokenizer::Any
    generation_config::GenerationConfig
    system_prompt::String = ""
    stateful::Bool = false
    message_history::Vector{V8ChatMessage} = V8ChatMessage[]
end


Base.@kwdef struct ChatReplSettings
    run_dir::String = DEFAULT_RUN_DIR
    model_key::String = ""
    bundle_dir::String = ""
    tokenizer_bundle_dir::String = ""
    max_new_tokens::Int = 160
    temperature::Float64 = 0.7
    top_k::Int = 40
    top_p::Float64 = 0.95
    seed::Union{Nothing,Int} = 20260419
    system_prompt::String = ""
    stateful::Bool = false
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
    run_chat_repl(settings)
end

function parse_args(args)::ChatReplSettings
    run_dir = DEFAULT_RUN_DIR
    model_key = ""
    bundle_dir = ""
    tokenizer_bundle_dir = ""
    max_new_tokens = 160
    temperature = 0.7
    top_k = 40
    top_p = 0.95
    seed = 20260419
    system_prompt = ""
    stateful = false
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
        elseif argument == "--stateful"
            stateful = true
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
    temperature >= 0.0 || throw(ArgumentError("--temperature must be >= 0.0"))
    top_k >= 0 || throw(ArgumentError("--top-k must be >= 0"))
    0.0 < top_p <= 1.0 || throw(ArgumentError("--top-p must be in (0.0, 1.0]"))

    return ChatReplSettings(
        run_dir = run_dir,
        model_key = model_key,
        bundle_dir = bundle_dir,
        tokenizer_bundle_dir = tokenizer_bundle_dir,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_k = top_k,
        top_p = top_p,
        seed = seed,
        system_prompt = system_prompt,
        stateful = stateful,
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
usage: julia --project=tools/subword_real_text tools/run_tiny_chatbot_v8_chat_repl.jl [options]

Options:
  --run-dir DIR                 Candidate run directory. Defaults to tmp/tiny_chatbot_v9_broad_336m_run.
  --model-key KEY              Official model artifact key, e.g. tiny-chatbot-v9-broad-336m.
  --bundle-dir DIR              Model bundle directory. Defaults to RUN_DIR/bundle.
  --tokenizer-bundle-dir DIR    KeemenaSubwords tokenizer bundle. Defaults to RUN_DIR/tokenizer_bundle.
  --max-new-tokens N            Tokens per reply. Defaults to 160.
  --temperature X               Sampling temperature. Use 0.0 for greedy. Defaults to 0.7.
  --top-k N                     Sampling top-k. Defaults to 40.
  --top-p X                     Sampling top-p. Defaults to 0.95.
  --seed N                      Deterministic sampling seed. Defaults to 20260419.
  --no-seed                     Use a time-based sampling seed.
  --system-prompt TEXT          Optional system prompt. Empty by default because the corpus did not train on system turns.
  --stateful                    Keep prior turns in the prompt. Default is stateless one-turn prompts.
  --device cpu|gpu|auto         Inference device. Defaults to auto.
""")
end

function run_chat_repl(settings::ChatReplSettings)
    if !isempty(settings.model_key)
        run_dir = resolve_model_artifact(settings.model_key)
        bundle_dir = isempty(settings.bundle_dir) ? resolve_bundle(settings.model_key) : abspath(settings.bundle_dir)
        tokenizer_bundle_dir = isempty(settings.tokenizer_bundle_dir) ? resolve_tokenizer_bundle(settings.model_key) : abspath(settings.tokenizer_bundle_dir)
    else
        run_dir = abspath(settings.run_dir)
        bundle_dir = isempty(settings.bundle_dir) ? joinpath(run_dir, "bundle") : abspath(settings.bundle_dir)
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
        stop_sequences = copy(V8_CHAT_STOP_SEQUENCES),
    )
    session = V8ChatReplSession(
        model = model,
        tokenizer = tokenizer,
        generation_config = generation_config,
        system_prompt = settings.system_prompt,
        stateful = settings.stateful,
    )

    println("Loaded v8 chatbot candidate bundle: ", bundle_dir)
    println("Loaded tokenizer bundle: ", tokenizer_bundle_dir)
    println("Inference device: ", settings.device)
    println("Prompt template: v8 User:/Assistant:/<END_ASSISTANT>/<CHAT_END>")
    println("History mode: ", settings.stateful ? "stateful" : "stateless one-turn")
    println("Type /exit or /quit to leave the REPL. Type /reset to clear history in stateful mode.")
    v8_chat_repl(session)
    return session
end

function v8_chat_repl(
    session::V8ChatReplSession;
    input::IO = stdin,
    output::IO = stdout,
    user_prompt::AbstractString = "user> ",
    assistant_prompt::AbstractString = "assistant> ",
    exit_commands::AbstractVector{<:AbstractString} = ["/exit", "/quit"],
)
    while true
        print(output, user_prompt)
        flush(output)

        user_text = try
            readline(input)
        catch error_object
            error_object isa EOFError && break
            rethrow()
        end

        stripped_text = strip(user_text)
        stripped_text in exit_commands && break
        if stripped_text == "/reset"
            empty!(session.message_history)
            println(output, "history reset")
            flush(output)
            continue
        end
        isempty(stripped_text) && continue

        assistant_text = v8_chat!(session, user_text)
        println(output, string(assistant_prompt, assistant_text))
        flush(output)
    end

    return session
end

function v8_chat!(session::V8ChatReplSession, user_text::AbstractString)::String
    prompt_text = v8_render_chat_prompt(session; pending_user_text = user_text)
    raw_completion = KeemenaLM.Core.generate_completion(
        session.model,
        session.tokenizer,
        nothing,
        prompt_text;
        generation_config = session.generation_config,
    )
    assistant_text = v8_pretty_completion(raw_completion)
    if session.stateful
        push!(session.message_history, (role = "user", content = String(user_text)))
        push!(session.message_history, (role = "assistant", content = assistant_text))
    end
    return assistant_text
end

function v8_render_chat_prompt(session::V8ChatReplSession; pending_user_text::Union{Nothing, AbstractString} = nothing)::String
    prompt_io = IOBuffer()

    if !isempty(session.system_prompt)
        println(prompt_io, "System: ", session.system_prompt)
    end

    for message in session.message_history
        if message.role == "user"
            println(prompt_io, "User: ", message.content)
        elseif message.role == "assistant"
            println(prompt_io, "Assistant: ", message.content)
            println(prompt_io, "<END_ASSISTANT>")
            println(prompt_io, "<CHAT_END>")
        else
            throw(ArgumentError("unsupported chat role $(message.role)"))
        end
    end

    pending_user_text === nothing || println(prompt_io, "User: ", pending_user_text)
    print(prompt_io, "Assistant:")
    return String(take!(prompt_io))
end

function v8_pretty_completion(completion::AbstractString)::String
    cleaned = String(completion)
    for stop_sequence in V8_CHAT_STOP_SEQUENCES
        stop_index = findfirst(stop_sequence, cleaned)
        stop_index === nothing || (cleaned = cleaned[firstindex(cleaned):prevind(cleaned, first(stop_index))])
    end
    cleaned = strip(replace(cleaned, r"\s+" => " "))
    for punctuation in (".", ",", "!", "?", ":", ";")
        cleaned = replace(cleaned, " " * punctuation => punctuation)
    end
    cleaned = replace(cleaned, "( " => "(")
    cleaned = replace(cleaned, " )" => ")")
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
        command_allows_gpu_device(ARGS) && KeemenaLM.FluxBackend.has_functional_cuda_gpu()
    end
    Base.invokelatest(main, ARGS)
end

const ChatMessage = NamedTuple{(:role, :content), Tuple{String, String}}
const CHAT_ROLES = ("user", "assistant")
const DEFAULT_CHAT_STOP_SEQUENCES = ("\nUser:", "\nSystem:")

"""
Minimal in-memory chat wrapper around `generate`.
"""
Base.@kwdef mutable struct ChatSession
    model::AbstractCausalLM
    tokenizer::Any
    preprocessing::Any
    system_prompt::String = ""
    generation_config::GenerationConfig = GenerationConfig()
    message_history::Vector{ChatMessage} = ChatMessage[]
end

function ChatSession(
    model::AbstractCausalLM,
    tokenizer,
    preprocessing = nothing;
    system_prompt::AbstractString = "",
    generation_config::GenerationConfig = GenerationConfig(),
    message_history::AbstractVector = ChatMessage[],
)
    return ChatSession(
        model = model,
        tokenizer = tokenizer,
        preprocessing = preprocessing,
        system_prompt = String(system_prompt),
        generation_config = generation_config,
        message_history = normalize_chat_history(message_history),
    )
end

function chat!(session::ChatSession, user_text::AbstractString; overrides...)::String
    prompt_text = render_chat_prompt(session; pending_user_text = user_text)
    generation_config = chat_generation_config(session.generation_config; overrides...)
    assistant_text = generate_completion(
        session.model,
        session.tokenizer,
        session.preprocessing,
        prompt_text;
        generation_config = generation_config,
    )

    push_chat_message!(session, "user", user_text)
    push_chat_message!(session, "assistant", assistant_text)
    return assistant_text
end

function chat_repl(
    session::ChatSession;
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
        isempty(stripped_text) && continue

        assistant_text = chat!(session, user_text)
        println(output, string(assistant_prompt, assistant_text))
        flush(output)
    end

    return session
end

function normalize_chat_history(message_history::AbstractVector)::Vector{ChatMessage}
    return [normalize_chat_message(message) for message in message_history]
end

function normalize_chat_message(message)::ChatMessage
    role = String(getproperty(message, :role))
    content = String(getproperty(message, :content))
    validate_chat_role(role)
    return (role = role, content = content)
end

function validate_chat_role(role::AbstractString)::String
    role in CHAT_ROLES || throw(ArgumentError("unsupported chat role $(role); expected one of $(collect(CHAT_ROLES))"))
    return String(role)
end

function push_chat_message!(session::ChatSession, role::AbstractString, content::AbstractString)
    push!(session.message_history, (role = validate_chat_role(role), content = String(content)))
    return session
end

function chat_generation_config(base::GenerationConfig; overrides...)::GenerationConfig
    merged_config = merge_generation_config(base; overrides...)
    stop_sequences = String[sequence for sequence in merged_config.stop_sequences]
    for stop_sequence in DEFAULT_CHAT_STOP_SEQUENCES
        stop_sequence in stop_sequences || push!(stop_sequences, stop_sequence)
    end

    return GenerationConfig(
        max_new_tokens = merged_config.max_new_tokens,
        temperature = merged_config.temperature,
        top_k = merged_config.top_k,
        top_p = merged_config.top_p,
        seed = merged_config.seed,
        eos_token_id = merged_config.eos_token_id,
        stop_sequences = stop_sequences,
    )
end

function merge_generation_config(base::GenerationConfig; overrides...)::GenerationConfig
    isempty(overrides) && return base

    field_names = fieldnames(GenerationConfig)
    field_values = Dict{Symbol, Any}(field_name => getfield(base, field_name) for field_name in field_names)
    for (field_name, value) in pairs(overrides)
        field_name in field_names ||
            throw(ArgumentError("unknown generation override $(field_name); expected one of $(collect(field_names))"))
        field_values[field_name] = value
    end

    return GenerationConfig(; (field_name => field_values[field_name] for field_name in field_names)...)
end

function render_chat_prompt(session::ChatSession; pending_user_text::Union{Nothing, AbstractString} = nothing)::String
    prompt_io = IOBuffer()

    if !isempty(session.system_prompt)
        println(prompt_io, "System: ", session.system_prompt)
    end

    for message in session.message_history
        println(prompt_io, chat_role_label(message.role), ": ", message.content)
    end

    pending_user_text === nothing || println(prompt_io, "User: ", pending_user_text)
    print(prompt_io, "Assistant: ")
    return String(take!(prompt_io))
end

function chat_role_label(role::AbstractString)::String
    if role == "user"
        return "User"
    elseif role == "assistant"
        return "Assistant"
    else
        throw(ArgumentError("unsupported chat role $(role)"))
    end
end

const ChatMessage = NamedTuple{(:role, :content), Tuple{String, String}}

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

function chat!(session::ChatSession, user_text::AbstractString; overrides...)::String
    error("TODO v0.1: chat! not implemented")
end

"""
Generate text from a prompt using a backend model.
"""
function generate(
    model::AbstractCausalLM,
    tokenizer,
    preprocessing,
    prompt_text::AbstractString;
    generation_config::GenerationConfig = GenerationConfig(),
)::String
    error("TODO v0.1: generate not implemented")
end

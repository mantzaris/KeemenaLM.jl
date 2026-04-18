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
    prompt, generated_completion = generate_prompt_completion(
        model,
        tokenizer,
        preprocessing,
        prompt_text;
        generation_config = generation_config,
    )
    return prompt * generated_completion
end

function generate_completion(
    model::AbstractCausalLM,
    tokenizer,
    preprocessing,
    prompt_text::AbstractString;
    generation_config::GenerationConfig = GenerationConfig(),
)::String
    _, generated_completion = generate_prompt_completion(
        model,
        tokenizer,
        preprocessing,
        prompt_text;
        generation_config = generation_config,
    )
    return generated_completion
end

function generate_prompt_completion(
    model::AbstractCausalLM,
    tokenizer,
    preprocessing,
    prompt_text::AbstractString;
    generation_config::GenerationConfig = GenerationConfig(),
)
    generation_config.max_new_tokens >= 0 || throw(ArgumentError("max_new_tokens must be >= 0"))

    prompt = preprocess_text(preprocessing, prompt_text)
    prompt_token_ids = tokenizer_encode(tokenizer, prompt)
    config = model_config(model)
    config isa GPT2Config || error("Stage 1 generation only supports GPT2Config-backed models")

    if isempty(prompt_token_ids)
        if config.bos_token_id === nothing
            throw(ArgumentError("prompt encoding produced no tokens and the model has no bos_token_id"))
        end
        prompt_token_ids = [config.bos_token_id]
    end

    context_length = config.context_length
    sequence_token_ids = copy(last(prompt_token_ids, min(length(prompt_token_ids), context_length)))
    generated_token_ids = Int[]
    rng = generation_config.seed === nothing ? Stage1RNG() : Stage1RNG(generation_config.seed)

    for _ in 1:generation_config.max_new_tokens
        input_token_ids = reshape(sequence_token_ids, :, 1)
        logits, _ = lm_forward(model, input_token_ids; cache = nothing, is_training = false)
        next_token_logits = vec(logits[:, end, 1])
        next_token_id = sample_next_token(next_token_logits, generation_config; rng = rng)

        if generation_config.eos_token_id !== nothing && next_token_id == generation_config.eos_token_id
            break
        end

        push!(sequence_token_ids, next_token_id)
        length(sequence_token_ids) > context_length && deleteat!(sequence_token_ids, 1)
        push!(generated_token_ids, next_token_id)

        generated_text = tokenizer_decode(tokenizer, generated_token_ids)
        should_stop_generation(generated_token_ids, generated_text, generation_config) && break
    end

    generated_text = tokenizer_decode(tokenizer, generated_token_ids)
    trimmed_generated_text = trim_stop_sequences(generated_text, generation_config)
    return prompt, trimmed_generated_text
end

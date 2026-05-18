function train_step!(
    trainer::Trainer{<:FluxGPT2Model},
    input_token_ids::AbstractMatrix{<:Integer},
    target_token_ids::AbstractMatrix{<:Integer},
    loss_mask = nothing,
)
    size(input_token_ids) == size(target_token_ids) ||
        throw(ArgumentError("input_token_ids and target_token_ids must have the same shape"))
    loss_mask === nothing || size(loss_mask) == size(target_token_ids) ||
        throw(ArgumentError("loss_mask must have the same shape as target_token_ids"))
    trainer.optimizer === nothing && (trainer.optimizer = Flux.Descent(0.01f0))
    trainer.optimizer_state === nothing && (trainer.optimizer_state = Flux.setup(trainer.optimizer, trainer.model))
    trainer.backend == :unknown && (trainer.backend = :flux)
    loss_target_ids = move_like(target_token_ids, trainer.model.token_embedding)
    loss_weights = loss_mask === nothing ? nothing : move_like(Float32.(loss_mask), trainer.model.token_embedding)

    loss_function(model) = begin
        logits, _ = lm_forward(model, input_token_ids; cache = nothing, is_training = false)
        return loss_weights === nothing ?
            causal_lm_cross_entropy(logits, loss_target_ids) :
            causal_lm_cross_entropy(logits, loss_target_ids, loss_weights)
    end

    loss_value, model_gradients = Flux.withgradient(loss_function, trainer.model)
    isfinite(loss_value) || throw(ArgumentError("training loss is not finite"))

    Flux.update!(trainer.optimizer_state, trainer.model, model_gradients[1])
    trainer.step += 1

    return (
        loss = Float64(loss_value),
        step = trainer.step,
        epoch = trainer.epoch,
        metrics = (backend = trainer.backend,),
    )
end

function train_step!(
    trainer::Trainer{<:FluxGPT2Model},
    input_token_ids::AbstractMatrix{<:Integer},
    target_token_ids::AbstractMatrix{<:Integer},
)
    size(input_token_ids) == size(target_token_ids) ||
        throw(ArgumentError("input_token_ids and target_token_ids must have the same shape"))
    trainer.optimizer === nothing && (trainer.optimizer = Flux.Descent(0.01f0))
    trainer.optimizer_state === nothing && (trainer.optimizer_state = Flux.setup(trainer.optimizer, trainer.model))
    trainer.backend == :unknown && (trainer.backend = :flux)

    loss_function(model) = begin
        logits, _ = lm_forward(model, input_token_ids; cache = nothing, is_training = false)
        return causal_lm_cross_entropy(logits, target_token_ids)
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

const CHECKPOINT_SCHEMA_VERSION = 1

Base.@kwdef struct CheckpointManifest
    schema_version::Int = CHECKPOINT_SCHEMA_VERSION
    backend::Symbol
    architecture::String = "gpt2"
    parameter_schema::String = GPT2_PARAMETER_SCHEMA_V1
end

Base.@kwdef struct Checkpoint
    manifest::CheckpointManifest
    model_config::AbstractModelConfig
    weights::Dict{String, Any}
    optimizer::Any = nothing
    optimizer_state::Any = nothing
    step::Int = 0
    epoch::Int = 0
    rng_state::Any = nothing
    metadata::Dict{String, Any} = Dict{String, Any}()
end

function validate_checkpoint_manifest(manifest::CheckpointManifest)::CheckpointManifest
    manifest.schema_version == CHECKPOINT_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported checkpoint schema_version $(manifest.schema_version)"))
    manifest.backend in (:flux, :lux) ||
        throw(ArgumentError("unsupported checkpoint backend $(manifest.backend)"))
    manifest.architecture == "gpt2" ||
        throw(ArgumentError("unsupported checkpoint architecture $(manifest.architecture)"))
    manifest.parameter_schema == GPT2_PARAMETER_SCHEMA_V1 ||
        throw(ArgumentError("unsupported checkpoint parameter_schema $(manifest.parameter_schema)"))
    return manifest
end

function save_checkpoint(
    path::AbstractString,
    trainer::Trainer,
    model::AbstractCausalLM;
    metadata...,
)
    trainer.model === model ||
        throw(ArgumentError("trainer.model and model must refer to the same object for checkpoint save"))
    trainer.backend == :unknown &&
        throw(ArgumentError("trainer.backend must be set before saving a checkpoint"))

    model_weights = extract_weights(model)
    model_snapshot = model_config(model)
    model_snapshot isa GPT2Config || throw(ArgumentError("Stage 3 checkpoints only support GPT2Config-backed models"))

    merged_metadata = copy(trainer.metadata)
    for (key, value) in pairs(metadata)
        merged_metadata[string(key)] = value
    end

    manifest = validate_checkpoint_manifest(
        CheckpointManifest(
            backend = trainer.backend,
            architecture = "gpt2",
            parameter_schema = GPT2_PARAMETER_SCHEMA_V1,
        ),
    )

    checkpoint = Checkpoint(
        manifest = manifest,
        model_config = model_snapshot,
        weights = model_weights,
        optimizer = trainer.optimizer,
        optimizer_state = trainer.optimizer_state,
        step = trainer.step,
        epoch = trainer.epoch,
        rng_state = trainer.rng_state,
        metadata = merged_metadata,
    )

    mkpath(dirname(path))
    JLD2.jldsave(
        path;
        manifest = checkpoint.manifest,
        model_config = checkpoint.model_config,
        weights = checkpoint.weights,
        optimizer = checkpoint.optimizer,
        optimizer_state = checkpoint.optimizer_state,
        step = checkpoint.step,
        epoch = checkpoint.epoch,
        rng_state = checkpoint.rng_state,
        metadata = checkpoint.metadata,
    )
    return path
end

function load_checkpoint(path::AbstractString)::Checkpoint
    isfile(path) || throw(ArgumentError("checkpoint file does not exist: $(path)"))

    loaded_data = JLD2.load(path)
    required_keys = (
        "manifest",
        "model_config",
        "weights",
        "optimizer",
        "optimizer_state",
        "step",
        "epoch",
        "rng_state",
        "metadata",
    )
    for key in required_keys
        haskey(loaded_data, key) || throw(ArgumentError("checkpoint is missing required key $(key)"))
    end

    manifest = loaded_data["manifest"]
    manifest isa CheckpointManifest || throw(ArgumentError("checkpoint manifest has unexpected type $(typeof(manifest))"))
    validate_checkpoint_manifest(manifest)

    model_snapshot = loaded_data["model_config"]
    model_snapshot isa GPT2Config || throw(ArgumentError("Stage 3 checkpoints only support GPT2Config"))
    validate(model_snapshot)

    loaded_weights = loaded_data["weights"]
    loaded_weights isa AbstractDict || throw(ArgumentError("checkpoint weights entry must be a dictionary"))
    weights = Dict{String, Any}(String(key) => value for (key, value) in pairs(loaded_weights))

    step = loaded_data["step"]
    epoch = loaded_data["epoch"]
    step isa Integer || throw(ArgumentError("checkpoint step must be an integer"))
    epoch isa Integer || throw(ArgumentError("checkpoint epoch must be an integer"))
    step >= 0 || throw(ArgumentError("checkpoint step must be >= 0"))
    epoch >= 0 || throw(ArgumentError("checkpoint epoch must be >= 0"))

    loaded_metadata = loaded_data["metadata"]
    loaded_metadata isa AbstractDict || throw(ArgumentError("checkpoint metadata must be a dictionary"))
    metadata = Dict{String, Any}(String(key) => value for (key, value) in pairs(loaded_metadata))

    return Checkpoint(
        manifest = manifest,
        model_config = model_snapshot,
        weights = weights,
        optimizer = loaded_data["optimizer"],
        optimizer_state = loaded_data["optimizer_state"],
        step = Int(step),
        epoch = Int(epoch),
        rng_state = loaded_data["rng_state"],
        metadata = metadata,
    )
end

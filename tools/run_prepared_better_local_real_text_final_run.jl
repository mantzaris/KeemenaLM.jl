#!/usr/bin/env julia

using Flux
using JSON3
using KeemenaLM
using Printf

include("run_prepared_better_local_real_text_experiment.jl")

const FINAL_RUN_OUTPUT_DIR = joinpath(pwd(), "tmp", "prepared_better_local_real_text_final_run")
const FINAL_PREFLIGHT_OUTPUT_DIR = joinpath(pwd(), "tmp", "prepared_better_local_real_text_final_run_preflight")
const FINAL_RUN_EXPERIMENT_NAME = "prepared_better_local_real_text_final_run"
const FINAL_RUN_PURPOSE = "best-effort real-text final run baseline, not chatbot benchmarking"
const FINAL_PREFLIGHT_PURPOSE = "final run preflight only, not the full long final training run"
const FINAL_BASELINE_EPOCHS = 38
const FINAL_PREFLIGHT_EPOCHS = 4
const FINAL_PREFLIGHT_RESUME_CHECKPOINT_EPOCH = 2

function main(args)
    dataset_dir = PREPARED_CORPUS_INPUT_DIR
    output_dir = FINAL_RUN_OUTPUT_DIR
    preflight = false
    epochs = FINAL_BASELINE_EPOCHS

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument == "--preflight"
            preflight = true
            output_dir = FINAL_PREFLIGHT_OUTPUT_DIR
            epochs = FINAL_PREFLIGHT_EPOCHS
        elseif argument == "--epochs"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --epochs")
            epochs = parse(Int, args[argument_index])
        elseif argument == "--output-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --output-dir")
            output_dir = abspath(args[argument_index])
        elseif argument == "--dataset-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --dataset-dir")
            dataset_dir = abspath(args[argument_index])
        else
            error("usage: julia --project=tools/benchmark_cfg tools/run_prepared_better_local_real_text_final_run.jl [--preflight] [--epochs N] [--dataset-dir DIR] [--output-dir DIR]")
        end
        argument_index += 1
    end

    if preflight
        run_final_run_preflight(dataset_dir, output_dir)
    else
        run_final_training(dataset_dir, output_dir; epochs = epochs)
    end
end

function final_run_settings(; epochs::Int)
    return merge_settings(
        ExperimentSettings();
        complexity = 0,
        num_sentences = 0,
        prompt_prefix_characters = 24,
        context_length = 48,
        embedding_size = 128,
        ffn_hidden_size = 256,
        epochs = epochs,
        learning_rate = 0.001f0,
    )
end

function final_run_recipe_dict(; epochs::Int)
    return Dict(
        "corpus" => PREPARED_CORPUS_INPUT_DIR,
        "backend" => "flux",
        "optimizer_name" => "Flux.Adam",
        "optimizer_hyperparameters" => Dict("learning_rate" => 0.001),
        "tokenizer" => "char-level experiment-local tokenizer",
        "context_length" => 48,
        "num_layers" => 2,
        "num_heads" => 2,
        "embedding_size" => 128,
        "ffn_hidden_size" => 256,
        "batch_size" => 16,
        "epochs" => epochs,
        "checkpoint_cadence" => "every epoch",
        "seed_style" => "same deterministic seeds as the current best prepared-corpus run",
    )
end

function run_final_training(
    dataset_dir::AbstractString,
    output_dir::AbstractString;
    epochs::Int = FINAL_BASELINE_EPOCHS,
    experiment_name::AbstractString = FINAL_RUN_EXPERIMENT_NAME,
    purpose::AbstractString = FINAL_RUN_PURPOSE,
)
    output_dir = abspath(output_dir)
    mkpath(output_dir)

    recipe = final_run_recipe_dict(; epochs = epochs)
    recipe["dataset_dir"] = abspath(dataset_dir)
    recipe["output_dir"] = output_dir
    write_json(joinpath(output_dir, "run_recipe.json"), recipe)

    metrics = run_prepared_better_local_real_text_experiment(
        dataset_dir,
        output_dir;
        settings = final_run_settings(; epochs = epochs),
        experiment_name = experiment_name,
        purpose = purpose,
        optimizer_builder = settings -> Flux.Adam(settings.learning_rate),
        optimizer_name = "Flux.Adam",
        optimizer_hparams = Dict("learning_rate" => 0.001f0),
    )

    prompts = [String(sample["prompt"]) for sample in metrics["samples"]]
    write_prompt_files(output_dir, prompts)
    return metrics
end

function run_final_run_preflight(dataset_dir::AbstractString, output_dir::AbstractString)
    output_dir = abspath(output_dir)
    fresh_run_dir = joinpath(output_dir, "fresh_run")
    resume_smoke_dir = joinpath(output_dir, "resume_smoke")
    mkpath(output_dir)

    fresh_metrics = run_final_training(
        dataset_dir,
        fresh_run_dir;
        epochs = FINAL_PREFLIGHT_EPOCHS,
        experiment_name = FINAL_RUN_EXPERIMENT_NAME * "_preflight_fresh",
        purpose = FINAL_PREFLIGHT_PURPOSE,
    )

    resume_checkpoint_path = joinpath(
        fresh_run_dir,
        "checkpoints",
        @sprintf("epoch_%02d_checkpoint.jld2", FINAL_PREFLIGHT_RESUME_CHECKPOINT_EPOCH),
    )
    resume_metrics = run_resume_smoke_test(
        dataset_dir,
        fresh_run_dir,
        resume_smoke_dir,
        resume_checkpoint_path;
        target_epochs = FINAL_PREFLIGHT_EPOCHS,
    )

    summary = Dict(
        "preflight" => true,
        "dataset_dir" => abspath(dataset_dir),
        "fresh_run_dir" => fresh_run_dir,
        "resume_smoke_dir" => resume_smoke_dir,
        "final_intended_recipe" => final_run_recipe_dict(; epochs = FINAL_BASELINE_EPOCHS),
        "validated" => Dict(
            "checkpoint_creation" => true,
            "bundle_export" => true,
            "bundle_reload" => true,
            "sample_generation" => true,
            "resume_smoke" => true,
        ),
        "fresh_run" => Dict(
            "metrics_path" => joinpath(fresh_run_dir, "metrics.json"),
            "test_loss" => fresh_metrics["training"]["test_loss"],
            "final_checkpoint" => fresh_metrics["artifacts"]["final_checkpoint"],
            "bundle_dir" => fresh_metrics["artifacts"]["bundle_dir"],
            "sample_outputs_path" => fresh_metrics["artifacts"]["sample_outputs_path"],
            "evaluation_prompts_path" => joinpath(fresh_run_dir, "evaluation_prompts.txt"),
        ),
        "resume_smoke" => resume_metrics,
    )
    write_json(joinpath(output_dir, "preflight_summary.json"), summary)

    println("final run preflight summary: ", joinpath(output_dir, "preflight_summary.json"))
    return summary
end

function run_resume_smoke_test(
    dataset_dir::AbstractString,
    source_run_dir::AbstractString,
    output_dir::AbstractString,
    checkpoint_path::AbstractString;
    target_epochs::Int,
)
    output_dir = abspath(output_dir)
    mkpath(output_dir)
    checkpoint_dir = joinpath(output_dir, "checkpoints")
    mkpath(checkpoint_dir)

    settings = final_run_settings(; epochs = target_epochs)
    split_texts = load_prepared_split_texts(dataset_dir)
    tokenizer = load_tokenizer(joinpath(source_run_dir, "tokenizer.json"))
    train_batches, _ = build_lm_batches(
        split_texts.training,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
    )
    validation_batches, _ = build_lm_batches(
        split_texts.validation,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
    )
    test_batches, _ = build_lm_batches(
        split_texts.testing,
        tokenizer;
        context_length = settings.context_length,
        batch_size = settings.batch_size,
    )

    checkpoint = load_checkpoint(checkpoint_path)
    model = instantiate(checkpoint.model_config; backend = :flux, seed = settings.model_seed)
    KeemenaLM.Core.load_weights!(model, checkpoint.weights)
    trainer = KeemenaLM.Core.Trainer(
        model;
        optimizer = checkpoint.optimizer,
        optimizer_state = checkpoint.optimizer_state,
        backend = checkpoint.manifest.backend,
        step = checkpoint.step,
        epoch = checkpoint.epoch,
        rng_state = checkpoint.rng_state,
        metadata = checkpoint.metadata,
    )

    epoch_metrics = Dict{String, Any}[]
    for epoch in (trainer.epoch + 1):target_epochs
        epoch_losses = Float64[]
        for (input_batch, target_batch) in train_batches
            step_result = KeemenaLM.Core.train_step!(trainer, input_batch, target_batch)
            push!(epoch_losses, step_result.loss)
        end

        trainer.epoch = epoch
        train_loss = sum(epoch_losses) / length(epoch_losses)
        validation_loss = mean_loss(model, validation_batches)
        resumed_checkpoint_path = joinpath(checkpoint_dir, @sprintf("epoch_%02d_checkpoint.jld2", epoch))
        save_checkpoint(resumed_checkpoint_path, trainer, model; experiment = "prepared_better_local_real_text_final_run_preflight_resume", epoch = epoch)

        push!(
            epoch_metrics,
            Dict(
                "epoch" => epoch,
                "step" => trainer.step,
                "train_loss" => train_loss,
                "validation_loss" => validation_loss,
                "checkpoint_path" => resumed_checkpoint_path,
            ),
        )
    end

    final_checkpoint_path = joinpath(checkpoint_dir, "final_resumed_checkpoint.jld2")
    save_checkpoint(final_checkpoint_path, trainer, model; experiment = "prepared_better_local_real_text_final_run_preflight_resume", stage = "final")

    test_loss = mean_loss(model, test_batches)
    result = Dict(
        "checkpoint_path" => checkpoint_path,
        "checkpoint_epoch" => checkpoint.epoch,
        "target_epochs" => target_epochs,
        "resumed" => true,
        "final_epoch" => trainer.epoch,
        "initial_step" => checkpoint.step,
        "final_step" => trainer.step,
        "test_loss" => test_loss,
        "final_checkpoint" => final_checkpoint_path,
        "epoch_metrics" => epoch_metrics,
    )
    write_json(joinpath(output_dir, "resume_smoke_metrics.json"), result)
    return result
end

function write_prompt_files(output_dir::AbstractString, prompts::Vector{String})
    write_text_lines(joinpath(output_dir, "evaluation_prompts.txt"), prompts)
    write_json(joinpath(output_dir, "evaluation_prompts.json"), Dict("prompts" => prompts))
end

function write_text_lines(path::AbstractString, lines::Vector{String})
    open(path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    return path
end

function write_json(path::AbstractString, value)
    open(path, "w") do io
        JSON3.write(io, value)
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

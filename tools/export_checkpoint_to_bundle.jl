#!/usr/bin/env julia

using KeemenaLM

function main(args)
    checkpoint_path = ""
    output_dir = ""

    argument_index = 1
    while argument_index <= length(args)
        argument = args[argument_index]
        if argument in ("--help", "-h")
            print_usage()
            return nothing
        elseif argument == "--checkpoint"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --checkpoint")
            checkpoint_path = abspath(args[argument_index])
        elseif argument == "--output-dir"
            argument_index += 1
            argument_index <= length(args) || error("missing value for --output-dir")
            output_dir = abspath(args[argument_index])
        else
            error("unknown argument $(argument). Run with --help for usage.")
        end
        argument_index += 1
    end

    isempty(checkpoint_path) && throw(ArgumentError("--checkpoint is required"))
    isempty(output_dir) && throw(ArgumentError("--output-dir is required"))

    checkpoint = load_checkpoint(checkpoint_path)
    bundle = Bundle(
        model_config = checkpoint.model_config,
        weights = checkpoint.weights,
    )
    save_bundle(output_dir, bundle)

    println("checkpoint: ", checkpoint_path)
    println("step: ", checkpoint.step)
    println("epoch: ", checkpoint.epoch)
    println("bundle export: ", output_dir)
    return output_dir
end

function print_usage()
    println("""
usage: julia --project=tools/subword_real_text tools/export_checkpoint_to_bundle.jl --checkpoint PATH --output-dir DIR

Exports a training checkpoint's model config and weights as an inference bundle.
Optimizer state is intentionally not copied into the bundle.
""")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

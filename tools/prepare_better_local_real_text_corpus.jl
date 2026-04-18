#!/usr/bin/env julia

using JSON3
using SHA

const BETTER_CORPUS_DEFAULT_OUTPUT_DIR = joinpath(pwd(), "tmp", "better_local_real_text_corpus_prepared")

const BETTER_CORPUS_SOURCE_FILES = [
    "../KeemenaPreprocessing.jl/README.md",
    "../KeemenaPreprocessing.jl/paper/paper.md",
    "../KeemenaPreprocessing.jl/docs/src/index.md",
    "../KeemenaPreprocessing.jl/docs/src/guides/alignment.md",
    "../KeemenaPreprocessing.jl/docs/src/guides/configuration.md",
    "../KeemenaPreprocessing.jl/docs/src/guides/levels.md",
    "../KeemenaPreprocessing.jl/docs/src/guides/offsets.md",
    "../KeemenaPreprocessing.jl/docs/src/guides/quickstart.md",
    "../KeemenaPreprocessing.jl/docs/src/guides/streaming.md",
    "../KeemenaPreprocessing.jl/docs/src/guides/tokenizers.md",
    "../KeemenaPreprocessing.jl/docs/src/guides/types.md",
    "../KeemenaSubwords.jl/README.md",
    "../KeemenaSubwords.jl/docs/src/index.md",
    "../KeemenaSubwords.jl/docs/src/concepts.md",
    "../KeemenaSubwords.jl/docs/src/loading.md",
    "../KeemenaSubwords.jl/docs/src/loading_local.md",
    "../KeemenaSubwords.jl/docs/src/integration.md",
    "../KeemenaSubwords.jl/docs/src/quick_guide_recipes.md",
    "../KeemenaSubwords.jl/docs/src/structured_outputs_and_batching.md",
    "../KeemenaSubwords.jl/docs/src/normalization_offsets_contract.md",
]

const EXCLUDED_NEARBY_SOURCES = [
    Dict(
        "path" => "../LlmStenoExplore/data/corpus.txt",
        "reason" => "local text provenance/licensing is not explicit enough for silent inclusion here",
    ),
    Dict(
        "path" => "../ThatIsOurs/.notes*",
        "reason" => "operational/private notes are outside the safe scope for a reusable training corpus",
    ),
    Dict(
        "path" => "../KeemenaSubwords.jl/docs/src/api.md",
        "reason" => "API reference is more list-like and less natural as prose training text",
    ),
    Dict(
        "path" => "../KeemenaSubwords.jl/docs/src/models.md",
        "reason" => "generated model inventory is less natural than the curated prose/docs subset",
    ),
]

function main(args)
    length(args) <= 1 || error("usage: julia --project=. tools/prepare_better_local_real_text_corpus.jl [output_dir]")
    output_dir = length(args) == 1 ? abspath(args[1]) : BETTER_CORPUS_DEFAULT_OUTPUT_DIR
    prepare_better_local_real_text_corpus(output_dir)
end

function prepare_better_local_real_text_corpus(output_dir::AbstractString)
    repo_root = normpath(joinpath(@__DIR__, ".."))
    dataset_dir = joinpath(output_dir, "dataset")
    metadata_path = joinpath(dataset_dir, "corpus_metadata.json")

    mkpath(dataset_dir)

    result = cd(repo_root) do
        corpus_entries = collect_corpus_entries(BETTER_CORPUS_SOURCE_FILES)
        split_entries = split_entries_deterministically(corpus_entries)
        split_texts = (
            training = [entry.text for entry in split_entries.training],
            validation = [entry.text for entry in split_entries.validation],
            testing = [entry.text for entry in split_entries.testing],
        )
        split_paths = write_split_files(dataset_dir, split_texts)
        metadata = build_prepared_corpus_metadata(corpus_entries, split_entries, split_texts, split_paths)
        open(metadata_path, "w") do io
            JSON3.write(io, metadata)
        end
        (
            metadata = metadata,
            split_paths = split_paths,
        )
    end

    println("== Better local real-text corpus prepared ==")
    println("output_dir: $(output_dir)")
    println("metadata: $(metadata_path)")
    println("training split: $(result.split_paths.training)")
    println("validation split: $(result.split_paths.validation)")
    println("testing split: $(result.split_paths.testing)")
    println("total paragraphs: $(result.metadata["total_paragraphs"])")
    println("training words: $(result.metadata["split_stats"]["training"]["word_count"])")
    println("validation words: $(result.metadata["split_stats"]["validation"]["word_count"])")
    println("testing words: $(result.metadata["split_stats"]["testing"]["word_count"])")

    return result
end

function build_prepared_corpus_metadata(corpus_entries, split_entries, split_texts, split_paths)
    source_file_stats = [
        build_source_file_stat(relative_path) for relative_path in BETTER_CORPUS_SOURCE_FILES
    ]

    return Dict(
        "corpus_name" => "better_local_real_text_corpus_v1",
        "purpose" => "prepared narrow-domain local technical prose corpus for the next real-text KeemenaLM experiment",
        "source_type" => "local_curated_markdown_docs",
        "source_repositories" => [
            "../KeemenaPreprocessing.jl",
            "../KeemenaSubwords.jl",
        ],
        "source_files" => BETTER_CORPUS_SOURCE_FILES,
        "source_file_stats" => source_file_stats,
        "excluded_nearby_sources" => EXCLUDED_NEARBY_SOURCES,
        "selection_rationale" => [
            "kept the corpus fully local and inspectable",
            "preferred coherent technical prose over roadmap-style planning text",
            "favored narrow-domain NLP/preprocessing/tokenization writing with more natural sentence variety",
            "excluded API/generated/private/provenance-unclear sources",
        ],
        "cleaning_steps" => [
            "remove fenced code blocks",
            "strip markdown heading, bullet, and numbered-list markers",
            "replace markdown links with visible link text",
            "remove inline backticks",
            "collapse repeated whitespace",
            "split into paragraphs",
            "drop paragraphs shorter than five words",
        ],
        "split_policy" => "deterministic SHA1-sorted split 80/10/10 over corpus entries using source_file, paragraph_index, and paragraph text",
        "split_method" => Dict(
            "kind" => "deterministic_sha1_sort_then_slice",
            "ordering_key_fields" => ["source_file", "paragraph_index", "text"],
            "hash" => "SHA1",
            "ratio" => "80/10/10",
        ),
        "split_paths" => Dict(
            "training" => split_paths.training,
            "validation" => split_paths.validation,
            "testing" => split_paths.testing,
        ),
        "total_paragraphs" => length(corpus_entries),
        "split_counts" => Dict(
            "training" => length(split_entries.training),
            "validation" => length(split_entries.validation),
            "testing" => length(split_entries.testing),
        ),
        "split_stats" => Dict(
            "training" => prepared_split_stats(split_texts.training),
            "validation" => prepared_split_stats(split_texts.validation),
            "testing" => prepared_split_stats(split_texts.testing),
        ),
    )
end

function build_source_file_stat(relative_path::AbstractString)
    source_text = read(joinpath(pwd(), relative_path), String)
    cleaned_paragraphs = markdown_to_paragraphs(source_text)
    return Dict(
        "path" => relative_path,
        "raw_character_count" => length(source_text),
        "raw_word_count" => word_count(source_text),
        "cleaned_paragraph_count" => length(cleaned_paragraphs),
        "cleaned_word_count" => sum(word_count, cleaned_paragraphs),
    )
end

function prepared_split_stats(texts::Vector{String})
    paragraph_count = length(texts)
    character_count = sum(length, texts)
    word_count_total = sum(word_count, texts)
    return Dict(
        "paragraph_count" => paragraph_count,
        "character_count" => character_count,
        "word_count" => word_count_total,
        "mean_characters_per_paragraph" => paragraph_count == 0 ? 0.0 : character_count / paragraph_count,
        "mean_words_per_paragraph" => paragraph_count == 0 ? 0.0 : word_count_total / paragraph_count,
    )
end

word_count(text::AbstractString) = length(split(text))

Base.@kwdef struct CorpusEntry
    source_file::String
    paragraph_index::Int
    text::String
end

function collect_corpus_entries(relative_paths::Vector{String})::Vector{CorpusEntry}
    entries = CorpusEntry[]
    for relative_path in relative_paths
        absolute_path = joinpath(pwd(), relative_path)
        isfile(absolute_path) || throw(ArgumentError("corpus file does not exist: $(absolute_path)"))
        paragraphs = markdown_to_paragraphs(read(absolute_path, String))
        for (paragraph_index, paragraph) in enumerate(paragraphs)
            push!(entries, CorpusEntry(relative_path, paragraph_index, paragraph))
        end
    end
    isempty(entries) && throw(ArgumentError("no usable paragraphs were extracted from the selected corpus files"))
    return entries
end

function markdown_to_paragraphs(markdown_text::AbstractString)::Vector{String}
    lines = split(markdown_text, '\n')
    paragraphs = String[]
    paragraph_lines = String[]
    in_code_fence = false

    for raw_line in lines
        line = replace(raw_line, '\r' => "")
        stripped = strip(line)

        if startswith(stripped, "```")
            in_code_fence = !in_code_fence
            continue
        end
        in_code_fence && continue

        normalized_line = normalize_markdown_line(line)
        if isempty(strip(normalized_line))
            maybe_push_paragraph!(paragraphs, paragraph_lines)
        else
            push!(paragraph_lines, normalized_line)
        end
    end

    maybe_push_paragraph!(paragraphs, paragraph_lines)
    return paragraphs
end

function normalize_markdown_line(line::AbstractString)::String
    normalized = strip(line)
    isempty(normalized) && return ""
    startswith(normalized, "[![") && return ""

    normalized = replace(normalized, r"^#+\s*" => "")
    normalized = replace(normalized, r"^[-*]\s+" => "")
    normalized = replace(normalized, r"^\d+\.\s+" => "")
    normalized = replace(normalized, r"\[([^\]]+)\]\([^)]+\)" => s"\1")
    normalized = replace(normalized, '`' => "")
    normalized = replace(normalized, r"\s+" => " ")

    return strip(normalized)
end

function maybe_push_paragraph!(paragraphs::Vector{String}, paragraph_lines::Vector{String})
    isempty(paragraph_lines) && return nothing
    paragraph = join(paragraph_lines, " ")
    word_count(paragraph) >= 5 && push!(paragraphs, paragraph)
    empty!(paragraph_lines)
    return nothing
end

function split_entries_deterministically(entries::Vector{CorpusEntry})
    ordered_entries = sort(copy(entries); by = split_order_key)
    total_entries = length(ordered_entries)
    total_entries >= 10 || throw(ArgumentError("need at least 10 extracted paragraphs for a stable 80/10/10 split"))

    training_count = max(1, floor(Int, 0.8 * total_entries))
    validation_count = max(1, floor(Int, 0.1 * total_entries))
    testing_count = total_entries - training_count - validation_count
    testing_count >= 1 || throw(ArgumentError("not enough corpus entries to create a test split"))

    return (
        training = ordered_entries[1:training_count],
        validation = ordered_entries[(training_count + 1):(training_count + validation_count)],
        testing = ordered_entries[(training_count + validation_count + 1):end],
    )
end

function split_order_key(entry::CorpusEntry)
    digest = bytes2hex(sha1(string(entry.source_file, '\0', entry.paragraph_index, '\0', entry.text)))
    return (digest, entry.source_file, entry.paragraph_index)
end

function write_split_files(dataset_dir::AbstractString, split_texts)
    paths = (
        training = joinpath(dataset_dir, "training.txt"),
        validation = joinpath(dataset_dir, "validation.txt"),
        testing = joinpath(dataset_dir, "testing.txt"),
    )

    open(paths.training, "w") do io
        write(io, join(split_texts.training, "\n\n"))
    end
    open(paths.validation, "w") do io
        write(io, join(split_texts.validation, "\n\n"))
    end
    open(paths.testing, "w") do io
        write(io, join(split_texts.testing, "\n\n"))
    end

    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

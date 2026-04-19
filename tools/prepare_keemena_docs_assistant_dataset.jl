using JSON3
using SHA

struct SourceSpec
    package::String
    path::String
    kind::Symbol
end

Base.@kwdef struct ChatPair
    id::String
    package::String
    source_file::String
    section_title::String
    category::String
    question::String
    answer::String
    chat_text::String
end

const DEFAULT_OUTPUT_DIRECTORY = joinpath(@__DIR__, "..", "tmp", "keemena_docs_assistant_dataset_v2")
const MAX_PARAGRAPHS_PER_SECTION = 2
const MIN_PARAGRAPH_LENGTH = 48
const MIN_ANSWER_WORDS = 6

function repository_root()::String
    return normpath(joinpath(@__DIR__, ".."))
end

function source_specs()::Vector{SourceSpec}
    root = repository_root()
    preprocessing_root = normpath(joinpath(root, "..", "KeemenaPreprocessing.jl"))
    subwords_root = normpath(joinpath(root, "..", "KeemenaSubwords.jl"))

    return SourceSpec[
        SourceSpec("KeemenaLM.jl", joinpath(root, "README.md"), :markdown),
        SourceSpec("KeemenaLM.jl", joinpath(root, "docs", "src", "index.md"), :markdown),
        SourceSpec("KeemenaLM.jl", joinpath(root, "notes", "official_models.md"), :markdown),
        SourceSpec("KeemenaLM.jl", joinpath(root, "src", "core", "io", "bundle_load.jl"), :docstrings),
        SourceSpec("KeemenaLM.jl", joinpath(root, "src", "core", "io", "bundle_save.jl"), :docstrings),
        SourceSpec("KeemenaLM.jl", joinpath(root, "src", "core", "io", "model_sources.jl"), :docstrings),
        SourceSpec("KeemenaLM.jl", joinpath(root, "src", "core", "generation", "chat.jl"), :docstrings),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "README.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "index.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "guides", "quickstart.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "guides", "configuration.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "guides", "tokenizers.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "guides", "streaming.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "guides", "alignment.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "guides", "offsets.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "guides", "types.md"), :markdown),
        SourceSpec("KeemenaPreprocessing.jl", joinpath(preprocessing_root, "docs", "src", "guides", "levels.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "README.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "index.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "concepts.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "loading.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "loading_local.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "integration.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "models.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "formats.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "quick_guide_recipes.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "structured_outputs_and_batching.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "normalization_offsets_contract.md"), :markdown),
        SourceSpec("KeemenaSubwords.jl", joinpath(subwords_root, "docs", "src", "troubleshooting.md"), :markdown),
    ]
end

function main(arguments)
    output_directory = isempty(arguments) ? DEFAULT_OUTPUT_DIRECTORY : abspath(arguments[1])
    pairs, metadata = build_dataset()
    write_dataset(output_directory, pairs, metadata)

    println("Prepared dataset at: ", output_directory)
    println("Total pairs: ", length(pairs))
    println("Training/validation/testing: ",
        metadata["dataset_counts"]["training"], "/",
        metadata["dataset_counts"]["validation"], "/",
        metadata["dataset_counts"]["testing"])
end

function build_dataset()
    specs = source_specs()
    candidates = ChatPair[]

    for spec in specs
        raw_text = read_source(spec)
        sections = sectionize(raw_text; kind = spec.kind)
        append!(candidates, generate_pairs(spec, sections))
    end

    append!(candidates, supplemental_keemenalm_pairs())

    pairs = filter(is_pair_usable, deduplicate_pairs(candidates))
    sort!(pairs; by = pair -> pair.id)

    training_pairs, validation_pairs, testing_pairs = split_pairs(pairs)

    metadata = Dict(
        "intended_scope" => "Tiny narrow-domain Keemena Docs Assistant for KeemenaLM.jl, KeemenaPreprocessing.jl, and KeemenaSubwords.jl. Intended for factual, procedural, limitation, and troubleshooting QA from local project documentation only.",
        "format" => Dict(
            "primary_split_files" => "Plain chat text with one sample rendered as 'User: ...\\nAssistant: ...' and blank-line separation between samples.",
            "auxiliary_split_files" => "JSONL with id, package, source_file, section_title, category, question, answer, and chat_text.",
        ),
        "formatting_policy" => Dict(
            "source_cleaning" => [
                "remove fenced code blocks",
                "normalize markdown links to visible text",
                "strip markdown heading/list markers and inline backticks",
                "collapse whitespace",
                "keep short user-facing docstrings only for selected KeemenaLM API files",
                "skip table-heavy, list-dump, and recipe-boilerplate paragraphs",
            ],
            "pair_generation" => [
                "derive section-aware QA/chat pairs from curated local docs only",
                "use deterministic question templates based on section title and paragraph content",
                "limit to the first few substantive paragraphs per section",
                "prefer factual, procedural, limitation, and troubleshooting prompts",
                "add a small deterministic KeemenaLM supplemental set for bundle, checkpoint, chat, and model-source coverage",
            ],
        ),
        "split_policy" => Dict(
            "method" => "Deterministic SHA1 ordering over each QA pair id, then fixed 80/10/10 split by pair count.",
            "train_fraction" => 0.8,
            "validation_fraction" => 0.1,
            "test_fraction" => 0.1,
        ),
        "dataset_counts" => Dict(
            "total" => length(pairs),
            "training" => length(training_pairs),
            "validation" => length(validation_pairs),
            "testing" => length(testing_pairs),
        ),
        "counts_by_package" => count_by_package(pairs),
        "counts_by_category" => count_by_category(pairs),
        "source_files" => [spec.path for spec in specs],
    )

    return pairs, metadata
end

function read_source(spec::SourceSpec)::String
    text = read(spec.path, String)
    return spec.kind === :docstrings ? extract_docstrings(text) : text
end

function extract_docstrings(text::AbstractString)::String
    matches = collect(eachmatch(r"\"\"\"(.*?)\"\"\""s, text))
    return join([clean_inline_markdown(strip(match.captures[1])) for match in matches], "\n\n")
end

function sectionize(text::AbstractString; kind::Symbol)::Vector{NamedTuple}
    stripped_text = strip_code_blocks(text)
    normalized_text = replace(stripped_text, "\r\n" => "\n", "\r" => "\n")
    normalized_text = replace(normalized_text, r"<!--.*?-->"s => "")

    lines = split(normalized_text, '\n')
    sections = NamedTuple[]
    heading_stack = String[]
    body_lines = String[]

    function flush_section!()
        body = join(body_lines, "\n")
        paragraphs = extract_paragraphs(body)
        isempty(paragraphs) && return

        title = isempty(heading_stack) ? "Overview" : join(heading_stack, " / ")
        skip_section(title) && return
        push!(sections, (title = title, paragraphs = paragraphs))
    end

    for line in lines
        stripped_line = strip(line)
        heading_match = match(r"^(#{1,6})\s+(.*)$", stripped_line)
        if heading_match !== nothing
            flush_section!()
            empty!(body_lines)

            heading_level = length(heading_match.captures[1])
            heading_text = clean_inline_markdown(heading_match.captures[2])
            isempty(heading_text) && continue

            if length(heading_stack) >= heading_level
                resize!(heading_stack, heading_level - 1)
            end
            push!(heading_stack, heading_text)
            continue
        end

        push!(body_lines, line)
    end

    flush_section!()

    if kind === :docstrings && isempty(sections)
        paragraphs = extract_paragraphs(normalized_text)
        isempty(paragraphs) || push!(sections, (title = "API Docstrings", paragraphs = paragraphs))
    end

    return sections
end

function strip_code_blocks(text::AbstractString)::String
    lines = split(replace(text, "\r\n" => "\n", "\r" => "\n"), '\n')
    kept_lines = String[]
    inside_code_block = false

    for line in lines
        stripped_line = strip(line)
        if startswith(stripped_line, "```")
            inside_code_block = !inside_code_block
            continue
        end
        inside_code_block && continue
        push!(kept_lines, line)
    end

    return join(kept_lines, "\n")
end

function extract_paragraphs(body::AbstractString)::Vector{String}
    chunks = split(body, r"\n\s*\n")
    paragraphs = String[]

    for chunk in chunks
        cleaned_chunk = clean_block(chunk)
        length(cleaned_chunk) >= MIN_PARAGRAPH_LENGTH || continue
        skip_paragraph(cleaned_chunk) && continue
        push!(paragraphs, cleaned_chunk)
    end

    return paragraphs
end

function clean_block(block::AbstractString)::String
    lines = split(block, '\n')
    cleaned_lines = String[]

    for line in lines
        count(==('|'), line) >= 4 && continue
        text = clean_inline_markdown(line)
        text = replace(text, r"^\s*[-*+]\s+" => "")
        text = replace(text, r"^\s*\d+\.\s+" => "")
        text = replace(text, r"^\s*>\s*" => "")
        text = replace(text, r"\s+" => " ")
        text = strip(text)
        isempty(text) && continue
        all(character -> character in ('|', '-', ':'), text) && continue
        push!(cleaned_lines, text)
    end

    joined = join(cleaned_lines, " ")
    joined = replace(joined, r"\s+" => " ")
    joined = strip(joined)
    return joined
end

function clean_inline_markdown(text::AbstractString)::String
    cleaned = replace(text, r"!\[([^\]]*)\]\([^)]+\)" => s"\1")
    cleaned = replace(cleaned, r"\[([^\]]+)\]\([^)]+\)" => s"\1")
    cleaned = replace(cleaned, '`' => "")
    cleaned = replace(cleaned, "**" => "")
    cleaned = replace(cleaned, "*" => "")
    return strip(cleaned)
end

function skip_section(title::AbstractString)::Bool
    lower_title = lowercase(title)
    return occursin("contributing", lower_title) ||
           occursin("code of conduct", lower_title) ||
           occursin("community guideline", lower_title) ||
           occursin("enforcement", lower_title) ||
           occursin("documentation map", lower_title)
end

function skip_paragraph(paragraph::AbstractString)::Bool
    lower_paragraph = lowercase(paragraph)
    return startswith(lower_paragraph, "go deeper:") ||
           startswith(lower_paragraph, "next:") ||
           startswith(lower_paragraph, "documentation map") ||
           startswith(lower_paragraph, "generated from registry metadata") ||
           startswith(lower_paragraph, "see the guides") ||
           startswith(lower_paragraph, "full api") ||
           startswith(lower_paragraph, "what you should see:") ||
           startswith(lower_paragraph, "concerns and setup notes:") ||
           startswith(lower_paragraph, "common knobs:") ||
           startswith(lower_paragraph, "returns:") ||
           startswith(lower_paragraph, "pick one:") ||
           startswith(lower_paragraph, "super simple:") ||
           startswith(lower_paragraph, "peek inside:") ||
           startswith(lower_paragraph, "batch encoding") ||
           startswith(lower_paragraph, "training-ready matrices") ||
           startswith(lower_paragraph, "you have:") ||
           occursin("objective:", lower_paragraph) ||
           occursin("steps:", lower_paragraph) ||
           count(==('|'), paragraph) >= 2
end

function generate_pairs(spec::SourceSpec, sections)::Vector{ChatPair}
    pairs = ChatPair[]

    for section in sections
        section_title = section.title
        paragraphs = first(unique(section.paragraphs), min(MAX_PARAGRAPHS_PER_SECTION, length(section.paragraphs)))
        for (index, paragraph) in enumerate(paragraphs)
            question = infer_question(spec, section_title, paragraph, index)
            category = infer_category(section_title, paragraph)
            answer = paragraph

            length(question) >= 12 || continue
            length(answer) >= MIN_PARAGRAPH_LENGTH || continue

            pair_id = bytes2hex(sha1(string(spec.package, "|", spec.path, "|", section_title, "|", index, "|", question, "|", answer)))
            chat_text = "User: $(question)\nAssistant: $(answer)"

            push!(pairs, ChatPair(
                id = pair_id,
                package = spec.package,
                source_file = spec.path,
                section_title = section_title,
                category = category,
                question = question,
                answer = answer,
                chat_text = chat_text,
            ))
        end
    end

    return pairs
end

function infer_question(spec::SourceSpec, section_title::AbstractString, paragraph::AbstractString, paragraph_index::Integer)::String
    package = spec.package
    source_name = basename(spec.path)
    lower_title = lowercase(section_title)
    lower_paragraph = lowercase(paragraph)
    leaf_title = section_leaf_title(section_title)

    if section_title == "Overview" || section_title == package
        return "What is $(package)?"
    elseif package == "KeemenaLM.jl" && (occursin("official model", lower_title) || occursin("tiny-demo", lower_paragraph) || occursin("build_public_model_artifact", lower_paragraph))
        return "How does the official model flow work in $(package)?"
    elseif occursin("not yet supported", lower_title) || occursin("not yet supported", lower_paragraph) || occursin("not supported", lower_paragraph)
        return "What is not supported yet in $(package)?"
    elseif occursin("supported", lower_title) || occursin("supported state", lower_title) || occursin("status", lower_title)
        return "What is currently supported in $(package)?"
    elseif occursin("checkpoint", lower_title) || source_name == "checkpoints.jl"
        return "How do checkpoints work in $(package)?"
    elseif occursin("bundle", lower_title) || source_name in ("bundle_load.jl", "bundle_save.jl")
        return paragraph_index == 1 ? "What is a bundle in $(package)?" : "How do I save or load bundles in $(package)?"
    elseif occursin("chat", lower_title) || source_name == "chat.jl"
        return "How does chat work in $(package)?"
    elseif occursin("quick start", lower_title) || occursin("quick guide", lower_title)
        return "How do I get started with $(package)?"
    elseif occursin("loading local", lower_title)
        return "How do I load local files in $(package)?"
    elseif occursin("loading", lower_title) || source_name == "model_sources.jl"
        return "How do I load resources in $(package)?"
    elseif occursin("saving", lower_title)
        return "How do I save outputs in $(package)?"
    elseif occursin("troubleshooting", lower_title)
        return "How do I troubleshoot $(leaf_title) in $(package)?"
    elseif occursin("training", lower_title) && occursin("tokenizer", lower_paragraph)
        return "How do tokenizer training workflows work in $(package)?"
    elseif occursin("tokenizer", lower_title)
        return occursin("training", lower_title) ?
            "How do tokenizer training workflows work in $(package)?" :
            "What tokenizer support does $(package) provide?"
    elseif occursin("structured outputs", lower_title) || occursin("training-ready", lower_title)
        return "How do I get training-ready outputs in $(package)?"
    elseif occursin("offset", lower_title)
        return "What offset convention does $(package) use?"
    elseif occursin("alignment", lower_title)
        return "How does alignment work in $(package)?"
    elseif occursin("streaming", lower_title)
        return "When should I use streaming in $(package)?"
    elseif occursin("integration", lower_title)
        return "How does $(package) integrate with related Keemena packages?"
    elseif package == "KeemenaSubwords.jl" && occursin("model", lower_title)
        return "What built-in tokenizer models are available in $(package)?"
    elseif occursin("format", lower_title)
        return "What formats does $(package) support?"
    elseif occursin("configuration", lower_title)
        return "How do I configure $(package)?"
    elseif occursin("concept", lower_title)
        return "What core concepts should I know in $(package)?"
    elseif paragraph_index == 1
        return "What should I know about $(leaf_title) in $(package)?"
    else
        return "What does $(leaf_title) mean in $(package)?"
    end
end

function infer_category(section_title::AbstractString, paragraph::AbstractString)::String
    lower_title = lowercase(section_title)
    lower_paragraph = lowercase(paragraph)

    if occursin("how do", lower_paragraph) || occursin("quick", lower_title) || occursin("load", lower_title) ||
       occursin("save", lower_title) || occursin("install", lower_paragraph) || occursin("streaming", lower_title) ||
       occursin("configuration", lower_title)
        return "procedural"
    elseif occursin("not supported", lower_paragraph) || occursin("limitation", lower_paragraph) ||
           occursin("warning", lower_paragraph) || occursin("experimental", lower_paragraph) ||
           occursin("gated", lower_paragraph)
        return "limitations"
    elseif occursin("troubleshooting", lower_title) || occursin("fails", lower_paragraph) ||
           occursin("missing", lower_paragraph)
        return "troubleshooting"
    else
        return "factual"
    end
end

function section_leaf_title(section_title::AbstractString)::String
    parts = split(section_title, " / ")
    return strip(parts[end])
end

function is_pair_usable(pair::ChatPair)::Bool
    answer = strip(pair.answer)
    question = strip(pair.question)
    lower_answer = lowercase(answer)
    lower_question = lowercase(question)

    count(==('|'), answer) >= 4 && return false
    count(word -> !isempty(word), split(answer, ' ')) < MIN_ANSWER_WORDS && return false
    endswith(answer, ":") && return false
    endswith(answer, "...") && return false
    startswith(lower_answer, "what you should see:") && return false
    startswith(lower_answer, "returns:") && return false
    startswith(lower_answer, "common knobs:") && return false
    startswith(lower_answer, "pick one:") && return false
    startswith(lower_answer, "super simple:") && return false
    startswith(lower_answer, "peek inside:") && return false
    startswith(lower_answer, "batch encoding") && return false
    startswith(lower_answer, "training-ready matrices") && return false
    startswith(lower_answer, "you have:") && return false
    occursin("objective:", lower_answer) && return false
    occursin("steps:", lower_answer) && return false
    occursin(" | ", answer) && return false
    occursin(r"\|\s*stage\s*\|"i, answer) && return false
    occursin(r"\|\s*symptom\s*\|"i, answer) && return false
    occursin(r"\|\s*keyword\s*\|"i, answer) && return false
    startswith(lower_question, "can you explain keemena") && return false
    return true
end

function supplemental_keemenalm_pairs()::Vector{ChatPair}
    package = "KeemenaLM.jl"
    root = repository_root()

    raw_pairs = [
        (
            source_file = joinpath(root, "README.md"),
            section_title = "Supplemental / Package Overview",
            category = "factual",
            question = "What is KeemenaLM.jl?",
            answer = "KeemenaLM.jl is a Julia proof-of-concept language-model package centered on a small GPT-2 style decoder-only model with portable bundles, resumable checkpoints, REPL chat, and a second inference backend.",
        ),
        (
            source_file = joinpath(root, "README.md"),
            section_title = "Supplemental / Supported State",
            category = "factual",
            question = "What is currently supported in KeemenaLM.jl?",
            answer = "The current proof-of-concept supports Flux inference and training, portable bundles, resumable checkpoints, REPL chat, official demo model resolution through local artifact registration, and Lux inference on CPU.",
        ),
        (
            source_file = joinpath(root, "README.md"),
            section_title = "Supplemental / Limitations",
            category = "limitations",
            question = "What is not supported yet in KeemenaLM.jl?",
            answer = "Lux training parity is not supported yet, tokenizer and preprocessing objects are still supplied explicitly instead of being persisted inside bundles, and official models do not have a remote hosted download path in this repo setup.",
        ),
        (
            source_file = joinpath(root, "src/core/io/model_sources.jl"),
            section_title = "Supplemental / Source Resolution",
            category = "procedural",
            question = "How does bundle source resolution work in KeemenaLM.jl?",
            answer = "KeemenaLM resolves an existing local bundle directory first. If the source is not a local directory, it can then resolve a supported official model key such as tiny-demo through the local Julia artifact registry.",
        ),
        (
            source_file = joinpath(root, "notes", "official_models.md"),
            section_title = "Supplemental / Official Models",
            category = "procedural",
            question = "How do I use the official tiny-demo model in KeemenaLM.jl?",
            answer = "In this repo setup the tiny-demo model is a locally registered artifact. You first run tools/build_public_model_artifact.jl, then resolve it with available_models(), download_model(\"tiny-demo\"), load_bundle(\"tiny-demo\"), or load_model(\"tiny-demo\").",
        ),
        (
            source_file = joinpath(root, "notes", "official_models.md"),
            section_title = "Supplemental / Official Model Limits",
            category = "limitations",
            question = "What should I know before using tiny-demo in KeemenaLM.jl?",
            answer = "The official demo flow is local artifact registration only, not a fresh-user remote download path, and callers still have to supply the matching tokenizer and preprocessing convention explicitly.",
        ),
        (
            source_file = joinpath(root, "src/core/io/bundle_load.jl"),
            section_title = "Supplemental / Bundle Loading",
            category = "procedural",
            question = "How do I load a saved bundle in KeemenaLM.jl?",
            answer = "Use load_bundle(source) to load a model bundle from a directory or a supported official model source. The bundle loader reads the manifest, model config, and stored weights before returning a validated Bundle object.",
        ),
        (
            source_file = joinpath(root, "src/core/io/bundle_save.jl"),
            section_title = "Supplemental / Bundle Saving",
            category = "procedural",
            question = "How do I save a bundle in KeemenaLM.jl?",
            answer = "Use save_bundle(directory_path, bundle) to write the bundle manifest, model config, and weights into a bundle directory so the model can be reloaded later.",
        ),
        (
            source_file = joinpath(root, "src/core/generation/chat.jl"),
            section_title = "Supplemental / ChatSession",
            category = "factual",
            question = "What is ChatSession in KeemenaLM.jl?",
            answer = "ChatSession is the minimal in-memory chat wrapper around generate. It keeps the model, tokenizer, preprocessing object, generation config, system prompt, and message history together for chat-style prompting.",
        ),
        (
            source_file = joinpath(root, "src/core/generation/chat.jl"),
            section_title = "Supplemental / Chat Workflow",
            category = "procedural",
            question = "How does chat work in KeemenaLM.jl?",
            answer = "chat! renders a chat prompt from the current session state, generates an assistant reply, and appends both the user message and assistant response to the in-memory message history.",
        ),
        (
            source_file = joinpath(root, "src/core/training/checkpoints.jl"),
            section_title = "Supplemental / Checkpoints",
            category = "procedural",
            question = "How do checkpoints work in KeemenaLM.jl?",
            answer = "A checkpoint stores the model config, portable weight dictionary, optimizer state, step, epoch, RNG state, and metadata. save_checkpoint writes that snapshot to JLD2 and load_checkpoint validates and restores it.",
        ),
        (
            source_file = joinpath(root, "examples", "chat_repl.jl"),
            section_title = "Supplemental / Chat REPL",
            category = "procedural",
            question = "How do I start the chat REPL in KeemenaLM.jl?",
            answer = "Run examples/chat_repl.jl with a bundle directory or official model key. In the Stage 6 demo flow, official model keys such as tiny-demo must be registered locally first and the caller still supplies tokenizer and preprocessing behavior explicitly.",
        ),
    ]

    return [
        ChatPair(
            id = bytes2hex(sha1(string(package, "|", pair.source_file, "|", pair.section_title, "|", pair.question, "|", pair.answer))),
            package = package,
            source_file = pair.source_file,
            section_title = pair.section_title,
            category = pair.category,
            question = pair.question,
            answer = pair.answer,
            chat_text = "User: $(pair.question)\nAssistant: $(pair.answer)",
        ) for pair in raw_pairs
    ]
end

function deduplicate_pairs(pairs::Vector{ChatPair})::Vector{ChatPair}
    seen = Set{String}()
    unique_pairs = ChatPair[]

    for pair in pairs
        fingerprint = lowercase(strip(pair.question)) * "\n" * lowercase(strip(pair.answer))
        fingerprint in seen && continue
        push!(seen, fingerprint)
        push!(unique_pairs, pair)
    end

    return unique_pairs
end

function split_pairs(pairs::Vector{ChatPair})
    ordered_pairs = sort(pairs; by = pair -> pair.id)
    total = length(ordered_pairs)
    training_end = floor(Int, 0.8 * total)
    validation_end = floor(Int, 0.9 * total)

    training_pairs = ordered_pairs[1:training_end]
    validation_pairs = ordered_pairs[(training_end + 1):validation_end]
    testing_pairs = ordered_pairs[(validation_end + 1):end]
    return training_pairs, validation_pairs, testing_pairs
end

function count_by_package(pairs::Vector{ChatPair})
    counts = Dict{String, Int}()
    for pair in pairs
        counts[pair.package] = get(counts, pair.package, 0) + 1
    end
    return counts
end

function count_by_category(pairs::Vector{ChatPair})
    counts = Dict{String, Int}()
    for pair in pairs
        counts[pair.category] = get(counts, pair.category, 0) + 1
    end
    return counts
end

function write_dataset(output_directory::AbstractString, pairs::Vector{ChatPair}, metadata::Dict)
    mkpath(output_directory)
    training_pairs, validation_pairs, testing_pairs = split_pairs(pairs)

    write_split(output_directory, "training", training_pairs)
    write_split(output_directory, "validation", validation_pairs)
    write_split(output_directory, "testing", testing_pairs)

    open(joinpath(output_directory, "metadata.json"), "w") do io
        JSON3.write(io, metadata; allow_inf = false)
    end
end

function write_split(output_directory::AbstractString, split_name::AbstractString, pairs::Vector{ChatPair})
    text_path = joinpath(output_directory, "$(split_name).txt")
    jsonl_path = joinpath(output_directory, "$(split_name).jsonl")

    open(text_path, "w") do io
        for (index, pair) in enumerate(pairs)
            index > 1 && write(io, "\n\n")
            write(io, pair.chat_text)
        end
    end

    open(jsonl_path, "w") do io
        for pair in pairs
            record = Dict(
                "id" => pair.id,
                "package" => pair.package,
                "source_file" => pair.source_file,
                "section_title" => pair.section_title,
                "category" => pair.category,
                "question" => pair.question,
                "answer" => pair.answer,
                "chat_text" => pair.chat_text,
            )
            JSON3.write(io, record)
            write(io, "\n")
        end
    end
end

main(ARGS)

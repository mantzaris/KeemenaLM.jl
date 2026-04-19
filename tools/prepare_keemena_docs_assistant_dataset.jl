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

const DEFAULT_OUTPUT_DIRECTORY = joinpath(@__DIR__, "..", "tmp", "keemena_docs_assistant_dataset_v4")
const MAX_PARAGRAPHS_PER_SECTION = 1
const MIN_PARAGRAPH_LENGTH = 48
const MIN_ANSWER_WORDS = 6
const ASSISTANT_END_MARKER = "<END_ASSISTANT>"
const MAX_AUTO_ANSWER_CHARACTERS = 220
const MAX_AUTO_ANSWER_SENTENCES = 2

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

    append!(candidates, supplemental_curated_pairs())

    pairs = filter(is_pair_usable, deduplicate_pairs(candidates))
    sort!(pairs; by = pair -> pair.id)

    training_pairs, validation_pairs, testing_pairs = split_pairs(pairs)

    metadata = Dict(
        "intended_scope" => "Tiny narrow-domain Keemena Docs Assistant for KeemenaLM.jl, KeemenaPreprocessing.jl, and KeemenaSubwords.jl. Intended for factual, procedural, limitation, and troubleshooting QA from local project documentation only.",
        "format" => Dict(
            "primary_split_files" => "Plain chat text with one sample rendered as 'User: ...\\nAssistant: ...\\n<END_ASSISTANT>' and blank-line separation between samples.",
            "auxiliary_split_files" => "JSONL with id, package, source_file, section_title, category, question, answer, and chat_text.",
        ),
        "formatting_policy" => Dict(
            "source_cleaning" => [
                "remove fenced code blocks",
                "normalize markdown links to visible text",
                "strip markdown heading/list markers and inline backticks",
                "collapse whitespace",
                "keep short user-facing docstrings only for selected KeemenaLM API files",
                "skip table-heavy, list-dump, registry-metadata, and recipe-boilerplate paragraphs",
            ],
            "pair_generation" => [
                "derive section-aware QA/chat pairs from curated local docs only",
                "use deterministic question templates based on section title and paragraph content",
                "limit to the first substantive paragraph per section for cleaner supervision",
                "prefer factual, procedural, limitation, and troubleshooting prompts",
                "add a small deterministic curated supplement for clearer bundle, checkpoint, chat, limitation, troubleshooting, tokenizer, and integration coverage",
                "keep one best deterministic answer per question so training sees a more consistent assistant target",
                "append an explicit assistant end marker after every answer",
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
        "source_files" => unique(vcat([spec.path for spec in specs], [pair.source_file for pair in pairs])),
        "chat_markers" => Dict(
            "assistant_end" => ASSISTANT_END_MARKER,
        ),
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
           occursin("documentation map", lower_title) ||
           lower_title == "api" ||
           occursin("current progress", lower_title) ||
           occursin("current project progress", lower_title) ||
           occursin("immediate next focus", lower_title) ||
           occursin("start here", lower_title) ||
           occursin("summary", lower_title) ||
           occursin("some recipes", lower_title) ||
           occursin("what you get", lower_title) ||
           occursin("key features", lower_title) ||
           occursin("scope and ecosystem", lower_title)
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
           startswith(lower_paragraph, "home / start here:") ||
           startswith(lower_paragraph, "fields ") ||
           startswith(lower_paragraph, "format symbol:") ||
           startswith(lower_paragraph, "format symbols:") ||
           startswith(lower_paragraph, "one string ->") ||
           startswith(lower_paragraph, "ids:") ||
           startswith(lower_paragraph, "clean_text =") ||
           startswith(lower_paragraph, "top-level structure") ||
           startswith(lower_paragraph, "an immutable bidirectional mapping") ||
           startswith(lower_paragraph, "this document is the canonical contract") ||
           startswith(lower_paragraph, "purpose:") ||
           startswith(lower_paragraph, "generated from registry metadata") ||
           startswith(lower_paragraph, "0 ... n") ||
           startswith(lower_paragraph, "the channel is unbuffered") ||
           startswith(lower_paragraph, "for tokenizer development") ||
           startswith(lower_paragraph, "directory preference order:") ||
           startswith(lower_paragraph, "use the hf export target") ||
           startswith(lower_paragraph, "calls _ensure_lower_levels!") ||
           startswith(lower_paragraph, "if you are new to the package") ||
           startswith(lower_paragraph, "choose a tokenizer source:") ||
           startswith(lower_paragraph, "this page gives a practical introduction") ||
           startswith(lower_paragraph, "current supported state:") ||
           startswith(lower_paragraph, "any callable f(") ||
           startswith(lower_paragraph, "once you have called build_ensure_alignments!") ||
           occursin("objective:", lower_paragraph) ||
           occursin("steps:", lower_paragraph) ||
           occursin("distribution:", lower_paragraph) ||
           occursin("expected files:", lower_paragraph) ||
           occursin("upstream:", lower_paragraph) ||
           occursin("license:", lower_paragraph) ||
           occursin("https://", lower_paragraph) ||
           occursin("http://", lower_paragraph) ||
           (count(==(':'), paragraph) >= 4 && count(==('.'), paragraph) <= 1) ||
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
            answer = normalize_answer(paragraph, category; curated = false)

            length(question) >= 12 || continue
            length(answer) >= MIN_PARAGRAPH_LENGTH || continue

            pair_id = bytes2hex(sha1(string(spec.package, "|", spec.path, "|", section_title, "|", index, "|", question, "|", answer)))
            chat_text = render_chat_text(question, answer)

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

function normalize_answer(answer::AbstractString, category::AbstractString)::String
    return normalize_answer(answer, category; curated = false)
end

function normalize_answer(answer::AbstractString, category::AbstractString; curated::Bool)::String
    normalized = replace(answer, '\u2011' => '-')
    normalized = replace(normalized, '\u2013' => '-')
    normalized = replace(normalized, '\u2014' => '-')
    normalized = replace(normalized, r"\s+" => " ")
    normalized = strip(normalized)
    normalized = replace(normalized, r":\s*$" => "")
    normalized = replace(normalized, r"\s*:\s+" => ": ")
    normalized = strip(normalized)

    if category == "procedural" && occursin("use `", normalized)
        normalized = replace(normalized, '`' => "")
    end

    curated || (normalized = summarize_auto_answer(normalized))

    startswith(normalized, "Not yet supported:") && (normalized = replace(normalized, "Not yet supported:" => "Not supported yet:"))
    startswith(normalized, "Current supported state:") && (normalized = replace(normalized, "Current supported state:" => "Supported now:"))

    if !isempty(normalized)
        first_char = first(normalized)
        if islowercase(first_char) && occursin(r"^[a-z][a-z0-9 -]+"i, normalized) && !occursin(r"^[a-z0-9_]+[\(\)]", normalized)
            normalized = uppercase(first(normalized)) * normalized[nextind(normalized, firstindex(normalized)):end]
        end
    end

    endswith(normalized, '.') || endswith(normalized, '!') || endswith(normalized, '?') || (normalized *= ".")
    return normalized
end

function summarize_auto_answer(answer::AbstractString)::String
    shortened = strip(answer)
    sentences = [strip(piece) for piece in split(shortened, r"(?<=[.!?])\s+") if !isempty(strip(piece))]

    if !isempty(sentences)
        shortened = join(first(sentences, min(MAX_AUTO_ANSWER_SENTENCES, length(sentences))), " ")
    end

    if length(shortened) > MAX_AUTO_ANSWER_CHARACTERS
        cutoff = findlast(==(' '), shortened[1:MAX_AUTO_ANSWER_CHARACTERS])
        cutoff === nothing || (shortened = shortened[1:cutoff])
        shortened = strip(shortened)
    end

    shortened = replace(shortened, r"\s+" => " ")
    return strip(shortened)
end

function render_chat_text(question::AbstractString, answer::AbstractString)::String
    return "User: $(strip(question))\nAssistant: $(strip(answer))\n$(ASSISTANT_END_MARKER)"
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
    elseif lower_title == "troubleshooting"
        return "How do I troubleshoot common issues in $(package)?"
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
    occursin("distribution:", lower_answer) && return false
    occursin("expected files:", lower_answer) && return false
    occursin("upstream:", lower_answer) && return false
    occursin("license:", lower_answer) && return false
    occursin("http://", lower_answer) && return false
    occursin("https://", lower_answer) && return false
    startswith(lower_answer, "fields ") && return false
    startswith(lower_answer, "format symbol:") && return false
    startswith(lower_answer, "format symbols:") && return false
    startswith(lower_answer, "home / start here:") && return false
    startswith(lower_answer, "one string ->") && return false
    startswith(lower_answer, "ids:") && return false
    startswith(lower_answer, "top-level structure") && return false
    startswith(lower_answer, "this document is the canonical contract") && return false
    startswith(lower_answer, "0 ... n") && return false
    startswith(lower_answer, "the channel is unbuffered") && return false
    startswith(lower_answer, "for tokenizer development") && return false
    startswith(lower_answer, "directory preference order:") && return false
    startswith(lower_answer, "use the hf export target") && return false
    startswith(lower_answer, "calls _ensure_lower_levels!") && return false
    startswith(lower_answer, "current supported state:") && return false
    startswith(lower_answer, "see the generated api reference page") && return false
    startswith(lower_answer, "if the bundle has") && return false
    startswith(lower_answer, "that translates") && return false
    startswith(lower_answer, "pairing of a corpus") && return false
    startswith(lower_answer, "offsets_coordinate_system()") && return false
    startswith(lower_answer, "validate_offsets_contract(") && return false
    startswith(lower_answer, "tokenize(tok, text)") && return false
    startswith(lower_answer, "keemenapreprocessing: produces") && return false
    startswith(lower_answer, "the streaming merge helper") && return false
    startswith(lower_answer, "creates the requested") && return false
    startswith(lower_answer, "this page is a choose-your-path") && return false
    startswith(lower_answer, "this page is a first-hour") && return false
    startswith(lower_answer, "for some tasks") && return false
    startswith(lower_answer, "for step-by-step usage patterns") && return false
    startswith(lower_answer, "some upstream models require") && return false
    startswith(lower_answer, "_generated from registry metadata") && return false
    startswith(lower_answer, "directories in sources are silently skipped") && return false
    startswith(lower_answer, "encode_result and encode_batch_result") && return false
    startswith(lower_answer, "vocab.json + merges.txt or encoder.json") && return false
    (count(==(':'), answer) >= 4 && count(==('.'), answer) <= 1) && return false
    startswith(lower_question, "can you explain keemena") && return false
    startswith(lower_question, "what should i know about") && return false
    startswith(lower_question, "what does ") && return false
    occursin("troubleshooting in", lower_question) && return false
    return true
end

function supplemental_curated_pairs()::Vector{ChatPair}
    root = repository_root()
    preprocessing_root = normpath(joinpath(root, "..", "KeemenaPreprocessing.jl"))
    subwords_root = normpath(joinpath(root, "..", "KeemenaSubwords.jl"))

    raw_pairs = [
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "README.md"),
            section_title = "Supplemental / Package Overview",
            category = "factual",
            question = "What is KeemenaLM.jl?",
            answer = "KeemenaLM.jl is a Julia proof-of-concept language-model package centered on a small GPT-2 style decoder-only model with portable bundles, resumable checkpoints, REPL chat, and a second inference backend.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "README.md"),
            section_title = "Supplemental / Supported State",
            category = "factual",
            question = "What is currently supported in KeemenaLM.jl?",
            answer = "The current proof-of-concept supports Flux inference and training, portable bundles, resumable checkpoints, REPL chat, official demo model resolution through local artifact registration, and Lux inference on CPU.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "README.md"),
            section_title = "Supplemental / Limitations",
            category = "limitations",
            question = "What is not supported yet in KeemenaLM.jl?",
            answer = "Lux training parity is not supported yet, tokenizer and preprocessing objects are still supplied explicitly instead of being persisted inside bundles, and official models do not have a remote hosted download path in this repo setup.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/model_sources.jl"),
            section_title = "Supplemental / Source Resolution",
            category = "procedural",
            question = "How does bundle source resolution work in KeemenaLM.jl?",
            answer = "KeemenaLM resolves an existing local bundle directory first. If the source is not a local directory, it can then resolve a supported official model key such as tiny-demo through the local Julia artifact registry.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "notes", "official_models.md"),
            section_title = "Supplemental / Official Models",
            category = "procedural",
            question = "How do I use the official tiny-demo model in KeemenaLM.jl?",
            answer = "In this repo setup the tiny-demo model is a locally registered artifact. You first run tools/build_public_model_artifact.jl, then resolve it with available_models(), download_model(\"tiny-demo\"), load_bundle(\"tiny-demo\"), or load_model(\"tiny-demo\").",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "notes", "official_models.md"),
            section_title = "Supplemental / Official Model Limits",
            category = "limitations",
            question = "What should I know before using tiny-demo in KeemenaLM.jl?",
            answer = "The official demo flow is local artifact registration only, not a fresh-user remote download path, and callers still have to supply the matching tokenizer and preprocessing convention explicitly.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/bundle_load.jl"),
            section_title = "Supplemental / Bundle Loading",
            category = "procedural",
            question = "How do I load a saved bundle in KeemenaLM.jl?",
            answer = "Use load_bundle(source) to load a model bundle from a directory or a supported official model source. The bundle loader reads the manifest, model config, and stored weights before returning a validated Bundle object.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/bundle_save.jl"),
            section_title = "Supplemental / Bundle Saving",
            category = "procedural",
            question = "How do I save a bundle in KeemenaLM.jl?",
            answer = "Use save_bundle(directory_path, bundle) to write the bundle manifest, model config, and weights into a bundle directory so the model can be reloaded later.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/model_sources.jl"),
            section_title = "Supplemental / Resolve Bundle",
            category = "procedural",
            question = "What does resolve_bundle do in KeemenaLM.jl?",
            answer = "resolve_bundle turns a bundle source into a validated local bundle directory path. It accepts a local directory and can also resolve supported official model keys through the local artifact registry.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/model_sources.jl"),
            section_title = "Supplemental / Load Model",
            category = "procedural",
            question = "How do I instantiate a model from a bundle source in KeemenaLM.jl?",
            answer = "Use load_model(source; backend=...) when you want one call that resolves the source, loads the bundle, and instantiates the model for the selected backend.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/model_sources.jl"),
            section_title = "Supplemental / Local Path Precedence",
            category = "factual",
            question = "What happens if a local bundle directory has the same name as an official model key in KeemenaLM.jl?",
            answer = "Local directory resolution wins first. KeemenaLM checks whether the source is an existing local bundle directory before it falls back to official model key resolution.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/bundle_schema.jl"),
            section_title = "Supplemental / Bundle Contents",
            category = "factual",
            question = "What does a KeemenaLM bundle contain?",
            answer = "A KeemenaLM bundle is the portable inference package. It stores the bundle manifest, the model config, and the model weights needed to reload the model later.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/bundle_schema.jl"),
            section_title = "Supplemental / Bundle Limits",
            category = "limitations",
            question = "Do KeemenaLM bundles include tokenizer and preprocessing objects?",
            answer = "Not yet. Bundles currently carry the model payload, but callers still provide the matching tokenizer and preprocessing behavior explicitly.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/io/model_sources.jl"),
            section_title = "Supplemental / Remote Model Support",
            category = "limitations",
            question = "Can KeemenaLM fetch official models from a remote host in this repo setup?",
            answer = "No. The current official-model flow is local artifact registration only, so you must register or build the artifact locally before resolving the model key.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/generation/chat.jl"),
            section_title = "Supplemental / ChatSession",
            category = "factual",
            question = "What is ChatSession in KeemenaLM.jl?",
            answer = "ChatSession is the minimal in-memory chat wrapper around generate. It keeps the model, tokenizer, preprocessing object, generation config, system prompt, and message history together for chat-style prompting.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/generation/chat.jl"),
            section_title = "Supplemental / Chat Workflow",
            category = "procedural",
            question = "How does chat work in KeemenaLM.jl?",
            answer = "chat! renders a chat prompt from the current session state, generates an assistant reply, and appends both the user message and assistant response to the in-memory message history.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/training/checkpoints.jl"),
            section_title = "Supplemental / Checkpoints",
            category = "procedural",
            question = "How do checkpoints work in KeemenaLM.jl?",
            answer = "A checkpoint stores the model config, portable weight dictionary, optimizer state, step, epoch, RNG state, and metadata. save_checkpoint writes that snapshot to JLD2 and load_checkpoint validates and restores it.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/training/checkpoints.jl"),
            section_title = "Supplemental / Checkpoint Resume",
            category = "procedural",
            question = "What do I need to resume training from a KeemenaLM checkpoint?",
            answer = "You need the checkpoint file and the same training path. load_checkpoint restores the saved model snapshot, optimizer state, counters, RNG state, and metadata so training can continue from that point.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "src/core/training/checkpoints.jl"),
            section_title = "Supplemental / Checkpoint Validation",
            category = "troubleshooting",
            question = "Why might load_checkpoint fail in KeemenaLM.jl?",
            answer = "load_checkpoint validates the checkpoint schema, backend, architecture, required keys, and basic counter fields. It fails when the file is missing required entries or does not match the supported checkpoint format.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "examples", "chat_repl.jl"),
            section_title = "Supplemental / Chat REPL",
            category = "procedural",
            question = "How do I start the chat REPL in KeemenaLM.jl?",
            answer = "Run examples/chat_repl.jl with a bundle directory or official model key. In the Stage 6 demo flow, official model keys such as tiny-demo must be registered locally first and the caller still supplies tokenizer and preprocessing behavior explicitly.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "examples", "chat_demo.jl"),
            section_title = "Supplemental / One Turn Chat Demo",
            category = "procedural",
            question = "How do I run a one-turn chat demo in KeemenaLM.jl?",
            answer = "Run examples/chat_demo.jl with a bundle directory or official model key. In this repo setup, official keys such as tiny-demo must be registered locally first.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "README.md"),
            section_title = "Supplemental / Lux State",
            category = "limitations",
            question = "What does Lux currently support in KeemenaLM.jl?",
            answer = "Lux currently supports model instantiation, forward pass, shared bundle weights, and CPU generation. Full Lux training parity is not part of the supported path yet.",
        ),
        (
            package = "KeemenaLM.jl",
            source_file = joinpath(root, "README.md"),
            section_title = "Supplemental / Quality Expectations",
            category = "limitations",
            question = "Is the current KeemenaLM demo model already a good chatbot?",
            answer = "No. The current baseline is still a proof-of-concept artifact with weak, domain-narrow generation, so it is useful for pipeline validation but not yet a strong chatbot.",
        ),
        (
            package = "KeemenaPreprocessing.jl",
            source_file = joinpath(preprocessing_root, "README.md"),
            section_title = "Supplemental / Package Overview",
            category = "factual",
            question = "What is KeemenaPreprocessing.jl?",
            answer = "KeemenaPreprocessing.jl is a streaming text-preparation pipeline for Julia. It turns raw text into normalized token ids, vocabularies, offset vectors, and alignment-ready bundle structures.",
        ),
        (
            package = "KeemenaPreprocessing.jl",
            source_file = joinpath(preprocessing_root, "docs", "src", "guides", "streaming.md"),
            section_title = "Supplemental / Streaming",
            category = "procedural",
            question = "When should I use streaming in KeemenaPreprocessing.jl?",
            answer = "Use streaming when the corpus does not fit comfortably in memory or when you want bounded-memory preprocessing. The streaming helpers trade some throughput for fixed-size chunk processing.",
        ),
        (
            package = "KeemenaPreprocessing.jl",
            source_file = joinpath(preprocessing_root, "docs", "src", "guides", "streaming.md"),
            section_title = "Supplemental / Streaming Full",
            category = "procedural",
            question = "What does preprocess_corpus_streaming_full do?",
            answer = "preprocess_corpus_streaming_full runs the streaming pipeline and merges the chunks into one cohesive bundle. It is the right choice when you still want a single final artifact without loading the raw corpus all at once.",
        ),
        (
            package = "KeemenaPreprocessing.jl",
            source_file = joinpath(preprocessing_root, "docs", "src", "guides", "quickstart.md"),
            section_title = "Supplemental / Bundle Save Load",
            category = "procedural",
            question = "How do I save and reload a preprocessing bundle?",
            answer = "Use save_preprocess_bundle to write the bundle to JLD2 and load_preprocess_bundle to read it back. That is the default convenience path for saving prepared corpora and metadata.",
        ),
        (
            package = "KeemenaPreprocessing.jl",
            source_file = joinpath(preprocessing_root, "docs", "src", "guides", "offsets.md"),
            section_title = "Supplemental / Offsets",
            category = "factual",
            question = "What offset convention does KeemenaPreprocessing.jl use?",
            answer = "KeemenaPreprocessing uses 1-based offset vectors that are monotone and sentinel-terminated. That lets you recover each segment with offsets[i] : offsets[i+1]-1.",
        ),
        (
            package = "KeemenaPreprocessing.jl",
            source_file = joinpath(preprocessing_root, "docs", "src", "guides", "levels.md"),
            section_title = "Supplemental / Alignments",
            category = "procedural",
            question = "What does build_ensure_alignments! do in KeemenaPreprocessing.jl?",
            answer = "build_ensure_alignments! ensures the canonical cross-level membership maps exist in the bundle. It is the standard way to restore byte-to-word and related alignments before downstream lookup work.",
        ),
        (
            package = "KeemenaPreprocessing.jl",
            source_file = joinpath(preprocessing_root, "docs", "src", "guides", "quickstart.md"),
            section_title = "Supplemental / Paragraph Offsets Warning",
            category = "troubleshooting",
            question = "Why are paragraph offsets not being recorded in KeemenaPreprocessing.jl?",
            answer = "Paragraph offsets require preserved newlines. If you request paragraph offsets while preserve_newlines is false, the pipeline warns and turns newline preservation on so paragraph structure can be recorded.",
        ),
        (
            package = "KeemenaPreprocessing.jl",
            source_file = joinpath(preprocessing_root, "docs", "src", "guides", "quickstart.md"),
            section_title = "Supplemental / Common Troubleshooting",
            category = "troubleshooting",
            question = "How do I troubleshoot common issues in KeemenaPreprocessing.jl?",
            answer = "Start by checking configuration mismatches, offset-recording flags, and whether preserve_newlines or build_ensure_alignments! is needed for the workflow you are using. Most common issues come from missing offsets, mixed vocabularies, or incompatible streaming assumptions.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "README.md"),
            section_title = "Supplemental / Package Overview",
            category = "factual",
            question = "What is KeemenaSubwords.jl?",
            answer = "KeemenaSubwords.jl is a tokenizer package for Julia that loads and works with multiple subword families, exposes ids, pieces, offsets, and masks, and helps build training-ready LM batches.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "loading.md"),
            section_title = "Supplemental / Loading",
            category = "procedural",
            question = "How should I load a tokenizer in KeemenaSubwords.jl?",
            answer = "Use load_tokenizer when you want key-based or auto-detected loading. Use an explicit loader such as load_bpe_gpt2, load_sentencepiece, or load_tiktoken when you want a strict file contract.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "troubleshooting.md"),
            section_title = "Supplemental / Wrong Auto Detect",
            category = "troubleshooting",
            question = "What should I do if KeemenaSubwords auto-detects the wrong tokenizer format?",
            answer = "Force the format explicitly in load_tokenizer. That is the recommended fix when a path could plausibly match more than one tokenizer family.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "troubleshooting.md"),
            section_title = "Supplemental / tokenizer.model Confusion",
            category = "troubleshooting",
            question = "What if tokenizer.model is not actually a SentencePiece file?",
            answer = "Treat the file format, not the filename, as the source of truth. Some models ship tiktoken text in a file named tokenizer.model, so you may need to force format=:tiktoken instead of SentencePiece loading.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "formats.md"),
            section_title = "Supplemental / Formats",
            category = "factual",
            question = "What tokenizer formats does KeemenaSubwords.jl support?",
            answer = "KeemenaSubwords supports classic BPE, GPT-2 style BPE, ByteBPE, WordPiece, Unigram, SentencePiece, tiktoken text files, and HF tokenizer.json formats.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "models.md"),
            section_title = "Supplemental / Model Registry",
            category = "factual",
            question = "What built-in tokenizer models are available in KeemenaSubwords.jl?",
            answer = "KeemenaSubwords includes a tokenizer registry with shipped, public, and gated model entries. Use the registry helpers to list available keys and inspect which models are built in versus installable.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "concepts.md"),
            section_title = "Supplemental / Offset Caveat",
            category = "limitations",
            question = "What should I know about byte-level offsets in KeemenaSubwords.jl?",
            answer = "Byte-level tokenizers return valid UTF-8 codeunit spans, but those spans are not always safe Julia string slice boundaries on multibyte text. Use the byte-level offset helpers when you need safe inspection.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "normalization_offsets_contract.md"),
            section_title = "Supplemental / Sentinel Offsets",
            category = "factual",
            question = "What does the offset sentinel (0, 0) mean in KeemenaSubwords.jl?",
            answer = "The sentinel (0, 0) means the token does not map to a real source-text span. Inserted special tokens commonly use that sentinel.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "structured_outputs_and_batching.md"),
            section_title = "Supplemental / Training Ready Outputs",
            category = "procedural",
            question = "How do I get training-ready outputs in KeemenaSubwords.jl?",
            answer = "Use the structured-output helpers when you want token ids, masks, and offsets together, and use quick_causal_lm_batch when you want a one-call path to padded causal-LM tensors.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "structured_outputs_and_batching.md"),
            section_title = "Supplemental / Training Ready LM Batches",
            category = "procedural",
            question = "How do I get training-ready LM batches in KeemenaSubwords.jl?",
            answer = "Use quick_causal_lm_batch for the one-call path from texts to ids, masks, and labels. Use the lower-level batching helpers when you already have tokenization results and want more control.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "concepts.md"),
            section_title = "Supplemental / Export",
            category = "procedural",
            question = "How do I export a tokenizer from KeemenaSubwords.jl?",
            answer = "Use export_tokenizer or save_tokenizer with the format you want. For HF-compatible fast-tokenizer loading, export to tokenizer.json with format=:hf_tokenizer_json.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "models.md"),
            section_title = "Supplemental / Built-in And Gated Models",
            category = "factual",
            question = "Does KeemenaSubwords.jl support both built-in and gated tokenizer models?",
            answer = "Yes. Some tokenizer models are shipped or publicly installable, while gated models require an explicit install_model! step and any needed upstream credentials.",
        ),
        (
            package = "KeemenaSubwords.jl",
            source_file = joinpath(subwords_root, "docs", "src", "integration.md"),
            section_title = "Supplemental / Integration",
            category = "procedural",
            question = "How does KeemenaSubwords.jl integrate with KeemenaPreprocessing.jl?",
            answer = "KeemenaSubwords tokenizers are callable and fit the tokenizer contract used by KeemenaPreprocessing. That makes it straightforward to preprocess text with KeemenaPreprocessing and then tokenize or batch it with KeemenaSubwords.",
        ),
    ]

    return [
        let normalized_answer = normalize_answer(pair.answer, pair.category; curated = true)
            ChatPair(
                id = bytes2hex(sha1(string(pair.package, "|", pair.source_file, "|", pair.section_title, "|", pair.question, "|", normalized_answer))),
                package = pair.package,
                source_file = pair.source_file,
                section_title = pair.section_title,
                category = pair.category,
                question = pair.question,
                answer = normalized_answer,
                chat_text = render_chat_text(pair.question, normalized_answer),
            )
        end for pair in raw_pairs
    ]
end

function deduplicate_pairs(pairs::Vector{ChatPair})::Vector{ChatPair}
    seen_fingerprints = Set{String}()
    by_question = Dict{String, ChatPair}()

    for pair in pairs
        fingerprint = lowercase(strip(pair.question)) * "\n" * lowercase(strip(pair.answer))
        fingerprint in seen_fingerprints && continue
        push!(seen_fingerprints, fingerprint)

        question_key = lowercase(strip(pair.question))
        existing = get(by_question, question_key, nothing)
        if existing === nothing || pair_rank(pair) < pair_rank(existing)
            by_question[question_key] = pair
        end
    end

    return collect(values(by_question))
end

function pair_rank(pair::ChatPair)
    answer = strip(pair.answer)
    lower_answer = lowercase(answer)
    is_curated = startswith(pair.section_title, "Supplemental /") ? 0 : 1
    punctuation_penalty = endswith(answer, '.') || endswith(answer, '!') || endswith(answer, '?') ? 0 : 1
    code_fragment_penalty = occursin("::", answer) || occursin("->", answer) || occursin(r"[A-Za-z_]+\(", answer) ? 1 : 0
    doc_fragment_penalty = startswith(lower_answer, "use ") ? 0 : 0
    return (is_curated, code_fragment_penalty, punctuation_penalty, length(answer), pair.id)
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

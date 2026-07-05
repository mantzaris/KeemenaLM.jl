const CHATBOT_BEHAVIOR_ROLE_MARKERS = ("User:", "Assistant:", "System:")

function chatbot_behavior_cases()
    return [
        (
            id = "greeting",
            prompt = "User: hello\nAssistant:",
            required_any = ["hi", "hello", "help"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 500,
        ),
        (
            id = "arithmetic_one_plus_one",
            prompt = "User: what is 1 plus 1?\nAssistant:",
            required_any = ["2", "two"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "arithmetic_two_plus_two",
            prompt = "User: what is 2 plus 2?\nAssistant:",
            required_any = ["4", "four"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "arithmetic_three_plus_four_symbol",
            prompt = "User: what is 3+4?\nAssistant:",
            required_any = ["7", "seven"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "green_is_color",
            prompt = "User: is green a color?\nAssistant:",
            required_any = String[],
            required_all = ["yes", "green", "color"],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "blue_is_color",
            prompt = "User: is blue a color?\nAssistant:",
            required_any = String[],
            required_all = ["yes", "blue", "color"],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "name_statement_ack",
            prompt = "User: My name is Alex.\nAssistant:",
            required_any = ["alex"],
            required_all = String[],
            forbidden = ["color", "capital", "answer is"],
            max_completion_characters = 300,
        ),
        (
            id = "name_recall",
            prompt = "User: my name is alex.\nAssistant: Nice to meet you, Alex.\n<END_ASSISTANT>\n<CHAT_END>\nUser: what is my name?\nAssistant:",
            required_any = ["alex"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "usa_democracy",
            prompt = "User: I live in the USA. Is the USA a democracy or a dictatorship?\nAssistant:",
            required_any = ["democracy", "democratic", "constitutional republic"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 500,
        ),
        (
            id = "france_capital",
            prompt = "User: what is the capital of France?\nAssistant:",
            required_any = ["paris"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "japan_capital",
            prompt = "User: what is the capital of Japan?\nAssistant:",
            required_any = ["tokyo"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "japan_capital_lowercase",
            prompt = "User: what is the capital of japan?\nAssistant:",
            required_any = ["tokyo"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "canada_capital_lowercase",
            prompt = "User: what is the capital of canada?\nAssistant:",
            required_any = ["ottawa"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "plato_fact",
            prompt = "User: Who was Plato?\nAssistant:",
            required_any = ["plato", "philosopher", "greek"],
            required_all = String[],
            forbidden = ["answer is", "capital of"],
            max_completion_characters = 500,
        ),
        (
            id = "car_wheels",
            prompt = "User: How many wheels on a car?\nAssistant:",
            required_any = ["4", "four"],
            required_all = ["wheel"],
            forbidden = ["seven days", "capital"],
            max_completion_characters = 400,
        ),
        (
            id = "meaning_of_life",
            prompt = "User: What is the meaning of life?\nAssistant:",
            required_any = ["meaning", "purpose", "people", "depends", "different"],
            required_all = String[],
            forbidden = ["capital of", "answer is"],
            max_completion_characters = 600,
        ),
        (
            id = "private_unknown",
            prompt = "User: What is my favorite food?\nAssistant:",
            required_any = ["do not know", "don't know", "not know", "tell me"],
            required_all = String[],
            forbidden = ["capital", "answer is"],
            max_completion_characters = 500,
        ),
        (
            id = "kind_rewrite",
            prompt = "User: rewrite this kindly: You forgot again.\nAssistant:",
            required_any = ["reminder", "remember", "check", "when you have a chance", "could you"],
            required_all = String[],
            forbidden = ["you forgot again"],
            max_completion_characters = 500,
        ),
        (
            id = "simple_dinner_ideas",
            prompt = "User: give me two simple dinner ideas.\nAssistant:",
            required_any = ["dinner", "pasta", "rice", "eggs", "salad", "soup", "tacos", "sandwich", "potato", "omelet"],
            required_all = String[],
            forbidden = String[],
            max_completion_characters = 600,
        ),
        (
            id = "john_is_name",
            prompt = "User: is John a name?\nAssistant:",
            required_any = ["yes"],
            required_all = ["john", "name"],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "john_is_name_lowercase",
            prompt = "User: is john a name?\nAssistant:",
            required_any = ["yes"],
            required_all = ["john", "name"],
            forbidden = String[],
            max_completion_characters = 300,
        ),
        (
            id = "cats_chat",
            prompt = "User: I like cats\nAssistant:",
            required_any = ["cat", "cats"],
            required_all = String[],
            forbidden = ["capital", "answer is"],
            max_completion_characters = 400,
        ),
    ]
end

function score_chatbot_behavior_completion(case, completion::AbstractString)::Dict{String,Any}
    completion_text = String(completion)
    required_any = String[value for value in behavior_case_value(case, :required_any, String[])]
    required_all = String[value for value in behavior_case_value(case, :required_all, String[])]
    forbidden = String[value for value in behavior_case_value(case, :forbidden, String[])]
    max_completion_characters = Int(behavior_case_value(case, :max_completion_characters, 0))

    stripped_completion = strip(completion_text)
    empty_completion = isempty(stripped_completion)
    role_leakage = chatbot_behavior_role_leakage(completion_text)
    repetition_loop = chatbot_behavior_repetition_loop(completion_text)
    too_long = max_completion_characters > 0 && length(completion_text) > max_completion_characters
    missing_required_any = !isempty(required_any) && !any(phrase -> chatbot_behavior_phrase_present(completion_text, phrase), required_any)
    missing_required_all = String[phrase for phrase in required_all if !chatbot_behavior_phrase_present(completion_text, phrase)]
    forbidden_hits = String[phrase for phrase in forbidden if chatbot_behavior_phrase_present(completion_text, phrase)]

    failures = String[]
    empty_completion && push!(failures, "empty_completion")
    role_leakage && push!(failures, "role_leakage")
    repetition_loop && push!(failures, "repetition_loop")
    too_long && push!(failures, "too_long")
    missing_required_any && push!(failures, "missing_required_any")
    !isempty(missing_required_all) && push!(failures, "missing_required_all")
    !isempty(forbidden_hits) && push!(failures, "forbidden_phrase")

    return Dict(
        "case_id" => String(behavior_case_value(case, :id, "")),
        "passed" => isempty(failures),
        "failures" => failures,
        "completion_characters" => length(completion_text),
        "empty_completion" => empty_completion,
        "role_leakage" => role_leakage,
        "repetition_loop" => repetition_loop,
        "too_long" => too_long,
        "missing_required_any" => missing_required_any,
        "missing_required_all" => missing_required_all,
        "forbidden_hits" => forbidden_hits,
    )
end

function score_chatbot_behavior_suite(completions; cases = chatbot_behavior_cases())::Dict{String,Any}
    case_results = Dict{String,Any}[]
    for (case_index, case) in enumerate(cases)
        case_id = String(behavior_case_value(case, :id, string(case_index)))
        completion = behavior_completion_for_case(completions, case_id, case_index)
        push!(case_results, score_chatbot_behavior_completion(case, completion))
    end

    passed_count = count(result -> Bool(result["passed"]), case_results)
    total_count = length(case_results)
    return Dict(
        "passed" => passed_count == total_count,
        "passed_count" => passed_count,
        "total_count" => total_count,
        "pass_rate" => total_count == 0 ? 0.0 : passed_count / total_count,
        "failed_case_ids" => String[String(result["case_id"]) for result in case_results if !Bool(result["passed"])],
        "case_results" => case_results,
    )
end

function behavior_case_value(case::NamedTuple, key::Symbol, default)
    return haskey(case, key) ? getfield(case, key) : default
end

function behavior_case_value(case::AbstractDict, key::Symbol, default)
    string_key = String(key)
    if haskey(case, key)
        return case[key]
    elseif haskey(case, string_key)
        return case[string_key]
    else
        return default
    end
end

behavior_case_value(case, key::Symbol, default) = hasproperty(case, key) ? getproperty(case, key) : default

function behavior_completion_for_case(completions::AbstractDict, case_id::AbstractString, case_index::Int)
    if haskey(completions, case_id)
        return String(completions[case_id])
    elseif haskey(completions, Symbol(case_id))
        return String(completions[Symbol(case_id)])
    elseif haskey(completions, case_index)
        return String(completions[case_index])
    else
        return ""
    end
end

function behavior_completion_for_case(completions::AbstractVector, case_id::AbstractString, case_index::Int)
    case_index <= length(completions) || return ""
    return String(completions[case_index])
end

function chatbot_behavior_role_leakage(completion::AbstractString)::Bool
    stripped = strip(completion)
    for marker in CHATBOT_BEHAVIOR_ROLE_MARKERS
        startswith(stripped, marker) && return true
        occursin("\n" * marker, completion) && return true
    end
    return false
end

function chatbot_behavior_repetition_loop(completion::AbstractString)::Bool
    safe_completion = chatbot_behavior_safe_lower_ascii(completion)
    matches = collect(eachmatch(r"[a-z0-9]+", safe_completion))
    word_values = String[match.match for match in matches]
    length(word_values) >= 4 || return false

    run_word = word_values[1]
    run_count = 1
    for word in word_values[2:end]
        if word == run_word
            run_count += 1
            run_count >= 4 && return true
        else
            run_word = word
            run_count = 1
        end
    end

    for ngram_size in 2:4
        length(word_values) >= ngram_size * 3 || continue
        for start_index in 1:(length(word_values) - ngram_size * 3 + 1)
            first_ngram = word_values[start_index:(start_index + ngram_size - 1)]
            second_ngram = word_values[(start_index + ngram_size):(start_index + 2 * ngram_size - 1)]
            third_ngram = word_values[(start_index + 2 * ngram_size):(start_index + 3 * ngram_size - 1)]
            first_ngram == second_ngram == third_ngram && return true
        end
    end

    return false
end

function chatbot_behavior_phrase_present(text::AbstractString, phrase::AbstractString)::Bool
    normalized_text = chatbot_behavior_normalize(text)
    normalized_phrase = chatbot_behavior_normalize(phrase)
    isempty(normalized_phrase) && return true
    return occursin(normalized_phrase, normalized_text)
end

function chatbot_behavior_normalize(text::AbstractString)::String
    return strip(replace(chatbot_behavior_safe_lower_ascii(text), r"\s+" => " "))
end

function chatbot_behavior_safe_lower_ascii(text::AbstractString)::String
    io = IOBuffer()
    for byte in codeunits(String(text))
        if 0x41 <= byte <= 0x5a
            write(io, UInt8(byte + 0x20))
        elseif byte <= 0x7f
            write(io, byte)
        else
            write(io, UInt8(' '))
        end
    end
    return String(take!(io))
end

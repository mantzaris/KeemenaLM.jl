using Test

@testset "chatbot behavior gate" begin
    cases = KeemenaLM.Core.chatbot_behavior_cases()
    arithmetic_case = only(filter(case -> case.id == "arithmetic_one_plus_one", cases))
    arithmetic_score = KeemenaLM.Core.score_chatbot_behavior_completion(arithmetic_case, " 2")
    @test arithmetic_score["passed"]

    leakage_score = KeemenaLM.Core.score_chatbot_behavior_completion(arithmetic_case, "User: what next?")
    @test !leakage_score["passed"]
    @test "role_leakage" in leakage_score["failures"]

    loop_score = KeemenaLM.Core.score_chatbot_behavior_completion(arithmetic_case, "q q q q q")
    @test !loop_score["passed"]
    @test "repetition_loop" in loop_score["failures"]

    missing_score = KeemenaLM.Core.score_chatbot_behavior_completion(arithmetic_case, "I am not sure.")
    @test !missing_score["passed"]
    @test "missing_required_any" in missing_score["failures"]

    invalid_text = String(UInt8[0xa5, 0x20, 0x71, 0x20, 0x71, 0x20, 0x71, 0x20, 0x71])
    invalid_score = KeemenaLM.Core.score_chatbot_behavior_completion(arithmetic_case, invalid_text)
    @test !invalid_score["passed"]
    @test "repetition_loop" in invalid_score["failures"]

    completions = Dict(
        "greeting" => "Hi. What would you like help with?",
        "arithmetic_one_plus_one" => "2",
        "arithmetic_two_plus_two" => "4",
        "arithmetic_three_plus_four_symbol" => "7",
        "green_is_color" => "Yes, green is a color.",
        "blue_is_color" => "Yes, blue is a color.",
        "name_statement_ack" => "Nice to meet you, Alex.",
        "name_recall" => "Your name is Alex.",
        "usa_democracy" => "The USA is a democracy and constitutional republic.",
        "france_capital" => "Paris is the capital of France.",
        "japan_capital" => "Tokyo is the capital of Japan.",
        "japan_capital_lowercase" => "Tokyo is the capital of Japan.",
        "canada_capital_lowercase" => "Ottawa is the capital of Canada.",
        "plato_fact" => "Plato was a Greek philosopher.",
        "car_wheels" => "A typical car has four wheels.",
        "meaning_of_life" => "People answer the meaning of life differently; it often involves purpose and relationships.",
        "private_unknown" => "I do not know your favorite food. Tell me and I can remember it for this chat.",
        "kind_rewrite" => "Could you check this when you have a chance?",
        "simple_dinner_ideas" => "Two simple dinner ideas are pasta and soup.",
        "john_is_name" => "Yes, John is a name.",
        "john_is_name_lowercase" => "Yes, john is a name.",
        "cats_chat" => "Cats are nice. What do you like about cats?",
    )
    suite = KeemenaLM.Core.score_chatbot_behavior_suite(completions; cases = cases)
    @test suite["passed"]
    @test suite["passed_count"] == suite["total_count"]

    bad_suite = KeemenaLM.Core.score_chatbot_behavior_suite(Dict("greeting" => "User: hi"); cases = cases)
    @test !bad_suite["passed"]
    @test "greeting" in bad_suite["failed_case_ids"]
end

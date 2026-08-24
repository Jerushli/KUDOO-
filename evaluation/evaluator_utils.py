def result_matches(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False

        for key, expected_value in expected.items():
            if key not in actual:
                return False

            actual_value = actual[key]
            if isinstance(expected_value, float):
                if not isinstance(actual_value, (int, float)):
                    return False
                if abs(actual_value - expected_value) > 0.1:
                    return False
            elif actual_value != expected_value:
                return False

        return True

    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and abs(actual - expected) <= 0.1
        )

    return actual == expected


def arguments_match(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False

    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(actual_value, str) and isinstance(expected_value, str):
            if actual_value.strip().lower() != expected_value.strip().lower():
                return False
        elif actual_value != expected_value:
            return False

    return True


def answer_matches(answer, expected_phrases):
    if not answer:
        return False

    normalized = answer.lower().replace(",", "").replace("$", "")
    return all(
        phrase.lower().replace(",", "").replace("$", "") in normalized
        for phrase in expected_phrases
    )


def print_failure(test, result):
    print(f"  Question: {test['question']}")
    print(f"  Expected tool: {test.get('expected_tool')}")
    print(f"  Actual tools: {[call['tool'] for call in result.get('tool_calls', [])]}")
    print(f"  Expected arguments: {test.get('expected_arguments')}")
    print(
        "  Actual arguments: "
        f"{result.get('tool_calls', [{}])[0].get('arguments') if result.get('tool_calls') else None}"
    )
    print(f"  Answer: {result.get('answer')}")
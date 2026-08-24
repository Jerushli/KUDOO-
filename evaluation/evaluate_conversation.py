import uuid

from app.agent import memory, run_agent

from .evaluator_utils import arguments_match
from .evaluate_agent import load_test_cases


CONTEXT_TESTS = [
    {
        "id": "CTX001",
        "questions": [
            "Which product made the most money?",
            "How much did it make?",
            "How much more did it make than Phone?",
            "What percentage of total revenue did it contribute?",
        ],
        "expected_tools": [
            "get_best_product",
            "get_product_revenue",
            "get_revenue_difference",
            "get_percentage_of_total",
        ],
        "expected_arguments": [
            {},
            {"product": "Laptop"},
            {
                "first_product": "Laptop",
                "second_product": "Phone",
            },
            {
                "column": "revenue",
                "filter_column": "product",
                "filter_value": "Laptop",
            },
        ],
    },
]


def run_context_test(test):
    conversation_id = f"{test['id']}-{uuid.uuid4()}"
    results = []

    try:
        for index, question in enumerate(test["questions"]):
            result = run_agent(
                user_question=question,
                conversation_id=conversation_id,
            )
            calls = result["tool_calls"]
            first_call = calls[0] if calls else {}
            expected_tool = test["expected_tools"][index]
            expected_arguments = test["expected_arguments"][index]

            results.append(
                {
                    "question": question,
                    "tool": first_call.get("tool") == expected_tool,
                    "arguments": arguments_match(
                        first_call.get("arguments", {}),
                        expected_arguments,
                    ),
                    "actual_tool": first_call.get("tool"),
                    "actual_arguments": first_call.get("arguments"),
                    "answer": result["answer"],
                }
            )
    finally:
        memory.clear(conversation_id)

    return results


def run_evaluation():
    # Loading the cases here verifies the shared single-turn test data remains valid.
    cases = load_test_cases()
    print(f"Loaded {len(cases)} shared single-turn cases.")

    total = 0
    tool_passes = 0
    argument_passes = 0

    for test in CONTEXT_TESTS:
        results = run_context_test(test)
        for result in results:
            total += 1
            tool_passes += int(result["tool"])
            argument_passes += int(result["arguments"])
            print(f"{test['id']}: {result['question']}")
            print(f"  Tool: {'PASS' if result['tool'] else 'FAIL'}")
            print(
                "  Arguments: "
                f"{'PASS' if result['arguments'] else 'FAIL'}"
            )
            if not result["tool"] or not result["arguments"]:
                print(f"  Actual tool: {result['actual_tool']}")
                print(f"  Actual arguments: {result['actual_arguments']}")
            print(f"  Answer: {result['answer']}")

    print("\nKUDOO CONVERSATION EVALUATION")
    print(f"Tool selection: {tool_passes}/{total} ({tool_passes / total * 100:.1f}%)")
    print(
        f"Arguments:      {argument_passes}/{total} "
        f"({argument_passes / total * 100:.1f}%)"
    )


if __name__ == "__main__":
    run_evaluation()
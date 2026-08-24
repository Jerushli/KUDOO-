import uuid

from app.agent import memory, run_agent

from .evaluator_utils import arguments_match, answer_matches


TEST_CASES = [
    {
        "id": "MS001",
        "question": (
            "Which product generated the most revenue, and what percentage "
            "of total revenue did it contribute?"
        ),
        "expected_tools": ["get_best_product", "get_percentage_of_total"],
        "expected_arguments": [
            {},
            {
                "column": "revenue",
                "filter_column": "product",
                "filter_value": "Laptop",
            },
        ],
        "expected_answer": ["laptop", "39.9"],
    },
    {
        "id": "MS002",
        "question": "Which region generated the most revenue, and how much did it make?",
        "expected_tools": ["get_best_region", "get_revenue_by_region"],
        "expected_arguments": [{}, {"region": "North"}],
        "expected_answer": ["north", "16800"],
    },
    {
        "id": "MS003",
        "question": "Which product made the most revenue and how much more did it make than Phone?",
        "expected_tools": ["get_best_product", "get_revenue_difference"],
        "expected_arguments": [
            {},
            {"first_product": "Laptop", "second_product": "Phone"},
        ],
        "expected_answer": ["laptop", "4200"],
    },
]


def run_evaluation():
    totals = {"tools": 0, "arguments": 0, "answers": 0}

    for test in TEST_CASES:
        conversation_id = f"{test['id']}-{uuid.uuid4()}"
        try:
            result = run_agent(
                user_question=test["question"],
                conversation_id=conversation_id,
            )
        finally:
            memory.clear(conversation_id)

        calls = result["tool_calls"]
        actual_tools = [call["tool"] for call in calls]
        tools_ok = actual_tools == test["expected_tools"]
        arguments_ok = (
            len(calls) == len(test["expected_arguments"])
            and all(
                arguments_match(call["arguments"], expected)
                for call, expected in zip(
                    calls,
                    test["expected_arguments"],
                )
            )
        )
        answer_ok = answer_matches(
            result["answer"],
            test["expected_answer"],
        )

        totals["tools"] += int(tools_ok)
        totals["arguments"] += int(arguments_ok)
        totals["answers"] += int(answer_ok)

        print(f"{test['id']}: {test['question']}")
        print(f"  Tools: {'PASS' if tools_ok else 'FAIL'} {actual_tools}")
        print(f"  Arguments: {'PASS' if arguments_ok else 'FAIL'}")
        print(f"  Answer: {'PASS' if answer_ok else 'FAIL'} {result['answer']}")

    total = len(TEST_CASES)
    print("\nKUDOO MULTI-STEP EVALUATION")
    for key, value in totals.items():
        print(f"{key.title():10} {value}/{total} ({value / total * 100:.1f}%)")


if __name__ == "__main__":
    run_evaluation()
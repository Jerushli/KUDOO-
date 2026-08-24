import json
import uuid

from app.agent import run_agent
from app.tool_executor import execute_tool
from app.tools import load_data

from .evaluator_utils import (
    answer_matches,
    arguments_match,
    print_failure,
    result_matches,
)


EXPECTED_ANSWERS = {
    "TC001": ["laptop", "16800"],
    "TC002": ["phone", "12600"],
    "TC003": ["laptop", "16800"],
    "TC004": ["north", "16800"],
    "TC005": ["105"],
    "TC006": ["4200"],
    "TC007": ["39.9", "laptop"],
    "TC008": ["north", "16800"],
    "TC009": ["headphones", "laptop", "monitor", "phone"],
    "TC010": ["tesla", "not found"],
    "TC011": ["laptop", "16800"],
    "TC012": ["42100"],
    "TC013": ["4200"],
    "TC014": ["29.9", "phone"],
    "TC015": ["apple", "not found"],
}


def load_test_cases():
    with open("evaluation/test_cases.json", encoding="utf-8") as file:
        return json.load(file)


def run_single_test(df, test):
    result = run_agent(
        user_question=test["question"],
        conversation_id=f"evaluation-{uuid.uuid4()}",
    )
    calls = result["tool_calls"]
    first_call = calls[0] if calls else {}

    expected_result = execute_tool(
        tool_name=test["expected_tool"],
        df=df,
        arguments=test["expected_arguments"],
    )

    return {
        "tool": first_call.get("tool") == test["expected_tool"],
        "arguments": arguments_match(
            first_call.get("arguments", {}),
            test["expected_arguments"],
        ),
        "result": result_matches(
            first_call.get("result"),
            expected_result,
        ),
        "answer": answer_matches(
            result["answer"],
            EXPECTED_ANSWERS[test["id"]],
        ),
        "agent_result": result,
    }


def run_evaluation():
    df = load_data()
    cases = load_test_cases()
    totals = {"tool": 0, "arguments": 0, "result": 0, "answer": 0}

    for test in cases:
        result = run_single_test(df, test)
        print(f"{test['id']}: {test['question']}")

        for key in totals:
            totals[key] += int(result[key])
            print(f"  {key}: {'PASS' if result[key] else 'FAIL'}")

        if not all(result[key] for key in totals):
            print_failure(test, result["agent_result"])

    total = len(cases)
    print("\nKUDOO AGENT EVALUATION")
    for key, value in totals.items():
        print(f"{key.title():12} {value}/{total} ({value / total * 100:.1f}%)")

    overall = sum(totals.values()) / (total * len(totals)) * 100
    print(f"Overall       {overall:.1f}%")


if __name__ == "__main__":
    run_evaluation()
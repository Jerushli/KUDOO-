from types import SimpleNamespace

import pytest

from app import agent


def tool_response(name, arguments):
    return SimpleNamespace(
        message=SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name=name,
                        arguments=arguments,
                    )
                )
            ]
        )
    )


def multi_tool_response(*calls):
    return SimpleNamespace(
        message=SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name=name,
                        arguments=arguments,
                    )
                )
                for name, arguments in calls
            ]
        )
    )


def final_response():
    return SimpleNamespace(
        message=SimpleNamespace(
            tool_calls=[],
            content="",
        )
    )


@pytest.mark.parametrize(
    ("question", "responses", "expected_tools", "expected_answer"),
    [
        (
            "Which product generated the most revenue, and what percentage of total revenue did it contribute?",
            [
                tool_response("get_best_product", {}),
                tool_response(
                    "get_percentage_of_total",
                    {
                        "column": "revenue",
                        "filter_column": "product",
                        "filter_value": "Laptop",
                    },
                ),
                final_response(),
            ],
            ["get_best_product", "get_percentage_of_total"],
            ["Laptop", "39.9%"],
        ),
        (
            "Which region generated the most revenue, and how much did it make?",
            [
                tool_response("get_best_region", {}),
                tool_response("get_revenue_by_region", {"region": "North"}),
                final_response(),
            ],
            ["get_best_region", "get_revenue_by_region"],
            ["North", "$16,800"],
        ),
        (
            "Which product made the most revenue and how much more did it make than Phone?",
            [
                tool_response("get_best_product", {}),
                tool_response(
                    "get_revenue_difference",
                    {
                        "first_product": "Laptop",
                        "second_product": "Phone",
                    },
                ),
                final_response(),
            ],
            ["get_best_product", "get_revenue_difference"],
            ["Laptop", "$4,200"],
        ),
    ],
)
def test_dependent_tool_workflows(
    monkeypatch,
    question,
    responses,
    expected_tools,
    expected_answer,
):
    conversation_id = f"workflow-{expected_tools[0]}"
    response_queue = iter(responses)

    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda _: next(response_queue),
    )

    try:
        result = agent.run_agent(
            user_question=question,
            conversation_id=conversation_id,
        )

        assert [call["tool"] for call in result["tool_calls"]] == expected_tools
        assert all(value in result["answer"] for value in expected_answer)
        assert [step["tool"] for step in result["trace"]["steps"]] == expected_tools
    finally:
        agent.memory.clear(conversation_id)


def test_tool_loop_stops_at_maximum(monkeypatch):
    conversation_id = "workflow-max-calls"
    calls = [
        tool_response("get_total_revenue", {})
        for _ in range(agent.MAX_TOOL_CALLS + 1)
    ]
    response_queue = iter(calls)

    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda _: next(response_queue),
    )

    try:
        result = agent.run_agent(
            user_question="What is the total revenue?",
            conversation_id=conversation_id,
        )

        assert len(result["tool_calls"]) == agent.MAX_TOOL_CALLS
        assert len(result["trace"]["steps"]) == agent.MAX_TOOL_CALLS
        assert "Total revenue was $42,100." in result["answer"]
    finally:
        agent.memory.clear(conversation_id)


def test_ollama_failure_returns_clean_answer(monkeypatch):
    conversation_id = "workflow-ollama-failure"

    def fail_call(_):
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(agent.ollama_client, "chat", fail_call)

    try:
        result = agent.run_agent(
            user_question="What is the total revenue?",
            conversation_id=conversation_id,
        )

        assert result["tool_calls"] == []
        assert "could not reach the local AI model" in result["answer"]
    finally:
        agent.memory.clear(conversation_id)


def test_dependent_arguments_are_bound_to_verified_product(monkeypatch):
    conversation_id = "workflow-verified-product"
    responses = iter(
        [
            tool_response("get_best_product", {}),
            tool_response(
                "get_percentage_of_total",
                {"filter_column": "product", "filter_value": "best_product"},
            ),
            final_response(),
        ]
    )

    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))

    try:
        result = agent.run_agent(
            "Which product made the most revenue and what percentage did it contribute?",
            conversation_id,
        )

        assert result["tool_calls"][1]["arguments"] == {
            "column": "revenue",
            "filter_column": "product",
            "filter_value": "Laptop",
        }
    finally:
        agent.memory.clear(conversation_id)


def test_dependent_arguments_are_bound_to_verified_region(monkeypatch):
    conversation_id = "workflow-verified-region"
    responses = iter(
        [
            tool_response("get_best_region", {}),
            tool_response("get_revenue_by_region", {"region": "best_region"}),
            final_response(),
        ]
    )

    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))

    try:
        result = agent.run_agent(
            "Which region made the most revenue and how much did it make?",
            conversation_id,
        )

        assert result["tool_calls"][1]["arguments"] == {"region": "North"}
    finally:
        agent.memory.clear(conversation_id)


def test_explicit_product_does_not_trigger_discovery(monkeypatch):
    conversation_id = "workflow-explicit-product"
    responses = iter(
        [
            tool_response(
                "get_percentage_of_total",
                {
                    "column": "revenue",
                    "filter_column": "product",
                    "filter_value": "Laptop",
                },
            ),
            final_response(),
        ]
    )

    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))

    try:
        result = agent.run_agent(
            "What share of our total revenue came from Laptop sales?",
            conversation_id,
        )

        assert [call["tool"] for call in result["tool_calls"]] == [
            "get_percentage_of_total"
        ]
    finally:
        agent.memory.clear(conversation_id)


def test_successful_discovery_is_not_repeated(monkeypatch):
    conversation_id = "workflow-no-duplicate-discovery"
    responses = iter(
        [
            tool_response("get_best_region", {}),
            tool_response("get_best_region", {}),
            tool_response("get_revenue_by_region", {"region": "North"}),
            final_response(),
        ]
    )

    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))

    try:
        result = agent.run_agent(
            "Which region generated the most revenue?",
            conversation_id,
        )

        assert [call["tool"] for call in result["tool_calls"]] == [
            "get_best_region",
            "get_revenue_by_region",
        ]
    finally:
        agent.memory.clear(conversation_id)


def test_one_explicit_product_still_requires_best_product(monkeypatch):
    conversation_id = "workflow-missing-best-product"
    responses = iter(
        [
            tool_response(
                "get_revenue_difference",
                {
                    "first_product": "get_best_product",
                    "second_product": "Phone",
                },
            ),
            tool_response(
                "get_revenue_difference",
                {
                    "first_product": "Laptop",
                    "second_product": "Phone",
                },
            ),
            final_response(),
        ]
    )

    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))

    try:
        result = agent.run_agent(
            "Which product made the most revenue and how much more did it make than Phone?",
            conversation_id,
        )

        assert [call["tool"] for call in result["tool_calls"]] == [
            "get_best_product",
            "get_revenue_difference",
        ]
    finally:
        agent.memory.clear(conversation_id)


@pytest.mark.parametrize(
    ("question", "calls", "expected_tools", "expected_arguments"),
    [
        (
            "Which product generated the most revenue, and what percentage of total revenue did it contribute?",
            [
                ("get_best_product", {}),
                (
                    "get_percentage_of_total",
                    {
                        "column": "revenue",
                        "filter_column": "product",
                        "filter_value": "get_best_product",
                    },
                ),
            ],
            ["get_best_product", "get_percentage_of_total"],
            {
                "column": "revenue",
                "filter_column": "product",
                "filter_value": "Laptop",
            },
        ),
        (
            "Which region generated the most revenue, and how much did it make?",
            [
                ("get_best_region", {}),
                ("get_revenue_by_region", {"region": "get_best_region"}),
            ],
            ["get_best_region", "get_revenue_by_region"],
            {"region": "North"},
        ),
        (
            "Which product made the most revenue and how much more did it make than Phone?",
            [
                ("get_best_product", {}),
                (
                    "get_revenue_difference",
                    {
                        "product_a": "get_best_product",
                        "product_b": "Phone",
                    },
                ),
            ],
            ["get_best_product", "get_revenue_difference"],
            {
                "first_product": "Laptop",
                "second_product": "Phone",
            },
        ),
    ],
)
def test_placeholder_dependency_in_same_response(
    monkeypatch,
    question,
    calls,
    expected_tools,
    expected_arguments,
):
    conversation_id = f"workflow-placeholder-{expected_tools[0]}"
    responses = iter(
        [
            multi_tool_response(*calls),
            final_response(),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))

    try:
        result = agent.run_agent(question, conversation_id)
        assert [call["tool"] for call in result["tool_calls"]] == expected_tools
        assert result["tool_calls"][1]["arguments"] == expected_arguments
    finally:
        agent.memory.clear(conversation_id)


def test_pending_dependency_continues_after_prerequisite(monkeypatch):
    conversation_id = "workflow-pending-dependency"
    responses = iter(
        [
            tool_response(
                "get_percentage_of_total",
                {
                    "column": "revenue",
                    "filter_column": "product",
                    "filter_value": "get_best_product",
                },
            ),
            final_response(),
        ]
    )

    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))

    try:
        result = agent.run_agent(
            "Which product generated the most revenue, and what percentage of total revenue did it contribute?",
            conversation_id,
        )

        assert [call["tool"] for call in result["tool_calls"]] == [
            "get_best_product",
            "get_percentage_of_total",
        ]
        assert result["tool_calls"][1]["arguments"] == {
            "column": "revenue",
            "filter_column": "product",
            "filter_value": "Laptop",
        }
    finally:
        agent.memory.clear(conversation_id)


def test_missing_intermediate_result_stops_dependent_workflow(monkeypatch):
    conversation_id = "workflow-missing-result"
    responses = iter(
        [
            tool_response("get_best_product", {}),
            tool_response("get_percentage_of_total", {}),
        ]
    )

    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))
    monkeypatch.setattr(
        agent,
        "execute_tool",
        lambda tool_name, df, arguments: {
            "success": False,
            "error_type": "MISSING_DEPENDENCY",
            "message": "A verified product is required.",
        }
        if tool_name == "get_best_product"
        else {"success": False, "error_type": "INVALID_ARGUMENTS"},
    )

    try:
        result = agent.run_agent("Find the best product percentage.", conversation_id)
        assert len(result["tool_calls"]) == 1
        assert "required" in result["answer"]
    finally:
        agent.memory.clear(conversation_id)


@pytest.mark.parametrize(
    "failure_tool",
    ["get_best_product", "get_percentage_of_total"],
)
def test_tool_failure_stops_workflow(monkeypatch, failure_tool):
    conversation_id = f"workflow-failure-{failure_tool}"
    responses = iter(
        [
            tool_response("get_best_product", {}),
            tool_response("get_percentage_of_total", {}),
        ]
    )

    monkeypatch.setattr(agent, "call_llm", lambda _: next(responses))

    def execute(tool_name, df, arguments):
        if tool_name == failure_tool:
            return {
                "success": False,
                "error_type": "TEST_FAILURE",
                "message": f"{failure_tool} failed",
            }
        return {"product": "Laptop", "revenue": 16800.0}

    monkeypatch.setattr(agent, "execute_tool", execute)

    try:
        result = agent.run_agent("Run a dependent analysis.", conversation_id)
        failure_index = 0 if failure_tool == "get_best_product" else 1
        assert result["tool_calls"][failure_index]["result"]["success"] is False
        assert failure_tool in result["answer"]
    finally:
        agent.memory.clear(conversation_id)

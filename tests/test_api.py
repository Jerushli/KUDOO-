from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_empty_messages_are_rejected_before_agent(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_agent must not run for invalid input")

    monkeypatch.setattr("app.api.run_agent", fail_if_called)

    for message in ["", "   ", "\n"]:
        response = client.post(
            "/chat",
            json={"message": message},
        )

        assert response.status_code == 422
        assert "message must not be empty" in response.text


def test_valid_message_reaches_agent(monkeypatch):
    monkeypatch.setattr(
        "app.api.run_agent",
        lambda user_question, conversation_id: {
            "answer": "ok",
            "conversation_id": conversation_id,
            "tool_calls": [],
            "execution_time": 0,
            "conversation": [],
            "trace": {},
        },
    )

    response = client.post(
        "/chat",
        json={"message": "  What is total revenue?  "},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
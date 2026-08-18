import json
import time
from datetime import datetime


class AgentTracer:
    """
    Tracks KUDOO's execution steps during an agent run.
    """

    def __init__(self):
        self.steps = []
        self.start_time = None

    def start(self, user_question):
        """
        Start a new trace.
        """

        self.steps = []

        self.start_time = time.perf_counter()

        self.user_question = user_question

    def record_tool_call(
        self,
        tool_name,
        arguments,
        result,
        duration,
    ):
        """
        Record one tool execution.
        """

        success = True

        if isinstance(result, dict):
            success = result.get("success", True)

            if result.get("error_type"):
                success = False

        step = {
            "step": len(self.steps) + 1,
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "tool": tool_name,
            "arguments": arguments,
            "status": "SUCCESS" if success else "FAILED",
            "duration_ms": round(
                duration * 1000,
                2
            ),
            "result": result,
        }

        self.steps.append(step)

    def finish(self, final_answer):
        """
        Finish the trace.
        """

        total_duration = (
            time.perf_counter() - self.start_time
        )

        return {
            "user_question": self.user_question,
            "steps": self.steps,
            "final_answer": final_answer,
            "total_duration_ms": round(
                total_duration * 1000,
                2
            ),
        }

    def print_trace(self, trace):
        """
        Print a human-readable trace.
        """

        print()
        print("=" * 60)
        print("                 KUDOO AGENT TRACE")
        print("=" * 60)

        print()
        print("User Query:")
        print(trace["user_question"])

        print()

        for step in trace["steps"]:

            print("-" * 60)

            print(f"Step {step['step']}")

            print(
                f"Tool: {step['tool']}"
            )

            print(
                "Arguments:"
            )

            print(
                json.dumps(
                    step["arguments"],
                    indent=4
                )
            )

            print(
                f"Status: {step['status']}"
            )

            print(
                f"Duration: {step['duration_ms']} ms"
            )

            print(
                "Result:"
            )

            print(
                json.dumps(
                    step["result"],
                    indent=4
                )
            )

        print()
        print("-" * 60)

        print("Final Answer:")
        print(trace["final_answer"])

        print()
        print(
            f"Total execution time: "
            f"{trace['total_duration_ms']} ms"
        )

        print("=" * 60)
        print()
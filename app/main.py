import json
import time
import logging

from app.logging_config import setup_logging

from ollama import chat

from .tools import load_data
from .tool_registry import (
    TOOLS,
    AVAILABLE_TOOLS,
)
from .tracing import AgentTracer


# --------------------------------------------------
# Logging setup
# --------------------------------------------------

setup_logging()

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Load sales data once
# --------------------------------------------------

df = load_data()

logger.info("Sales dataset loaded successfully")


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are KUDOO, a helpful AI data analyst.

You analyze the provided sales dataset using tools.

IMPORTANT RULES:

1. Always use a tool when the answer requires information
   from the dataset.

2. Never invent numbers.

3. Use previous conversation context when interpreting
   follow-up questions.

4. If a question requires multiple pieces of information,
   use multiple tools when necessary.

5. For one specific product's revenue, use
   get_product_revenue.

6. For revenue for every product, use
   get_revenue_by_product.

7. For a percentage of total revenue, use
   get_percentage_of_total.

8. When a tool returns an error, explain the error clearly
   and do not invent an answer.

9. After all required tools have been executed, provide
   one concise final answer combining the results.
"""


# --------------------------------------------------
# Execute one tool
# --------------------------------------------------

def execute_tool(tool_call):
    """
    Execute a tool selected by the LLM.
    """

    tool_name = tool_call.function.name
    arguments = tool_call.function.arguments or {}

    print(
        f"\n🔧 Tool selected: {tool_name}"
    )

    print(
        f"🧩 Tool arguments: {arguments}"
    )

    logger.info(
        "Tool selected: %s | Arguments: %s",
        tool_name,
        arguments,
    )

    function_to_call = AVAILABLE_TOOLS.get(
        tool_name
    )

    if function_to_call is None:

        result = {
            "success": False,
            "error_type": "UNKNOWN_TOOL",
            "message": (
                f"Tool '{tool_name}' is not registered."
            ),
        }

        print(
            f"📊 Tool result: {result}"
        )

        logger.error(
            "Unknown tool requested: %s",
            tool_name,
        )

        return (
            tool_name,
            arguments,
            result,
        )

    try:

        result = function_to_call(
            df,
            **arguments,
        )

    except Exception as e:

        result = {
            "success": False,
            "error_type": "TOOL_EXECUTION_ERROR",
            "message": str(e),
        }

        logger.exception(
            "Tool execution failed: %s",
            tool_name,
        )

    else:

        logger.info(
            "Tool execution completed successfully: %s",
            tool_name,
        )

    print(
        f"📊 Tool result: {result}"
    )

    return (
        tool_name,
        arguments,
        result,
    )


# --------------------------------------------------
# Safe Ollama call
# --------------------------------------------------

def call_llm(messages):
    """
    Safely call the local Ollama model.

    Returns:
        response object on success
        None on failure
    """

    try:

        response = chat(
            model="llama3.2:3b",
            messages=messages,
            tools=TOOLS,
            options={
                "temperature": 0,
            },
        )

        return response

    except Exception as e:

        logger.exception(
            "LLM request failed: %s",
            e,
        )

        print(
            "\n⚠️ KUDOO could not connect to the local AI model."
        )

        print(
            "Please make sure Ollama is running and try again."
        )

        return None


# --------------------------------------------------
# Main agent
# --------------------------------------------------

def main():

    print("====================================")
    print("          KUDOO 🤖📊")
    print("====================================")
    print("Ask questions about your sales data.")
    print("Type 'quit' to exit.\n")

    logger.info(
        "KUDOO agent started"
    )

    conversation = []

    while True:

        user_question = input("You: ").strip()

        # --------------------------------------------------
        # Exit
        # --------------------------------------------------

        if user_question.lower() == "quit":

            logger.info(
                "KUDOO agent stopped by user"
            )

            print("\nGoodbye! 👋")

            break

        if not user_question:

            continue

        logger.info(
            "User query received: %s",
            user_question,
        )

        # --------------------------------------------------
        # Start tracing this request
        # --------------------------------------------------

        tracer = AgentTracer()

        tracer.start(
            user_question
        )

        # --------------------------------------------------
        # Add user message
        # --------------------------------------------------

        conversation.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        # --------------------------------------------------
        # Build messages
        # --------------------------------------------------

        messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }

        ] + conversation

        # --------------------------------------------------
        # Agent tool loop
        # --------------------------------------------------

        request_failed = False

        while True:

            response = call_llm(
                messages
            )

            # --------------------------------------------------
            # LLM failure
            # --------------------------------------------------

            if response is None:

                request_failed = True

                final_answer = (
                    "I couldn't process that request because "
                    "the local AI model is unavailable. "
                    "Please make sure Ollama is running and try again."
                )

                break

            # --------------------------------------------------
            # Check whether model selected a tool
            # --------------------------------------------------

            if not response.message.tool_calls:

                final_answer = (
                    response.message.content
                    or "I could not determine an answer."
                )

                logger.info(
                    "Final answer generated successfully"
                )

                break

            # --------------------------------------------------
            # Execute all requested tools
            # --------------------------------------------------

            for tool_call in response.message.tool_calls:

                tool_start = time.perf_counter()

                (
                    tool_name,
                    tool_arguments,
                    result,
                ) = execute_tool(
                    tool_call
                )

                tool_duration = (
                    time.perf_counter()
                    - tool_start
                )

                logger.info(
                    "Tool duration: %s | %.4f seconds",
                    tool_name,
                    tool_duration,
                )

                # --------------------------------------------------
                # Record tool execution
                # --------------------------------------------------

                tracer.record_tool_call(

                    tool_name=tool_name,

                    arguments=tool_arguments,

                    result=result,

                    duration=tool_duration,

                )

                # --------------------------------------------------
                # Add assistant tool call to conversation
                # --------------------------------------------------

                messages.append(

                    {
                        "role": "assistant",

                        "tool_calls": [

                            {
                                "function": {
                                    "name": tool_name,

                                    "arguments": tool_arguments,
                                }
                            }

                        ],

                    }

                )

                # --------------------------------------------------
                # Add tool result
                # --------------------------------------------------

                messages.append(

                    {
                        "role": "tool",

                        "content": json.dumps(
                            result
                        ),

                    }

                )

        # --------------------------------------------------
        # Save assistant response to conversation
        # --------------------------------------------------

        conversation.append(

            {
                "role": "assistant",

                "content": final_answer,

            }

        )

        # --------------------------------------------------
        # Display final answer
        # --------------------------------------------------

        print("\n🤖 KUDOO:")
        print(final_answer)

        # --------------------------------------------------
        # Finish tracing
        # --------------------------------------------------

        trace = tracer.finish(
            final_answer
        )

        tracer.print_trace(
            trace
        )

        if request_failed:

            logger.warning(
                "Request completed with LLM failure"
            )

        else:

            logger.info(
                "Request completed successfully"
            )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":

    main()
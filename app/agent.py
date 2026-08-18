import json
import time
import uuid

from ollama import chat

from .tools import load_data
from .tool_registry import TOOLS, AVAILABLE_TOOLS


# --------------------------------------------------
# Load dataset once
# --------------------------------------------------

df = load_data()


# --------------------------------------------------
# Conversation memory
# --------------------------------------------------

CONVERSATIONS = {}


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are KUDOO, a helpful AI data analyst.

You answer questions about the provided sales dataset.

IMPORTANT RULES:

1. Always use a tool when the answer requires information
   from the dataset.

2. Never invent numbers.

3. Use the conversation history to understand follow-up
   questions.

4. Resolve pronouns such as:
   - it
   - that product
   - that region
   - the first one
   - the second one

   using previous conversation context.

5. If a question requires multiple pieces of information,
   use multiple tools when necessary.

6. For revenue of ONE specific product, use:
   get_product_revenue

7. For revenue of EVERY product, use:
   get_revenue_by_product

8. For finding the product with the HIGHEST revenue,
   use:
   get_best_product

9. For finding the region or AREA with the HIGHEST revenue,
   use:
   get_best_region

   IMPORTANT:
   Words such as:
   - best area
   - best region
   - highest-performing area
   - highest-performing region
   - top region
   - top-performing area
   - area that made the most money
   - region that generated the most revenue
   - which area performed best

   all mean that the user wants the BEST REGION.

   For these questions use get_best_region with NO arguments.

   Do NOT use get_revenue_by_region for these questions.

10. For total quantity, use:
    get_total_quantity

11. For total revenue, use:
    get_total_revenue

12. For comparing two products, use:
    get_revenue_difference

13. For a percentage of total revenue, use:
    get_percentage_of_total

14. For revenue of ONE SPECIFIC named region, use:
    get_revenue_by_region

    Example:
    "How much revenue did North bring in?"

    This should use:
    get_revenue_by_region
    {"region": "North"}

15. When a tool returns an error, explain the error clearly
    and do not invent an answer.

16. After all required tools have been executed, provide
    one concise final answer.

17. Do not call a tool again if the required information
    has already been obtained.

18. Do not describe internal tool calls to the user.

19. Do not output JSON tool calls as text.

20. Do not say that you are going to call a tool.
    Actually call the tool when needed.
"""

# --------------------------------------------------
# Execute tool
# --------------------------------------------------

def execute_tool(tool_call):
    """
    Execute one tool selected by the LLM.
    """

    tool_name = tool_call.function.name
    arguments = tool_call.function.arguments or {}

    function_to_call = AVAILABLE_TOOLS.get(tool_name)

    if function_to_call is None:
        return {
            "success": False,
            "error_type": "UNKNOWN_TOOL",
            "message": f"Tool '{tool_name}' is not registered.",
        }

    try:
        result = function_to_call(
            df,
            **arguments,
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "error_type": "TOOL_EXECUTION_ERROR",
            "message": str(e),
        }


# --------------------------------------------------
# Run agent
# --------------------------------------------------

def run_agent(user_question, conversation_id=None):

    start_time = time.perf_counter()

    # --------------------------------------------------
    # Create conversation if necessary
    # --------------------------------------------------

    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    if conversation_id not in CONVERSATIONS:
        CONVERSATIONS[conversation_id] = []

    conversation = CONVERSATIONS[conversation_id]

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
    # Build messages for LLM
    # --------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ] + conversation

    executed_tools = []

    # --------------------------------------------------
    # Agent tool loop
    # --------------------------------------------------

    while True:

        # --------------------------------------------------
        # Measure LLM response time
        # --------------------------------------------------

        llm_start = time.perf_counter()

        response = chat(
            model="llama3.2:3b",
            messages=messages,
            tools=TOOLS,
            options={
                "temperature": 0,
                "num_ctx": 2048,
                "num_predict": 128,
            },
        )

        llm_duration = (
            time.perf_counter()
            - llm_start
        )

        print(
            f"[LLM] {llm_duration:.2f}s | "
            f"tool_calls={len(response.message.tool_calls or [])}"
        )

        # --------------------------------------------------
        # No tool required
        # --------------------------------------------------

        if not response.message.tool_calls:

            final_answer = (
                response.message.content
                or "I could not determine an answer."
            )

            # Save final assistant response
            conversation.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                }
            )

            break

        # --------------------------------------------------
        # Save assistant tool-call message
        # --------------------------------------------------

        assistant_tool_calls = []

        for tool_call in response.message.tool_calls:

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments or {}

            assistant_tool_calls.append(
                {
                    "function": {
                        "name": tool_name,
                        "arguments": arguments,
                    }
                }
            )

        messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls,
            }
        )

        # --------------------------------------------------
        # Execute tools
        # --------------------------------------------------

        for tool_call in response.message.tool_calls:

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments or {}

            tool_start = time.perf_counter()

            result = execute_tool(tool_call)

            tool_duration = (
                time.perf_counter()
                - tool_start
            )

            executed_tools.append(
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "duration": tool_duration,
                }
            )

            print(
                f"[TOOL] {tool_name} "
                f"{tool_duration:.3f}s"
            )

            # --------------------------------------------------
            # Add tool result to current LLM context
            # --------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                }
            )

        # --------------------------------------------------
        # Continue loop
        # LLM will now process the tool result
        # --------------------------------------------------

    # --------------------------------------------------
    # Calculate total execution time
    # --------------------------------------------------

    total_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"[TOTAL] {total_time:.2f}s"
    )

    # --------------------------------------------------
    # Return result
    # --------------------------------------------------

    return {
        "answer": final_answer,
        "tool_calls": executed_tools,
        "execution_time": total_time,
        "conversation_id": conversation_id,
        "conversation": conversation,
    }
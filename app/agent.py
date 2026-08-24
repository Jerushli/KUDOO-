import json
import re
import time
import uuid

from ollama import Client

from .tools import load_data
from .tool_registry import TOOLS
from .memory import ConversationMemory
from .tracing import AgentTracer
from .tool_executor import (
    execute_tool,
    normalize_tool_arguments,
    resolve_dependent_arguments,
)
from .config import (
    MODEL_NAME,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT_SECONDS,
)


MAX_TOOL_CALLS = 5


# --------------------------------------------------
# Dataset
# --------------------------------------------------

df = load_data()


# --------------------------------------------------
# Conversation memory
# --------------------------------------------------

memory = ConversationMemory()
ollama_client = Client(
    host=OLLAMA_HOST,
    timeout=OLLAMA_TIMEOUT_SECONDS,
)


def call_llm(messages):
    """Call Ollama with a bounded client timeout."""

    try:
        return ollama_client.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            options={
                "temperature": 0,
                "num_ctx": 2048,
                "num_predict": 128,
            },
        )
    except Exception as error:
        print(f"[LLM ERROR] {error}")
        return None


def trace_llm_request(messages, iteration):
    tool_names = []
    for message in messages:
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            tool_names.append(function.get("name"))

    print(
        f"[TRACE] iteration={iteration} "
        f"history_tools={tool_names}"
    )


def trace_llm_response(response):
    tool_calls = response.message.tool_calls or []
    print(
        "[TRACE] response="
        f"{[(call.function.name, call.function.arguments or {}) for call in tool_calls]}"
    )


def trace_tool_result(tool_name, result):
    entity = None
    if isinstance(result, dict):
        entity = result.get("product") or result.get("region")

    print(f"[TRACE] execute {tool_name}")
    print(f"[TRACE] result {entity if entity else result}")


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

4. Resolve pronouns and conversational references such as:
   - it
   - that product
   - this product
   - that region
   - this region
   - that one
   - this one

   using the resolved conversation context provided to you.

5. If a question requires multiple pieces of information,
   use multiple tools when necessary.

     For dependent questions, always execute the prerequisite
     discovery tool first. Wait for its verified result before
     selecting the dependent tool. Never use a tool name, a
     placeholder, or an unverified value as a dependent argument.

     Examples:
     - Best product plus percentage: get_best_product, then
         get_percentage_of_total using the returned product.
     - Best region plus revenue: get_best_region, then
         get_revenue_by_region using the returned region.
     - Best product plus comparison: get_best_product, then
         get_revenue_difference using the returned product and the
         explicitly named comparison product.

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
# Follow-up question resolution
# --------------------------------------------------

def resolve_follow_up(
    question,
    conversation_id,
):
    """
    Resolve conversational references such as:

        it
        that product
        this product
        that region
        this region
        that one
        this one

    using the latest entity stored in conversation memory.
    """

    context = memory.get_context(
        conversation_id
    )

    if not context:
        return question

    entity = context.get("entity")
    entity_type = context.get("entity_type")

    if not entity:
        return question

    patterns = [
        r"\bit\b",
        r"\bthat product\b",
        r"\bthis product\b",
        r"\bthat region\b",
        r"\bthis region\b",
        r"\bthat one\b",
        r"\bthis one\b",
    ]

    contains_reference = any(
        re.search(
            pattern,
            question,
            re.IGNORECASE,
        )
        for pattern in patterns
    )

    if not contains_reference:
        return question

    resolved_question = question

    # --------------------------------------------------
    # Product context
    # --------------------------------------------------

    if entity_type == "product":

        product_patterns = [
            r"\bit\b",
            r"\bthat product\b",
            r"\bthis product\b",
            r"\bthat one\b",
            r"\bthis one\b",
        ]

        for pattern in product_patterns:

            resolved_question = re.sub(
                pattern,
                entity,
                resolved_question,
                flags=re.IGNORECASE,
            )

    # --------------------------------------------------
    # Region context
    # --------------------------------------------------

    elif entity_type == "region":

        region_patterns = [
            r"\bit\b",
            r"\bthat region\b",
            r"\bthis region\b",
            r"\bthat one\b",
            r"\bthis one\b",
        ]

        for pattern in region_patterns:

            resolved_question = re.sub(
                pattern,
                entity,
                resolved_question,
                flags=re.IGNORECASE,
            )

    # --------------------------------------------------
    # Debug output
    # --------------------------------------------------

    if resolved_question != question:

        print(
            "[MEMORY] Resolved follow-up: "
            f"'{question}' -> '{resolved_question}'"
        )

    return resolved_question


# --------------------------------------------------
# Store entity context
# --------------------------------------------------

def update_entity_context(
    conversation_id,
    tool_name,
    result,
):
    """
    Store the most recent important entity returned by a tool.

    Product tools store product context.

    Region tools store region context.

    This allows follow-up questions such as:

        Which product made the most revenue?
        How much revenue did it make?

    to be resolved as:

        How much revenue did Laptop make?
    """

    if not isinstance(result, dict):
        return

    # --------------------------------------------------
    # Product context
    # --------------------------------------------------

    if tool_name in {
        "get_product_revenue",
        "get_best_product",
    }:

        product = result.get("product")

        if product:

            memory.set_context(
                conversation_id=conversation_id,
                entity_type="product",
                entity=product,
            )

            print(
                f"[MEMORY] Stored product context: "
                f"{product}"
            )

    # --------------------------------------------------
    # Region context
    # --------------------------------------------------

    elif tool_name in {
        "get_revenue_by_region",
        "get_best_region",
    }:

        region = result.get("region")

        if region:

            memory.set_context(
                conversation_id=conversation_id,
                entity_type="region",
                entity=region,
            )

            print(
                f"[MEMORY] Stored region context: "
                f"{region}"
            )


def format_tool_result(tool_name, result, arguments=None):
    """Format verified tool output without another model-generated answer."""

    arguments = arguments or {}

    if isinstance(result, dict) and result.get("success") is False:
        if result.get("error_type") == "UNKNOWN_PRODUCT":
            product = arguments.get("product", "The requested product")
            return f"{product} was not found in the product data."

        if result.get("error_type") == "UNKNOWN_REGION":
            region = arguments.get("region", "The requested region")
            return f"{region} was not found in the region data."

        return result.get(
            "message",
            "The requested information was not found.",
        )

    if tool_name == "get_product_revenue":
        return (
            f"{result['product']} generated "
            f"${result['revenue']:,.0f} in revenue."
        )

    if tool_name == "get_best_product":
        return (
            f"{result['product']} was the top-performing product "
            f"with ${result['revenue']:,.0f} in revenue."
        )

    if tool_name == "get_best_region":
        return (
            f"{result['region']} was the best-performing region "
            f"with ${result['revenue']:,.0f} in revenue."
        )

    if tool_name == "get_total_quantity":
        quantity = (
            result.get("result")
            if isinstance(result, dict)
            else result
        )
        return f"We sold {quantity:,.0f} units altogether."

    if tool_name == "get_total_revenue":
        revenue = (
            result.get("result")
            if isinstance(result, dict)
            else result
        )
        return f"Total revenue was ${revenue:,.0f}."

    if tool_name == "get_revenue_difference":
        return (
            f"{result['first_product']} made "
            f"${result['difference']:,.0f} more than "
            f"{result['second_product']}."
        )

    if tool_name == "get_percentage_of_total":
        return (
            f"{result['filter_value']} contributed "
            f"{result['percentage']:.1f}% of total revenue."
        )

    if tool_name == "get_revenue_by_region":
        if len(result) == 1:
            region, revenue = next(iter(result.items()))
            return f"{region} generated ${revenue:,.0f} in revenue."

        lines = [
            f"{region}: ${revenue:,.0f}"
            for region, revenue in sorted(result.items())
        ]
        return "Revenue by region:\n" + "\n".join(lines)

    if tool_name == "get_revenue_by_product":
        lines = [
            f"{product}: ${revenue:,.0f}"
            for product, revenue in sorted(result.items())
        ]
        return "Revenue by product:\n" + "\n".join(lines)

    return str(result)


def explicit_entities(question, column):
    """Return dataset entities explicitly mentioned in the question."""

    values = []
    for value in df[column].dropna().unique():
        value = str(value)
        if re.search(rf"\b{re.escape(value)}\b", question, re.IGNORECASE):
            values.append(value)
    return values


def prerequisite_tool_call(tool_name, arguments, prior_results, question):
    """Return a discovery call when a dependent request lacks a verified entity."""

    has_product = any(
        isinstance(result, dict) and isinstance(result.get("product"), str)
        for result in prior_results
    )
    has_region = any(
        isinstance(result, dict) and isinstance(result.get("region"), str)
        for result in prior_results
    )
    explicit_products = explicit_entities(question, "product")
    explicit_regions = explicit_entities(question, "region")
    discovery_words = ("best", "top", "most", "highest")
    asks_for_discovery = any(
        word in question.lower()
        for word in discovery_words
    )

    if tool_name == "get_percentage_of_total":
        filter_column = arguments.get("filter_column")
        filter_value = arguments.get("filter_value")
        placeholder = str(filter_value or "").strip().lower()
        if (
            placeholder in {"get_best_product", "best_product"}
            and not has_product
        ):
            return "get_best_product"
        if (
            filter_column != "region"
            and not has_product
            and not explicit_products
            and asks_for_discovery
        ):
            return "get_best_product"

    if tool_name == "get_revenue_by_region":
        region = arguments.get("region")
        placeholder = str(region or "").strip().lower()
        if (
            placeholder in {"get_best_region", "best_region"}
            and not has_region
        ):
            return "get_best_region"
        if (
            not has_region
            and not explicit_regions
            and asks_for_discovery
        ):
            return "get_best_region"

    if tool_name == "get_revenue_difference":
        first_product = arguments.get("first_product") or arguments.get("product_a")
        second_product = arguments.get("second_product") or arguments.get("product_b")
        placeholder = str(first_product or "").strip().lower()
        if (
            placeholder in {"get_best_product", "best_product"}
            and not has_product
        ):
            return "get_best_product"
        if (
            not has_product
            and len(explicit_products) < 2
            and asks_for_discovery
        ):
            return "get_best_product"
        if (
            isinstance(second_product, str)
            and second_product.strip().lower() == "phone"
            and not has_product
        ):
            return "get_best_product"

    return None


# --------------------------------------------------
# Run KUDOO agent
# --------------------------------------------------

def run_agent(
    user_question,
    conversation_id=None,
):

    start_time = time.perf_counter()

    tracer = AgentTracer()

    tracer.start(user_question)

    # --------------------------------------------------
    # Create conversation ID
    # --------------------------------------------------

    if not conversation_id:

        conversation_id = str(
            uuid.uuid4()
        )

    # --------------------------------------------------
    # Resolve conversational references
    # --------------------------------------------------

    resolved_question = resolve_follow_up(
        question=user_question,
        conversation_id=conversation_id,
    )

    # --------------------------------------------------
    # Retrieve conversation history
    # --------------------------------------------------

    conversation = memory.get(
        conversation_id
    )

    # --------------------------------------------------
    # Add ORIGINAL user message to memory
    # --------------------------------------------------

    memory.add(
        conversation_id=conversation_id,
        role="user",
        content=user_question,
    )

    # --------------------------------------------------
    # Refresh conversation
    # --------------------------------------------------

    conversation = memory.get(
        conversation_id
    )

    # --------------------------------------------------
    # Build LLM messages
    # --------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Add previous conversation messages.
    #
    # The current user message is already stored above,
    # so we replace its content with the resolved version
    # for the current LLM call.

    if conversation:

        messages.extend(
            conversation[:-1]
        )

    messages.append(
        {
            "role": "user",
            "content": resolved_question,
        }
    )

    # --------------------------------------------------
    # Track executed tools
    # --------------------------------------------------

    executed_tools = []
    tool_call_count = 0
    iteration = 0
    pending_dependencies = []

    print(f"[TRACE] model={MODEL_NAME} temperature=0 context=2048")

    # --------------------------------------------------
    # Agent tool loop
    # --------------------------------------------------

    while tool_call_count < MAX_TOOL_CALLS:
        requests = []
        tool_failed = False

        if pending_dependencies:
            requests = pending_dependencies
            pending_dependencies = []
        else:
            iteration += 1
            trace_llm_request(messages, iteration)

            llm_start = time.perf_counter()
            response = call_llm(messages)

            if response is None:
                if executed_tools:
                    final_answer = "\n".join(
                        format_tool_result(
                            tool_name=tool_call["tool"],
                            result=tool_call["result"],
                            arguments=tool_call["arguments"],
                        )
                        for tool_call in executed_tools
                    )
                else:
                    final_answer = (
                        "I could not reach the local AI model. "
                        "Please make sure Ollama is running and try again."
                    )

                memory.add(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_answer,
                )
                break

            trace_llm_response(response)
            llm_duration = time.perf_counter() - llm_start
            tool_calls = response.message.tool_calls or []
            print(f"[LLM] {llm_duration:.2f}s | tool_calls={len(tool_calls)}")

            if not tool_calls:
                if executed_tools:
                    final_answer = "\n".join(
                        format_tool_result(
                            tool_name=tool_call["tool"],
                            result=tool_call["result"],
                            arguments=tool_call["arguments"],
                        )
                        for tool_call in executed_tools
                    )
                else:
                    final_answer = (
                        response.message.content
                        or "I could not determine an answer."
                    )

                memory.add(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_answer,
                )
                break

            duplicate_calls = []
            injected_prerequisite = False
            response_tool_names = {call.function.name for call in tool_calls}

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments or {}

                resolved_arguments = resolve_dependent_arguments(
                    tool_name=tool_name,
                    arguments=arguments,
                    prior_results=[item["result"] for item in executed_tools],
                )
                resolved_arguments = normalize_tool_arguments(
                    tool_name=tool_name,
                    df=df,
                    arguments=resolved_arguments,
                )

                if any(
                    item["tool"] == tool_name
                    and tool_name in {"get_best_product", "get_best_region"}
                    and not (
                        isinstance(item["result"], dict)
                        and item["result"].get("success") is False
                    )
                    for item in executed_tools
                ):
                    tool_call_count += 1
                    duplicate_calls.append(tool_name)
                    continue

                if any(
                    item["tool"] == tool_name
                    and tool_name in {"get_percentage_of_total", "get_revenue_difference", "get_revenue_by_region"}
                    and item["arguments"] == resolved_arguments
                    and not (
                        isinstance(item["result"], dict)
                        and item["result"].get("success") is False
                    )
                    for item in executed_tools
                ):
                    tool_call_count += 1
                    duplicate_calls.append(tool_name)
                    continue

                prerequisite = prerequisite_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                    prior_results=[item["result"] for item in executed_tools],
                    question=user_question,
                )

                if prerequisite and prerequisite in response_tool_names:
                    requests.append((tool_name, arguments))
                elif prerequisite:
                    injected_prerequisite = True
                    requests.append((prerequisite, {}))
                    pending_dependencies.append((tool_name, arguments))
                else:
                    requests.append((tool_name, arguments))

            if not requests:
                for duplicate_tool in duplicate_calls:
                    prior_call = next(
                        item for item in reversed(executed_tools)
                        if item["tool"] == duplicate_tool
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [{
                                "function": {
                                    "name": duplicate_tool,
                                    "arguments": prior_call["arguments"],
                                }
                            }],
                        }
                    )
                    messages.append(
                        {"role": "tool", "content": json.dumps(prior_call["result"])}
                    )

                if duplicate_calls and tool_call_count < MAX_TOOL_CALLS:
                    continue

                final_answer = "\n".join(
                    format_tool_result(
                        tool_name=tool_call["tool"],
                        result=tool_call["result"],
                        arguments=tool_call["arguments"],
                    )
                    for tool_call in executed_tools
                )
                memory.add(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_answer,
                )
                break

            if not injected_prerequisite and len(requests) == len(tool_calls):
                assistant_message = response.message
                if hasattr(assistant_message, "model_dump"):
                    assistant_message = assistant_message.model_dump(exclude_none=True)
                elif not isinstance(assistant_message, dict):
                    assistant_message = {
                        "role": "assistant",
                        "content": getattr(assistant_message, "content", ""),
                        "tool_calls": [{
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments or {},
                            }
                        } for call in tool_calls],
                    }
                assistant_message["role"] = "assistant"
            else:
                assistant_message = {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        }
                    } for name, arguments in requests],
                }

            messages.append(assistant_message)

        for tool_name, raw_arguments in requests:
            if tool_call_count >= MAX_TOOL_CALLS:
                break

            tool_call_count += 1
            arguments = resolve_dependent_arguments(
                tool_name=tool_name,
                arguments=raw_arguments,
                prior_results=[item["result"] for item in executed_tools],
            )
            arguments = normalize_tool_arguments(
                tool_name=tool_name,
                df=df,
                arguments=arguments,
            )

            tool_start = time.perf_counter()
            result = execute_tool(tool_name=tool_name, df=df, arguments=arguments)
            tool_duration = time.perf_counter() - tool_start

            executed_tools.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "duration": tool_duration,
            })

            if isinstance(result, dict) and result.get("success") is False:
                tool_failed = True

            update_entity_context(
                conversation_id=conversation_id,
                tool_name=tool_name,
                result=result,
            )

            tracer.record_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                duration=tool_duration,
            )

            print(f"[TOOL] {tool_name} {tool_duration:.3f}s")
            trace_tool_result(tool_name, result)
            messages.append({"role": "tool", "content": json.dumps(result)})

        if tool_failed or tool_call_count >= MAX_TOOL_CALLS:
            final_answer = "\n".join(
                format_tool_result(
                    tool_name=tool_call["tool"],
                    result=tool_call["result"],
                    arguments=tool_call["arguments"],
                )
                for tool_call in executed_tools
            )
            memory.add(
                conversation_id=conversation_id,
                role="assistant",
                content=final_answer,
            )
            break

    else:
        final_answer = "\n".join(
            format_tool_result(
                tool_name=tool_call["tool"],
                result=tool_call["result"],
                arguments=tool_call["arguments"],
            )
            for tool_call in executed_tools
        )
        memory.add(
            conversation_id=conversation_id,
            role="assistant",
            content=final_answer,
        )

    # --------------------------------------------------
    # Total execution time
    # --------------------------------------------------

    total_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------
    # Finish trace
    # --------------------------------------------------

    trace = tracer.finish(
        final_answer
    )

    print(
        f"[TOTAL] {total_time:.2f}s"
    )

    # --------------------------------------------------
    # Return structured result
    # --------------------------------------------------

    return {
        "answer": final_answer,
        "tool_calls": executed_tools,
        "execution_time": total_time,
        "conversation_id": conversation_id,
        "conversation": memory.get(
            conversation_id
        ),
        "trace": trace,
    }
from .tool_registry import AVAILABLE_TOOLS


def execute_tool(tool_name, df, arguments):
    """
    Safely execute a registered KUDOO tool.

    Parameters:
        tool_name: Name of the tool selected by the LLM.
        df: Sales dataframe.
        arguments: Arguments extracted from the LLM tool call.

    Returns:
        A normalized dictionary containing either the tool result
        or a structured error.
    """

    # --------------------------------------------------
    # Validate tool name
    # --------------------------------------------------

    function_to_call = AVAILABLE_TOOLS.get(tool_name)

    if function_to_call is None:
        return {
            "success": False,
            "error_type": "UNKNOWN_TOOL",
            "message": f"Tool '{tool_name}' is not registered.",
        }

    # --------------------------------------------------
    # Validate arguments
    # --------------------------------------------------

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return {
            "success": False,
            "error_type": "INVALID_ARGUMENTS",
            "message": "Tool arguments must be a dictionary.",
        }

    # --------------------------------------------------
    # Execute tool safely
    # --------------------------------------------------

    try:
        result = function_to_call(
            df,
            **arguments
        )

        return normalize_tool_result(result)

    except TypeError as e:
        return {
            "success": False,
            "error_type": "INVALID_TOOL_ARGUMENTS",
            "message": str(e),
        }

    except Exception as e:
        return {
            "success": False,
            "error_type": "TOOL_EXECUTION_ERROR",
            "message": str(e),
        }


def normalize_tool_result(result):
    """
    Normalize tool output into a predictable structure.
    """

    # --------------------------------------------------
    # Dictionary result
    # --------------------------------------------------

    if isinstance(result, dict):
        return result

    # --------------------------------------------------
    # Primitive / other result
    # --------------------------------------------------

    return {
        "success": True,
        "result": result,
    }
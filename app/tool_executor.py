from .tool_registry import AVAILABLE_TOOLS


def normalize_tool_arguments(tool_name, df, arguments):
    """Canonicalize dataset entities without changing arbitrary arguments."""

    if arguments is None:
        return {}

    if not isinstance(arguments, dict):
        return arguments

    normalized = dict(arguments)

    if tool_name in {
        "get_product_revenue",
        "get_revenue_difference",
        "get_percentage_of_total",
    }:
        product_columns = {
            "product",
            "first_product",
            "second_product",
            "filter_value",
        }
        products = {
            str(value).strip().lower(): str(value).strip()
            for value in df["product"].dropna().unique()
        }

        for key in product_columns:
            value = normalized.get(key)
            if (
                isinstance(value, str)
                and key in normalized
                and normalized.get("filter_column", "product") == "product"
                and value.strip().lower() in products
            ):
                normalized[key] = products[value.strip().lower()]

    if tool_name in {"get_revenue_by_region", "get_percentage_of_total"}:
        regions = {
            str(value).strip().lower(): str(value).strip()
            for value in df["region"].dropna().unique()
        }
        value = normalized.get("region")
        if (
            isinstance(value, str)
            and value.strip().lower() in regions
        ):
            normalized["region"] = regions[value.strip().lower()]

        if (
            normalized.get("filter_column") == "region"
            and isinstance(normalized.get("filter_value"), str)
            and normalized["filter_value"].strip().lower() in regions
        ):
            normalized["filter_value"] = regions[
                normalized["filter_value"].strip().lower()
            ]

    return normalized


def resolve_dependent_arguments(tool_name, arguments, prior_results):
    """Fill dependent arguments only from verified earlier tool results."""

    resolved = dict(arguments or {})

    if tool_name == "get_revenue_difference":
        if "product_a" in resolved and "first_product" not in resolved:
            resolved["first_product"] = resolved.pop("product_a")
        if "product_b" in resolved and "second_product" not in resolved:
            resolved["second_product"] = resolved.pop("product_b")

    product = None
    region = None
    for result in reversed(prior_results):
        if not isinstance(result, dict):
            continue
        if product is None and isinstance(result.get("product"), str):
            product = result["product"]
        if region is None and isinstance(result.get("region"), str):
            region = result["region"]

    if product and tool_name == "get_percentage_of_total":
        if resolved.get("filter_column") not in {"region"}:
            resolved["column"] = "revenue"
            resolved["filter_column"] = "product"
            filter_value = resolved.get("filter_value")
            if isinstance(filter_value, str):
                value = filter_value.strip().lower()
                if value in {"get_best_product", "best_product"}:
                    resolved["filter_value"] = product
            elif "filter_value" not in resolved:
                resolved["filter_value"] = product

    if product and tool_name == "get_revenue_difference":
        first_product = resolved.get("first_product")
        if isinstance(first_product, str):
            value = first_product.strip().lower()
            if value in {"get_best_product", "best_product"}:
                resolved["first_product"] = product
        elif "first_product" not in resolved and "product_a" not in resolved:
            resolved["first_product"] = product

    if region and tool_name == "get_revenue_by_region":
        region_value = resolved.get("region")
        if isinstance(region_value, str):
            value = region_value.strip().lower()
            if value in {"get_best_region", "best_region"}:
                resolved["region"] = region
        elif "region" not in resolved:
            resolved["region"] = region

    return resolved


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
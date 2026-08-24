from .tools import (
    get_product_revenue,
    get_best_product,
    get_best_region,
    get_total_quantity,
    get_total_revenue,
    get_revenue_difference,
    get_percentage_of_total,
    get_revenue_by_region,
    get_revenue_by_product,
)


# ============================================================
# Tool implementations
# ============================================================

AVAILABLE_TOOLS = {
    "get_product_revenue": get_product_revenue,
    "get_best_product": get_best_product,
    "get_best_region": get_best_region,
    "get_total_quantity": get_total_quantity,
    "get_total_revenue": get_total_revenue,
    "get_revenue_difference": get_revenue_difference,
    "get_percentage_of_total": get_percentage_of_total,
    "get_revenue_by_region": get_revenue_by_region,
    "get_revenue_by_product": get_revenue_by_product,
}


# ============================================================
# Tool schemas for Ollama
# ============================================================

TOOLS = [

    # --------------------------------------------------------
    # ONE SPECIFIC PRODUCT
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_product_revenue",

            "description": (
                "Get revenue for ONE SPECIFIC PRODUCT named by the user. "
                "ALWAYS use this tool when the user asks how much revenue, "
                "money, or cash a SINGLE named product generated. "
                "This tool MUST be used even if the product may not exist. "
                "For example: 'How much revenue did Laptop generate?', "
                "'How much revenue did Tesla generate?', or "
                "'How much money did Apple make?'. "
                "NEVER use get_revenue_by_product for a single named product."
            ),

            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": (
                            "The EXACT product name mentioned by the user. "
                            "Preserve the product name from the question."
                        ),
                    }
                },
                "required": ["product"],
            },
        },
    },


    # --------------------------------------------------------
    # BEST PRODUCT
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_best_product",

            "description": (
                "Find which product has the highest total revenue. "
                "Use this when the user asks for the best, top, highest, "
                "or most profitable product by revenue."
            ),

            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },


    # --------------------------------------------------------
    # BEST REGION
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_best_region",

            "description": (
                "Find which region has the highest total revenue. "
                "Use this when the user asks which region, area, or "
                "location performed best by revenue."
            ),

            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },


    # --------------------------------------------------------
    # TOTAL QUANTITY
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_total_quantity",

            "description": (
                "Get the total number of units sold across the entire dataset."
            ),

            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },


    # --------------------------------------------------------
    # TOTAL REVENUE
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_total_revenue",

            "description": (
                "Get the total revenue across all sales in the dataset."
            ),

            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },


    # --------------------------------------------------------
    # COMPARE TWO PRODUCTS
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_revenue_difference",

            "description": (
                "Compare revenue between TWO SPECIFIC PRODUCTS. "
                "Use this when the user asks how much more or less "
                "one named product made compared with another named product."
            ),

            "parameters": {
                "type": "object",
                "properties": {
                    "first_product": {
                        "type": "string",
                        "description": "First product name.",
                    },
                    "second_product": {
                        "type": "string",
                        "description": "Second product name.",
                    },
                },
                "required": [
                    "first_product",
                    "second_product",
                ],
            },
        },
    },


    # --------------------------------------------------------
    # PERCENTAGE OF TOTAL
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_percentage_of_total",

            "description": (
                "Calculate what percentage of total revenue came from "
                "ONE SPECIFIC PRODUCT or ONE SPECIFIC REGION. "
                "For a best-product or best-region compound question, "
                "wait for that tool's result and use its exact returned "
                "entity as filter_value. Never use a tool name as a value."
            ),

            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "enum": ["revenue"],
                        "description": "Always use revenue.",
                    },

                    "filter_column": {
                        "type": "string",
                        "enum": ["product", "region"],
                        "description": (
                            "Use product when the question refers to a product; "
                            "use region when it refers to a region."
                        ),
                    },

                    "filter_value": {
                        "type": "string",
                        "description": (
                            "The exact product or region mentioned by the user."
                        ),
                    },
                },

                "required": [
                    "column",
                    "filter_column",
                    "filter_value",
                ],
            },
        },
    },


    # --------------------------------------------------------
    # REGION REVENUE
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_revenue_by_region",

            "description": (
                "Get revenue for ONE SPECIFIC REGION or revenue for ALL regions. "
                "Use this only when the user asks about a region's revenue "
                "or asks for revenue across regions."
            ),

            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": (
                            "Specific region name. "
                            "Omit this field when asking for all regions."
                        ),
                    },
                },
                "required": [],
            },
        },
    },


    # --------------------------------------------------------
    # ALL PRODUCTS
    # --------------------------------------------------------

    {
        "type": "function",
        "function": {
            "name": "get_revenue_by_product",

            "description": (
                "Get revenue for EVERY PRODUCT in the dataset. "
                "Use ONLY when the user explicitly asks for all products, "
                "each product, every product, or a product-by-product "
                "revenue breakdown. "
                "NEVER use this tool when the user asks about ONE specific "
                "named product, including a product that may not exist."
            ),

            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
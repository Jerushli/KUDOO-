import pandas as pd


def load_data():
    """Load the sales dataset and calculate revenue."""
    df = pd.read_csv("data/sales.csv")
    df["revenue"] = df["quantity"] * df["price"]
    return df


def get_total_revenue(df):
    """Calculate total revenue across all sales."""
    return float(df["revenue"].sum())


def get_best_product(df):
    """Find the product with the highest total revenue."""
    revenue_by_product = df.groupby("product")["revenue"].sum()

    best_product = revenue_by_product.idxmax()
    best_revenue = revenue_by_product.max()

    return {
        "product": best_product,
        "revenue": float(best_revenue),
    }


def get_best_region(df):
    """Find the region with the highest total revenue."""
    revenue_by_region = df.groupby("region")["revenue"].sum()

    best_region = revenue_by_region.idxmax()
    best_revenue = revenue_by_region.max()

    return {
        "region": best_region,
        "revenue": float(best_revenue),
    }


def get_revenue_by_region(df, region=None):
    """
    Return revenue by region.

    If a region is provided, return revenue for that region.
    If no region is provided, return revenue for every region.
    """

    grouped = (
        df.groupby("region")["revenue"]
        .sum()
        .to_dict()
    )

    # Return all regions
    if region is None:
        return {
            key: float(value)
            for key, value in grouped.items()
        }

    # Validate requested region
    if region not in grouped:
        return {
            "success": False,
            "error_type": "UNKNOWN_REGION",
            "message": f"'{region}' was not found in the region data.",
        }

    # Return only requested region
    return {
        region: float(grouped[region])
    }
    
def get_total_quantity(df):
    """Calculate the total number of units sold."""
    return float(df["quantity"].sum())

def get_revenue_by_product(df):
    """Calculate total revenue for every product."""
    result = df.groupby("product")["revenue"].sum()

    return {
        product: float(revenue)
        for product, revenue in result.items()
    }

def get_revenue_by_date(df):
    """Calculate total revenue for every date."""
    result = df.groupby("date")["revenue"].sum()

    return {
        str(date): float(revenue)
        for date, revenue in result.items()
    }
def run_analysis(df, operation, column=None, filter_column=None, filter_value=None):
    """
    Perform a controlled analytical operation on the dataset.
    """

    # ----------------------------------------------
    # Filter data if requested
    # ----------------------------------------------

    working_df = df.copy()

    if filter_column and filter_value:

    # ----------------------------------------------
    # Validate requested filter value
    # ----------------------------------------------

        validation = validate_filter_value(
            df,
            filter_column,
            filter_value,
        )

        if not validation["valid"]:
            return {
                "error": validation["error"]
            }

        working_df = df[
            df[filter_column].astype(str).str.lower()
            == str(filter_value).lower()
        ]
def get_product_revenue(df, product):
    """
    Get revenue for one specific product.
    """

    # Check whether the product exists
    available_products = (
        df["product"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    requested_product = str(product).strip().lower()

    if requested_product not in available_products:
        return tool_error(
            "UNKNOWN_PRODUCT",
            f"'{product}' was not found in the product data."
        )

    # Calculate revenue
    revenue = df[
        df["product"].astype(str).str.lower()
        == requested_product
    ]["revenue"].sum()

    return {
        "success": True,
        "product": product,
        "revenue": float(revenue),
    }

    # ----------------------------------------------
    # Total
    # ----------------------------------------------

    if operation == "sum":

        if column not in df.columns:
            return {"error": f"Unknown column: {column}"}

        return {
            "operation": "sum",
            "column": column,
            "result": float(working_df[column].sum()),
        }


    # ----------------------------------------------
    # Average
    # ----------------------------------------------

    elif operation == "average":

        if column not in df.columns:
            return {"error": f"Unknown column: {column}"}

        return {
            "operation": "average",
            "column": column,
            "result": float(working_df[column].mean()),
        }


    # ----------------------------------------------
    # Count
    # ----------------------------------------------

    elif operation == "count":

        return {
            "operation": "count",
            "result": int(len(working_df)),
        }


    # ----------------------------------------------
    # Percentage of total
    # ----------------------------------------------

    elif operation == "percentage_of_total":

        if column not in df.columns:
            return {"error": f"Unknown column: {column}"}

        filtered_total = working_df[column].sum()
        overall_total = df[column].sum()

        if overall_total == 0:
            return {
                "error": "Cannot calculate percentage because total is zero."
            }

        percentage = (filtered_total / overall_total) * 100

        return {
            "operation": "percentage_of_total",
            "filter": {
                "column": filter_column,
                "value": filter_value,
            },
            "result": round(float(percentage), 2),
        }
    
    # ----------------------------------------------
    # Difference between two values
    # ----------------------------------------------

    elif operation == "difference":

        if column not in df.columns:
            return {"error": f"Unknown column: {column}"}

        if not filter_column:
            return {
                "error": "filter_column is required for difference."
            }

        if not filter_value:
            return {
                "error": "filter_value is required for difference."
            }

        parts = str(filter_value).split("|")

        if len(parts) != 2:
            return {
                "error": (
                    "For difference, provide two values separated "
                    "by '|', for example 'Laptop|Phone'."
                )
            }

        first_value = parts[0].strip()
        second_value = parts[1].strip()

        first_total = df[
            df[filter_column].astype(str).str.lower()
            == first_value.lower()
        ][column].sum()

        second_total = df[
            df[filter_column].astype(str).str.lower()
            == second_value.lower()
        ][column].sum()

        difference = first_total - second_total

        return {
            "operation": "difference",
            "first": {
                "value": first_value,
                "total": float(first_total),
            },
            "second": {
                "value": second_value,
                "total": float(second_total),
            },
            "difference": float(difference),
        }


    # ----------------------------------------------
    # Unsupported operation
    # ----------------------------------------------

    else:

        return {
            "error": f"Unsupported operation: {operation}"
        }
def validate_filter_value(df, filter_column, filter_value):
    """
    Check whether the requested filter value exists
    in the specified dataset column.
    """

    if filter_column not in df.columns:
        return {
            "valid": False,
            "error": f"Unknown column: {filter_column}",
        }

    available_values = (
        df[filter_column]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    requested_value = str(filter_value).strip().lower()

    if requested_value not in available_values:
        return {
            "valid": False,
            "error": (
                f"'{filter_value}' was not found in "
                f"the '{filter_column}' column."
            ),
        }

    return {
        "valid": True,
        "value": filter_value,
    }
def tool_error(error_type, message):
    """
    Create a consistent error response for all tools.
    """

    return {
        "success": False,
        "error_type": error_type,
        "message": message,
    }
def get_revenue_difference(df, first_product, second_product):
    """
    Calculate the revenue difference between two products.
    """

    products = (
        df["product"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    first = str(first_product).strip().lower()
    second = str(second_product).strip().lower()

    if first not in products.values:

        return tool_error(
            "UNKNOWN_PRODUCT",
            f"'{first_product}' was not found in the product data."
        )

    if second not in products.values:

        return tool_error(
            "UNKNOWN_PRODUCT",
            f"'{second_product}' was not found in the product data."
        )

    first_revenue = float(
        df.loc[
            products == first,
            "revenue"
        ].sum()
    )

    second_revenue = float(
        df.loc[
            products == second,
            "revenue"
        ].sum()
    )

    difference = abs(first_revenue - second_revenue)

    return {
        "success": True,
        "first_product": first_product,
        "first_revenue": first_revenue,
        "second_product": second_product,
        "second_revenue": second_revenue,
        "difference": difference,
    }
def get_percentage_of_total(df, column, filter_column, filter_value):
    """
    Calculate the percentage of total revenue represented
    by a particular product or region.
    """

    if column not in df.columns:

        return tool_error(
            "UNKNOWN_COLUMN",
            f"Column '{column}' does not exist in the dataset."
        )

    if filter_column not in df.columns:

        return tool_error(
            "UNKNOWN_COLUMN",
            f"Column '{filter_column}' does not exist in the dataset."
        )

    values = (
        df[filter_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    requested_value = (
        str(filter_value)
        .strip()
        .lower()
    )

    if requested_value not in values.values:

        return tool_error(
            "UNKNOWN_FILTER_VALUE",
            f"'{filter_value}' was not found in the "
            f"'{filter_column}' column."
        )

    total = float(df[column].sum())

    filtered_total = float(
        df.loc[
            values == requested_value,
            column
        ].sum()
    )

    if total == 0:

        return tool_error(
            "ZERO_TOTAL",
            "The total value is zero, so a percentage "
            "cannot be calculated."
        )

    percentage = (filtered_total / total) * 100

    return {
        "success": True,
        "filter_column": filter_column,
        "filter_value": filter_value,
        "filtered_total": filtered_total,
        "total": total,
        "percentage": round(percentage, 2),
    }
from app.tools import (
    load_data,
    get_total_revenue,
    get_total_quantity,
    get_best_product,
    get_best_region,
    get_product_revenue,
    get_revenue_difference,
    get_percentage_of_total,
)


df = load_data()


def test_total_revenue():

    result = get_total_revenue(df)

    assert result == 42100.0


def test_total_quantity():

    result = get_total_quantity(df)

    assert result == 105.0


def test_best_product():

    result = get_best_product(df)

    assert result["product"] == "Laptop"

    assert result["revenue"] == 16800.0


def test_best_region():

    result = get_best_region(df)

    assert result["region"] == "North"

    assert result["revenue"] == 16800.0


def test_laptop_revenue():

    result = get_product_revenue(
        df,
        "Laptop"
    )

    assert result["success"] is True

    assert result["revenue"] == 16800.0


def test_unknown_product():

    result = get_product_revenue(
        df,
        "Tesla"
    )

    assert result["success"] is False

    assert result["error_type"] == "UNKNOWN_PRODUCT"


def test_revenue_difference():

    result = get_revenue_difference(
        df,
        "Laptop",
        "Phone"
    )

    assert result["success"] is True

    assert result["difference"] == 4200.0


def test_percentage_of_total():

    result = get_percentage_of_total(
        df,
        "revenue",
        "product",
        "Laptop"
    )

    assert result["success"] is True

    assert result["percentage"] == 39.9
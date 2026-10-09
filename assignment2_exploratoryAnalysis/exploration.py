
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. SET UP FILE PATHS
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_FOLDER / "data" / "Sales-Export_2019-2020.csv"

OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("MGS 3101: EXPLORATORY AND DESCRIPTIVE ANALYTICS")
print("=" * 60)


# --------------------------------------------------
# 2. LOAD THE DATA
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find the dataset at: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

# Remove extra spaces from column names.
df.columns = df.columns.str.strip()

# Remove accidental spaces around text values.
for column in df.select_dtypes(include=["object", "str"]).columns:
    df[column] = df[column].str.strip()

# Convert sales and cost columns to numeric values.

# Clean the sales and cost columns.
# Sales values contain commas, such as "17,524.02".
df["order_value_EUR"] = pd.to_numeric(
    df["order_value_EUR"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

df["cost"] = pd.to_numeric(
    df["cost"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)
# Convert dates without deleting valid rows unnecessarily.
df["date"] = pd.to_datetime(
    df["date"].astype(str).str.strip(),
    errors="coerce"
)

print("\n1. DATASET OVERVIEW")
print(f"Dataset file: {DATA_FILE.name}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nColumn information:")
df.info()

if df.empty:
    raise ValueError(
        "The dataset contains zero rows. Check the CSV file."
    )


# --------------------------------------------------
# 3. CHECK FOR MISSING VALUES
# --------------------------------------------------

print("\n2. MISSING VALUES")
print(df.isna().sum())

print("\nRows with missing sales values:", df["order_value_EUR"].isna().sum())
print("Rows with missing cost values:", df["cost"].isna().sum())
print("Rows with invalid dates:", df["date"].isna().sum())


# --------------------------------------------------
# 4. SUMMARY STATISTICS
# --------------------------------------------------

print("\n3. SUMMARY STATISTICS")
print(df[["order_value_EUR", "cost"]].describe().round(2))


# --------------------------------------------------
# 5. TOTAL SALES AND COSTS
# --------------------------------------------------

total_sales = df["order_value_EUR"].sum()
total_costs = df["cost"].sum()
sales_minus_costs = total_sales - total_costs

print("\n4. TOTAL SALES AND COSTS")
print(f"Total sales: €{total_sales:,.2f}")
print(f"Total costs: €{total_costs:,.2f}")
print(f"Sales minus recorded costs: €{sales_minus_costs:,.2f}")
print(f"Average order value: €{df['order_value_EUR'].mean():,.2f}")
print(f"Median order value: €{df['order_value_EUR'].median():,.2f}")
print(f"Largest order: €{df['order_value_EUR'].max():,.2f}")


# --------------------------------------------------
# 6. SALES BY COUNTRY
# --------------------------------------------------

sales_by_country = (
    df.groupby("country")["order_value_EUR"]
    .sum()
    .sort_values(ascending=False)
)

print("\n5. SALES BY COUNTRY")
print(sales_by_country.round(2).to_string())


# --------------------------------------------------
# 7. SALES BY CATEGORY
# --------------------------------------------------

sales_by_category = (
    df.groupby("category")["order_value_EUR"]
    .sum()
    .sort_values(ascending=False)
)

print("\n6. SALES BY CATEGORY")
print(sales_by_category.round(2).to_string())


# --------------------------------------------------
# 8. SALES BY DEVICE
# --------------------------------------------------

sales_by_device = (
    df.groupby("device_type")["order_value_EUR"]
    .sum()
    .sort_values(ascending=False)
)

print("\n7. SALES BY DEVICE")
print(sales_by_device.round(2).to_string())


# --------------------------------------------------
# 9. MONTHLY SALES
# --------------------------------------------------

valid_dates = df.dropna(subset=["date"]).copy()

valid_dates["month"] = (
    valid_dates["date"].dt.to_period("M").astype(str)
)

monthly_sales = (
    valid_dates.groupby("month")["order_value_EUR"]
    .sum()
    .sort_index()
)

print("\n8. MONTHLY SALES")
print(monthly_sales.round(2).to_string())

if not monthly_sales.empty:
    best_month = monthly_sales.idxmax()
    print(
        f"\nHighest-sales month: {best_month} "
        f"(€{monthly_sales.max():,.2f})"
    )


# --------------------------------------------------
# 10. TOP 10 LARGEST ORDERS
# --------------------------------------------------

top_orders = (
    df.nlargest(10, "order_value_EUR")[
        ["order_id", "country", "category", "date", "order_value_EUR"]
    ]
)

print("\n9. TOP 10 LARGEST ORDERS")
print(top_orders.to_string(index=False))


# --------------------------------------------------
# 11. CREATE VISUALIZATIONS
# --------------------------------------------------

def create_bar_chart(series, title, xlabel, filename, color=None):
    if series.empty:
        print(f"Skipping chart because no data is available: {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    series.plot(
        kind="bar",
        ax=ax,
        color=color
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sales (EUR)")
    ax.tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / filename, dpi=300)
    plt.close(fig)

    print(f"Saved chart: {OUTPUT_FOLDER / filename}")


create_bar_chart(
    sales_by_country,
    "Total Sales by Country",
    "Country",
    "sales_by_country.png"
)

create_bar_chart(
    sales_by_category,
    "Total Sales by Product Category",
    "Category",
    "sales_by_category.png"
)

create_bar_chart(
    sales_by_device,
    "Total Sales by Device Type",
    "Device Type",
    "sales_by_device.png"
)

create_bar_chart(
    monthly_sales,
    "Monthly Sales Over Time",
    "Month",
    "monthly_sales.png"
)


# --------------------------------------------------
# 12. EXPORT ANALYSIS RESULTS
# --------------------------------------------------

sales_by_country.to_csv(OUTPUT_FOLDER / "sales_by_country.csv")
sales_by_category.to_csv(OUTPUT_FOLDER / "sales_by_category.csv")
sales_by_device.to_csv(OUTPUT_FOLDER / "sales_by_device.csv")
monthly_sales.to_csv(OUTPUT_FOLDER / "monthly_sales.csv")
top_orders.to_csv(OUTPUT_FOLDER / "top_10_orders.csv", index=False)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print(f"Results and charts saved in: {OUTPUT_FOLDER}")
print("=" * 60)
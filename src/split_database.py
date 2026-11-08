from pathlib import Path
import pandas as pd
import sqlite3

# Setup paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "src" else current_dir
data_dir = project_root / "data"

# Auto-locate dataset
matched_files = list(data_dir.glob("personal_transactions*.*"))
if not matched_files:
    raise FileNotFoundError(f"No transaction file found in {data_dir}")

target_file = matched_files[0]
print(f"Reading: {target_file.name}")

# Read CSV or Excel
if target_file.suffix.lower() == ".csv":
    df = pd.read_csv(target_file)
else:
    df = pd.read_excel(target_file)

# Clean headers and formats
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.strftime("%Y-%m")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

# Filter out internal transfers
df_clean = df[df["Category"] != "Credit Card Payment"].copy()

# Define category boundaries
income_categories = ["Paycheck"]

static_categories = [
    "Mortgage & Rent",
    "Utilities",
    "Mobile Phone",
    "Internet",
    "Auto Insurance",
    "Music",
    "Movies & Dvds",
    "Television"
]

dynamic_categories = [
    "Groceries",
    "Restaurants",
    "Shopping",
    "Gas & Fuel",
    "Home Improvement",
    "Coffee Shops",
    "Alcohol & Bars",
    "Fast Food",
    "Haircut",
    "Electronics & Software",
    "Food & Dining",
    "Entertainment"
]

# Standardize column selection for clean tables
columns_to_keep = ["Date", "Month", "Description", "Category", "Amount"]

# Build the three tables
df_paycheck = df_clean[df_clean["Category"].isin(income_categories)][columns_to_keep].copy()
df_static = df_clean[df_clean["Category"].isin(static_categories)][columns_to_keep].copy()
df_dynamic = df_clean[df_clean["Category"].isin(dynamic_categories)][columns_to_keep].copy()

# Sort each chronologically
df_paycheck = df_paycheck.sort_values(by="Date").reset_index(drop=True)
df_static = df_static.sort_values(by="Date").reset_index(drop=True)
df_dynamic = df_dynamic.sort_values(by="Date").reset_index(drop=True)

# Export to CSV in the data folder
paycheck_path = data_dir / "table_paycheck.csv"
static_path = data_dir / "table_static.csv"
dynamic_path = data_dir / "table_dynamic.csv"

df_paycheck.to_csv(paycheck_path, index=False)
df_static.to_csv(static_path, index=False)
df_dynamic.to_csv(dynamic_path, index=False)

# store in an SQLite database for SQL queries
db_path = data_dir / "finance.db"
with sqlite3.connect(db_path) as conn:
    df_paycheck.to_sql("paycheck", conn, if_exists="replace", index=False)
    df_static.to_sql("static_expenses", conn, if_exists="replace", index=False)
    df_dynamic.to_sql("dynamic_expenses", conn, if_exists="replace", index=False)

# ==========================================
# Verification Summary
# ==========================================
print("\n" + "=" * 55)
print("SPLIT COMPLETE - SUMMARY")
print("=" * 55)
print(f"1. Paycheck Table: {len(df_paycheck):,} rows | Total: ${df_paycheck['Amount'].sum():,.2f}")
print(f"2. Static Table:   {len(df_static):,} rows | Total: ${df_static['Amount'].sum():,.2f}")
print(f"3. Dynamic Table:  {len(df_dynamic):,} rows | Total: ${df_dynamic['Amount'].sum():,.2f}")
print(f"\nFiles generated in '{data_dir}':")
print(f" - table_paycheck.csv")
print(f" - table_static.csv")
print(f" - table_dynamic.csv")
print(f" - finance.db (SQLite database)")
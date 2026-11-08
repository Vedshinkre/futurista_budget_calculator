from pathlib import Path
import pandas as pd

# Dynamically resolve the project root and find the file inside 'data/'
# (Works whether you run the script from 'hack/' or from inside 'src/')
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "src" else current_dir
data_dir = project_root / "data"

# Auto-detect the dataset file matching 'personal_transactions*' (csv or excel)
matched_files = list(data_dir.glob("personal_transactions*.*"))
if not matched_files:
    # Fallback to any spreadsheet in the data folder
    matched_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xlsx"))

if not matched_files:
    raise FileNotFoundError(f"No transaction file found in {data_dir}")

target_file = matched_files[0]
print(f"Loading file: {target_file.name}")

# Read the file according to its format
if target_file.suffix.lower() == ".csv":
    df = pd.read_csv(target_file)
else:
    df = pd.read_excel(target_file)

# Clean up column names (strip whitespace and handle truncated names)
df.columns = df.columns.str.strip()
for col in df.columns:
    if "nsaction" in col.lower() or "type" in col.lower():
        df.rename(columns={col: "Transaction Type"}, inplace=True)
        break

# Format Date and numerical Amount
df["Date"] = pd.to_datetime(df["Date"])
df["Year_Month"] = df["Date"].dt.to_period("M")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")


# Category Analysis

categories = df["Category"].dropna().unique()
print("\n" + "=" * 55)
print(f"TOTAL UNIQUE CATEGORIES: {len(categories)}")
print("=" * 55)
cat_summary = (
    df.groupby("Category")
    .agg(
        transaction_count=("Amount", "count"),
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
    )
    .sort_values(by="transaction_count", ascending=False)
)
cat_summary["total_amount"] = cat_summary["total_amount"].map("${:,.2f}".format)
cat_summary["avg_amount"] = cat_summary["avg_amount"].map("${:,.2f}".format)
print(cat_summary)


# Month Coverage & Timeline

min_date = df["Date"].min()
max_date = df["Date"].max()
unique_months = sorted(df["Year_Month"].unique())

print("\n" + "=" * 55)
print(f"TIMELINE: {len(unique_months)} MONTHS COVERED")
print("=" * 55)
print(f"Start Date: {min_date.date()}")
print(f"End Date:   {max_date.date()}")
print("\nTransactions per month:")
print(df["Year_Month"].value_counts().sort_index())


# High-Level Cash Flow (Debit vs. Credit)

print("\n" + "=" * 55)
print("CASH FLOW & NATURE PREVIEW")
print("=" * 55)
if "Transaction Type" in df.columns:
    flow = df.groupby("Transaction Type")["Amount"].agg(["count", "sum"])
    flow["sum"] = flow["sum"].map("${:,.2f}".format)
    print(flow)
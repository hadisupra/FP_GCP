#!/usr/bin/env python3
"""
preprocess_sql.py

Preprocess Olist CSVs into a single SQLite database + export cleaned CSVs.
- CSV output folder: ./FP/
- SQLite output: olist.db
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path

# CONFIG
from pathlib import Path
# Default: gunakan environment variable OHS_BASE_PATH; kalau tidak ada, pakai current working dir.
BASE_PATH = os.getenv("OHS_BASE_PATH", str(Path.cwd()))
OUT_DB = os.getenv("OHS_OUT_DB", "olist.db")


CSV_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "cat_translation": "product_category_name_translation.csv",
}

# ============================================================
# LOAD
# ============================================================

def load_csv(name):
    path = Path(BASE_PATH) / CSV_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f" -> {len(df):,} rows, {len(df.columns)} cols")
    return df

# ============================================================
# CLEANING
# ============================================================

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def prepare_dataframes(dfs: dict) -> dict:
    out = {}
    for k, df in dfs.items():
        df = sanitize_column_names(df)

        # parse datetime
        if k == "orders":
            date_cols = [c for c in df.columns if "date" in c or "timestamp" in c or "approved" in c]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

        # numeric conversions
        if k in ("order_payments", "order_items"):
            for col in df.columns:
                if any(token in col for token in ("price", "value", "freight", "installments")):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # ensure ID columns are strings
        id_cols = [c for c in df.columns if c.endswith("_id") or c.endswith("_ID")]
        for col in id_cols:
            df[col] = df[col].astype(str)

        out[k] = df

    return out

# ============================================================
# EXPORT CLEAN CSVs
# ============================================================

def export_dataframes_to_csv(dfs: dict, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting cleaned CSVs to folder: {out_path.resolve()}\n")
    for name, df in dfs.items():
        file_path = out_path / f"{name}.csv"
        print(f" -> Writing {file_path} ({len(df):,} rows)")
        df.to_csv(file_path, index=False)

# ============================================================
# WRITE SQLITE
# ============================================================

def write_sqlite(dfs: dict, db_path: str):
    if os.path.exists(db_path):
        print(f"Removing existing DB at {db_path}")
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.commit()

    # Write tables
    for name, df in dfs.items():
        print(f"Writing table `{name}` ({len(df):,} rows)...")
        df.to_sql(name, conn, index=False, if_exists="replace")

    cur = conn.cursor()

    # Indices
    index_statements = [
        "CREATE INDEX idx_orders_order_id ON orders(order_id);",
        "CREATE INDEX idx_items_order_id ON order_items(order_id);",
        "CREATE INDEX idx_items_product_id ON order_items(product_id);",
        "CREATE INDEX idx_items_seller_id ON order_items(seller_id);",
        "CREATE INDEX idx_payments_order_id ON order_payments(order_id);",
        "CREATE INDEX idx_customers_customer_id ON customers(customer_id);",
        "CREATE INDEX idx_sellers_seller_id ON sellers(seller_id);",
        "CREATE INDEX idx_products_product_id ON products(product_id);",
        "CREATE INDEX idx_customers_zip ON customers(customer_zip_code_prefix);",
        "CREATE INDEX idx_sellers_zip ON sellers(seller_zip_code_prefix);",
        "CREATE INDEX idx_geo_zip ON geolocation(geolocation_zip_code_prefix);",
    ]

    print("Creating indices...")
    for stmt in index_statements:
        try:
            cur.execute(stmt)
        except Exception as e:
            print("  index error:", e)

    # Views
    print("Creating views...")

    try:
        cur.execute("""
        CREATE VIEW order_item_full AS
        SELECT
            oi.order_id,
            oi.product_id,
            p.product_category_name,
            p.product_name_lenght,
            oi.seller_id,
            s.seller_city,
            s.seller_state,
            oi.price,
            oi.freight_value,
            o.customer_id,
            c.customer_city,
            c.customer_state,
            o.order_status,
            o.order_purchase_timestamp,
            r.review_score,
            r.review_comment_message
        FROM order_items oi
        LEFT JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN sellers s ON oi.seller_id = s.seller_id
        LEFT JOIN orders o ON oi.order_id = o.order_id
        LEFT JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN order_reviews r ON o.order_id = r.order_id;
        """)

        cur.execute("""
        CREATE VIEW payments_aggregated AS
        SELECT
            order_id,
            SUM(payment_value) AS total_paid,
            MAX(payment_installments) AS max_installments,
            COUNT(*) AS num_payments
        FROM order_payments
        GROUP BY order_id;
        """)
    except Exception as e:
        print("View creation error:", e)

    conn.commit()
    conn.close()
    print(f"\nSQLite DB written to: {db_path}")

# ============================================================
# RUN
# ============================================================

def run():
    # Load CSVs
    dfs = {key: load_csv(key) for key in CSV_FILES.keys()}

    # Clean + type normalization
    print("\nPreparing dataframes (sanitizing, type conversions)...")
    dfs = prepare_dataframes(dfs)

    # Export cleaned CSVs to folder ./FP/
    export_dataframes_to_csv(dfs, "./FP")

    # Write SQLite
    out_db_path = Path.cwd() / OUT_DB
    write_sqlite(dfs, str(out_db_path))

    print("\nDone. Clean CSVs located in folder: ./FP")
    print(f"SQLite DB: {out_db_path}")

if __name__ == "__main__":
    run()

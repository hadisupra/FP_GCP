# merge_reviews_per_product.py
import pandas as pd
import os
from textwrap import shorten

# -------------------------
# CONFIG
# -------------------------
INPUT_PRODUCTS = "olist_products_dataset.csv"
INPUT_REVIEWS = "olist_order_reviews_dataset.csv"
INPUT_ORDERS = "olist_orders_dataset.csv"
INPUT_ITEMS = "olist_order_items_dataset.csv"
INPUT_SELLERS = "olist_sellers_dataset.csv"
INPUT_CAT_TRANS = "product_category_name_translation.csv"

OUTPUT_FILE = "merged_per_product_docbase.csv"

# limits
MAX_REVIEWS_TO_CONCAT = 50          # ambil maksimal X review per product untuk digabung
MAX_COMBINED_LENGTH = 4000          # potong hasil gabungan dokument agar tidak terlalu panjang

# -------------------------
# LOAD CSV
# -------------------------
print("Loading CSV files...")
products = pd.read_csv(INPUT_PRODUCTS)
reviews = pd.read_csv(INPUT_REVIEWS)
orders = pd.read_csv(INPUT_ORDERS)
items = pd.read_csv(INPUT_ITEMS)
sellers = pd.read_csv(INPUT_SELLERS)
cat_translation = pd.read_csv(INPUT_CAT_TRANS)

# -------------------------
# MAP REVIEWS -> PRODUCT
# (reviews keyed by order_id; items map order_id -> product_id)
# -------------------------
print("Mapping reviews to product_id via order_items...")
# join items (order_id, product_id) with reviews (order_id)
items_reviews = items[['order_id', 'product_id', 'seller_id']].merge(
    reviews, on='order_id', how='inner'
)

# jika tidak ada review terkait product, items_reviews mungkin kosong
print(f"Total item-review rows: {len(items_reviews)}")

# -------------------------
# AGGREGATE REVIEWS PER PRODUCT
# -------------------------
print("Aggregating reviews per product_id...")

def concat_limited(texts, max_items=MAX_REVIEWS_TO_CONCAT, sep=" || "):
    # ambil first N non-null texts
    vals = [str(t).strip() for t in texts if pd.notna(t) and str(t).strip() != ""]
    if not vals:
        return ""
    vals = vals[:max_items]
    joined = sep.join(vals)
    # truncate length for safety
    return shorten(joined, width=MAX_COMBINED_LENGTH, placeholder=" ...")

agg_funcs = {
    'review_score': ['count', 'mean'],
    'review_comment_title': lambda s: concat_limited(s.dropna().astype(str).tolist()),
    'review_comment_message': lambda s: concat_limited(s.dropna().astype(str).tolist()),
}

grouped = items_reviews.groupby('product_id').agg(
    num_reviews = ('review_score', 'count'),
    avg_review_score = ('review_score', 'mean'),
    combined_review_titles = ('review_comment_title', lambda s: concat_limited(s.dropna().astype(str).tolist())),
    combined_review_messages = ('review_comment_message', lambda s: concat_limited(s.dropna().astype(str).tolist())),
)

grouped = grouped.reset_index()

print(f"Products with >=1 review: {len(grouped)}")

# -------------------------
# ADD seller locations per product (unique cities)
# -------------------------
print("Collecting seller city/state per product...")
# merge items -> sellers to get seller_city per item
items_sellers = items[['product_id', 'seller_id']].merge(sellers[['seller_id','seller_city','seller_state']], on='seller_id', how='left')
# aggregate unique cities per product
seller_loc = items_sellers.groupby('product_id').agg(
    seller_cities = ('seller_city', lambda s: ", ".join(sorted({str(x) for x in s.dropna()}) ) ),
    seller_states = ('seller_state', lambda s: ", ".join(sorted({str(x) for x in s.dropna()}) ) ),
    seller_count = ('seller_id', lambda s: len(set(s.dropna())) )
).reset_index()

# -------------------------
# MERGE product metadata (category translation etc.)
# -------------------------
print("Merging product metadata...")
products = products.merge(cat_translation, how='left', on='product_category_name')

# Keep only needed product columns to join
prod_meta = products[['product_id','product_category_name','product_category_name_english','product_name_lenght']] \
            .drop_duplicates(subset=['product_id'])

# Merge all
print("Merging aggregated reviews, seller info, and product metadata...")
merged = prod_meta.merge(grouped, on='product_id', how='left')
merged = merged.merge(seller_loc, on='product_id', how='left')

# Fill NaN for products without review
merged['num_reviews'] = merged['num_reviews'].fillna(0).astype(int)
merged['avg_review_score'] = merged['avg_review_score'].fillna(0.0)

# -------------------------
# BUILD DOCUMENT PER PRODUCT
# -------------------------
print("Building document text per product...")
def build_product_doc(row):
    parts = [
        f"Product ID: {row.get('product_id','')}",
        f"Product Name Length: {row.get('product_name_lenght', '')}",
        f"Product Category: {row.get('product_category_name','')}",
        f"Category (English): {row.get('product_category_name_english','')}",
        f"Sellers (count): {row.get('seller_count',0)}",
        f"Seller Cities: {row.get('seller_cities','')}",
        f"Seller States: {row.get('seller_states','')}",
        f"Number of Reviews: {row.get('num_reviews',0)}",
        f"Average Review Score: {round(row.get('avg_review_score',0.0),2)}",
        f"Combined Review Titles: {row.get('combined_review_titles','')}",
        f"Combined Review Messages: {row.get('combined_review_messages','')}",
    ]
    # filter empty and join
    return "\n".join([p for p in parts if p and str(p).strip() != ""])

merged['document'] = merged.apply(build_product_doc, axis=1)

# optional: drop empty doc (shouldn't happen)
merged = merged[merged['document'].str.strip() != ""]

# -------------------------
# SAVE RESULT
# -------------------------
print(f"Saving merged per-product CSV to {OUTPUT_FILE} ...")
merged.to_csv(OUTPUT_FILE, index=False)

# -------------------------
# STATS
# -------------------------
total_products = merged['product_id'].nunique()
total_docs = len(merged)
total_item_rows = len(items)
total_item_review_rows = len(items_reviews)

print("DONE.")
print(f"Total item rows (order_items): {total_item_rows}")
print(f"Total item-review rows (items joined with reviews): {total_item_review_rows}")
print(f"Total products in products.csv: {products['product_id'].nunique()}")
print(f"Total products with >=1 review (docs created): {total_products}")
print(f"Output rows (documents): {total_docs}")
print(f"Output saved: {os.path.abspath(OUTPUT_FILE)}")

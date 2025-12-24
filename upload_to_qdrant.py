# upload_to_qdrant.py
import os
import sys
import time
import toml
import pandas as pd
from tqdm import tqdm
import math

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

# -----------------------
# Config
# -----------------------
INPUT_CSV = "merged_per_product_docbase.csv"
COLLECTION_NAME = "olist_products"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# You can override these via env vars for troubleshooting large uploads
BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "128"))
UPSERT_CHUNK_SIZE = int(os.getenv("QDRANT_UPSERT_CHUNK_SIZE", "32"))
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "60"))
MAX_PAYLOAD_TEXT = int(os.getenv("QDRANT_MAX_PAYLOAD_TEXT", "400"))  # trim stored text to avoid huge payloads

# -----------------------
# Load secrets (if secret.toml exists)
# -----------------------
SECRETS_FILE = "secrets.toml"
if os.path.exists(SECRETS_FILE):
    try:
        secrets = toml.load(SECRETS_FILE)
        for k, v in secrets.items():
            # do not overwrite existing env vars
            if v and not os.getenv(k):
                os.environ[k] = v
    except Exception as e:
        print("Warning: failed to load secret.toml:", repr(e))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not OPENAI_API_KEY:
    print("ERROR: QDRANT_URL, QDRANT_API_KEY, and OPENAI_API_KEY must be set (env or secret.toml).")
    sys.exit(1)

# -----------------------
# Load CSV
# -----------------------
if not os.path.exists(INPUT_CSV):
    print(f"ERROR: input CSV not found: {INPUT_CSV}")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)
if "document" not in df.columns:
    print("ERROR: input CSV must contain a 'document' column.")
    sys.exit(1)

texts = df["document"].fillna("").astype(str).tolist()
n_docs = len(texts)
print(f"Loaded {n_docs} documents from {INPUT_CSV}")

# -----------------------
# Connect to Qdrant
# -----------------------
print("Connecting to Qdrant...")
# prefer_grpc=True generally improves throughput and stability; increase timeout for large batches
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
    timeout=QDRANT_TIMEOUT,
)

try:
    info = client.get_collections()
    print("Connected to Qdrant. Collections:", [c.name for c in info.collections])
except Exception as e:
    print("ERROR: unable to list collections:", repr(e))
    sys.exit(1)

# create collection if not exists
if not client.collection_exists(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' not found. Creating...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
    )
    print("Collection created.")
else:
    print(f"Collection '{COLLECTION_NAME}' exists. Proceeding to upload.")

# -----------------------
# Prepare embeddings
# -----------------------
emb = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

def embed_texts_batch(text_list):
    """
    Try to use a batch embed function if provided; otherwise fall back to per-item embed_query.
    Returns list of vectors in same order.
    """
    # prefer embed_documents if available (batch)
    if hasattr(emb, "embed_documents"):
        return emb.embed_documents(text_list)
    else:
        vectors = []
        for t in text_list:
            vectors.append(emb.embed_query(t))
        return vectors

# -----------------------
# Upload in batches
# -----------------------
print(f"Uploading embeddings to Qdrant in batches of {BATCH_SIZE} ...")

# create deterministic integer ids (0..n-1)
start_idx = 0
for start in range(0, n_docs, BATCH_SIZE):
    end = min(n_docs, start + BATCH_SIZE)
    batch_texts = texts[start:end]
    # embed
    try:
        vectors = embed_texts_batch(batch_texts)
    except Exception as e:
        print(f"ERROR during embedding batch {start}-{end-1}:", repr(e))
        print("Retrying once after 5s...")
        time.sleep(5)
        try:
            vectors = embed_texts_batch(batch_texts)
        except Exception as e2:
            print("Second attempt failed:", repr(e2))
            sys.exit(1)

    # prepare payloads and ids
    ids = list(range(start, end))
    # helpers to sanitize payload values for JSON
    def sane_str(v):
        if pd.isna(v):
            return ""
        return str(v)

    def sane_int(v):
        if pd.isna(v):
            return None
        try:
            vv = float(v)
            if math.isfinite(vv):
                return int(vv)
            return None
        except Exception:
            return None

    def sane_float(v):
        if pd.isna(v):
            return None
        try:
            vv = float(v)
            return vv if math.isfinite(vv) else None
        except Exception:
            return None

    payloads = []
    for i, idx in enumerate(range(start, end)):
        row = df.iloc[idx]
        payload = {
            "product_id": sane_str(row.get("product_id", "")),
            "product_category": sane_str(row.get("product_category_name", "")),
            "product_category_en": sane_str(row.get("product_category_name_english", "")),
            "num_reviews": sane_int(row.get("num_reviews", None)) if "num_reviews" in row else None,
            "avg_review_score": sane_float(row.get("avg_review_score", None)) if "avg_review_score" in row else None,
            # store trimmed document to inspect later; keep full doc offline if needed
            "text": sane_str(row.get("document", ""))[:MAX_PAYLOAD_TEXT],
        }
        payloads.append(payload)

    # upsert in smaller chunks with retries to avoid timeouts
    def chunks(lst, size):
        for i in range(0, len(lst), size):
            yield i, lst[i:i+size]

    for offset, _ in chunks(ids, UPSERT_CHUNK_SIZE):
        sub_ids = ids[offset: offset + UPSERT_CHUNK_SIZE]
        sub_vectors = vectors[offset: offset + UPSERT_CHUNK_SIZE]
        sub_payloads = payloads[offset: offset + UPSERT_CHUNK_SIZE]

        attempts = 0
        max_attempts = 3
        backoff = 3
        while True:
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=models.Batch(ids=sub_ids, vectors=sub_vectors, payloads=sub_payloads)
                )
                break
            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    print(f"ERROR during upsert batch {start}-{end-1} (sub {offset}-{offset+len(sub_ids)-1}):", repr(e))
                    sys.exit(1)
                else:
                    print(f"Upsert timeout/error on sub-batch {offset}-{offset+len(sub_ids)-1}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2

    print(f"Uploaded batch {start}-{end-1} ({end-start} items)")

print("Upload complete.")
print(f"Total uploaded: {n_docs} vectors to collection '{COLLECTION_NAME}'")

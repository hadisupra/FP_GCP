import os
from dotenv import load_dotenv
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import Tool, initialize_agent
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import qdrant_client as qdrant_client_pkg
import langchain as langchain_pkg



# ================= CONFIG =================

SQLITE_PATH = "olist.db"
COLLECTION_NAME = "olist_products"


# ================= LLMs =================

# Tool selector LLM
router_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Response formatter LLM (Agent Utama)
formatter_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# ================= ENV & QDRANT =================

# Load .env if present (local/dev convenience)
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Startup diagnostics
print({
    "qdrant_client_version": getattr(qdrant_client_pkg, "__version__", "unknown"),
    "langchain_version": getattr(langchain_pkg, "__version__", "unknown"),
    "qdrant_url_set": bool(QDRANT_URL),
    "qdrant_api_key_set": bool(QDRANT_API_KEY),
    "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
})

# Initialize client more defensively for Cloud Run
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    timeout=60.0,
)

# Try a lightweight connectivity check (guarded)
try:
    _info = client.get_collections()
    print("Qdrant connectivity OK. Collections:", [c.name for c in _info.collections])
except Exception as e:
    print("Warning: Qdrant connectivity check failed:", repr(e))

# Diagnostic: verify QdrantClient has 'search' method (required by langchain_community)
if not hasattr(client, 'search'):
    print("ERROR: QdrantClient does not have 'search' method. Verify qdrant-client>=1.7.4 is installed.")
else:
    print("OK: QdrantClient.search method available.")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ================= RAG AGENT =================
CATEGORY_SYNONYMS = {
    # Indonesia → dataset
    "jam tangan": "relogio",
    "alat tulis": "papelaria",
    "parfum": "perfumaria",
    "telepon": "telefonia",

    # English → dataset
    "watch": "relogio",
    "stationery": "papelaria",
    "perfume": "perfumaria",
    "phone": "telefonia",
}

def build_retriever(category: str | None = None):
    search_kwargs = {"k": 4}

    if category:
        search_kwargs["filter"] = {
            "should": [
                {"key": "product_category", "match": {"value": category}},
                {"key": "product_category_en", "match": {"value": category}},
            ]
        }

    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def rag_search(query: str) -> list[str]:
    query_lower = query.lower()

    category_map = {
        "parfum": "perfumaria",
        "perfume": "perfumaria",
        "telepon": "telefonia",
        "phone": "telefonia",
        "jam tangan": "relogio",
        "watch": "relogio",
    }

    category = None
    for k, v in category_map.items():
        if k in query_lower:
            category = v
            break

    retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
    docs = retriever.get_relevant_documents(query)

    # ✅ HARD FILTER di Python (INI KUNCINYA)
    if category:
        filtered_docs = [
            d.page_content
            for d in docs
            if f"Product Category: {category}" in d.page_content
        ]
    else:
        filtered_docs = [d.page_content for d in docs]

    return filtered_docs



# ================= SQL AGENT =================

def sql_query(query: str) -> list:
    import re
    import sqlite3

    # 1️⃣ Bersihkan markdown ```sql ```
    cleaned_query = query.strip()
    cleaned_query = re.sub(r"```sql|```", "", cleaned_query, flags=re.IGNORECASE).strip()

    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()

    try:
        cur.execute(cleaned_query)
        return cur.fetchall()
    except Exception as e:
        return [f"SQL Error: {str(e)}"]
    finally:
        conn.close()


# ================= TOOLS =================

tools = [
    Tool(
        name="RAG_Search",
        func=rag_search,
        description="""
        Gunakan HANYA untuk:
        - ringkasan
        - review pelanggan
        - opini
        - insight kualitatif
        - pertanyaan deskriptif

        Contoh:
        - "ringkasan review pelanggan kategori telefonia"
        - "apa pendapat pelanggan tentang parfum"

        JANGAN gunakan untuk:
        - hitungan
        - rata-rata
        - total
        """
    ),
    Tool(
        name="SQL_Query",
        func=sql_query,
        description="""
        Gunakan SQLite database Olist (Brazilian e-commerce).

        ====================
        DATABASE SCHEMA (GROUND TRUTH)
        ====================

        Tables & columns yang VALID:

        1) orders
        - order_id
        - customer_id
        - order_status
        - order_purchase_timestamp
        - order_delivered_customer_date

        2) order_reviews
        - review_id
        - order_id
        - review_score
        - review_comment_title
        - review_comment_message

        3) order_items
        - order_id
        - product_id
        - seller_id
        - price
        - freight_value

        4) products
        - product_id
        - product_category_name

        5) customers
        - customer_id
        - customer_city
        - customer_state

        ====================
        BAHASA & SYNONYM MAPPING (WAJIB)
        ====================

        Jika input menggunakan Bahasa Indonesia atau Inggris,
        lakukan mapping berikut SEBELUM membuat SQL.

        ENTITAS:
        - pesanan, order, transaksi → orders
        - ulasan, review, penilaian, rating → order_reviews
        - pelanggan, pembeli, customer → customers
        - produk, barang, product → products

        KOLOM:
        - skor, skor review, rating, nilai → review_score
        - harga → price
        - ongkir → freight_value

        AGREGASI:
        - jumlah, total, berapa banyak → COUNT
        - rata-rata, average, mean → AVG
        - total harga, jumlah harga → SUM(price)

        ====================
        ATURAN KERAS
        ====================

        - Gunakan HANYA tabel & kolom di atas
        - JANGAN mengarang nama tabel atau kolom
        - Gunakan JOIN eksplisit jika diperlukan
        - Database menggunakan SQLite

        Jika permintaan TIDAK bisa dipenuhi karena data tidak tersedia:
        - JANGAN mengarang SQL
        - Jelaskan keterbatasan data dengan bahasa natural

        ====================
        CONTOH
        ====================

        Q: Berapa jumlah pesanan?
        SQL:
        SELECT COUNT(*) FROM orders;

        Q: Berapa rata-rata skor review semua pesanan?
        SQL:
        SELECT AVG(review_score) FROM order_reviews;

        Q: Berapa rata-rata skor review per kategori produk?
        SQL:
        SELECT p.product_category_name, AVG(r.review_score)
        FROM order_reviews r
        JOIN order_items i ON r.order_id = i.order_id
        JOIN products p ON i.product_id = p.product_id
        GROUP BY p.product_category_name;

        ====================
        END
        ====================
        """
    )

]


# ================= ROUTER AGENT =================

router_agent = initialize_agent(
    tools=tools,
    llm=router_llm,
    agent="zero-shot-react-description",
    verbose=True
)

# ================= FASTAPI =================

app = FastAPI()


class QueryInput(BaseModel):
    query: str


@app.get("/")
def health():
    # Return brief env + client info to aid debugging
    ok = True
    details = {
        "qdrant_client_version": getattr(qdrant_client_pkg, "__version__", "unknown"),
        "langchain_version": getattr(langchain_pkg, "__version__", "unknown"),
        "qdrant_url_set": bool(QDRANT_URL),
        "qdrant_api_key_set": bool(QDRANT_API_KEY),
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "collection_name": COLLECTION_NAME,
    }
    return {"status": "running", "details": details}


@app.post("/ask")
def ask(q: QueryInput):
    raw_result = router_agent.run(q.query)

    if isinstance(raw_result, list):
        raw_text = "\n".join(raw_result)
    else:
        raw_text = str(raw_result)

    final_prompt = f"""
Anda adalah analis produk.

Tugas Anda:
- Merangkum data review pelanggan menjadi insight yang bermakna
- Fokus pada POLA, bukan item individual
- Jelaskan secara ringkas, terstruktur, dan non-teknis

ATURAN WAJIB:
- JANGAN menyebut product_id atau data teknis mentah
- JANGAN mengarang jika data tidak mendukung
- JANGAN menampilkan SQL, query, atau format dataset
- Jika data kosong atau tidak relevan, jelaskan keterbatasannya secara eksplisit

STRUKTUR JAWABAN:
1. Ringkasan umum sentimen pelanggan
2. Pola kepuasan utama (jika ada)
3. Pola keluhan utama (jika ada)
4. Implikasi atau insight utama

Gunakan bahasa yang sama dengan pertanyaan pengguna.

Pertanyaan pengguna:
{q.query}

Data mentah:
{raw_text}
"""

    final_answer = formatter_llm.predict(final_prompt)

    return {"answer": final_answer}

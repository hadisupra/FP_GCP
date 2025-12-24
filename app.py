#build: ./Dockerfile
#python: 3.10-slim
#docker composer on python 3.13.5 base

"""
FastAPI Application for LLM Agent Service
This module contains the FastAPI web service that exposes the agent functionality
"""

from xmlrpc import client
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
from agent import SimpleAgent
import os
import json
import uvicorn
from dotenv import load_dotenv
import logging
import pandas as pd
import numpy as np
from qdrant_client.http import models
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
import requests
import re
import asyncio
# import streamlit as st  # Not used in FastAPI app
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
from uuid import uuid4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
# from langchain_community.vectorstores import qdrant as QdrantVectorStore  # Deprecated
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from qdrant_client.http import models
# from langchain.vectorstores import Qdrant  # Deprecated
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_community.utilities import SQLDatabase
# from langchain_openai import ChatOpenAI
# from langchain.chains import SQLDatabaseChain
# from langchain.graphs.memgraph_graph import RAW_SCHEMA_QUERY
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain_classic.chains import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_qdrant import QdrantVectorStore
# from langchain_community.retrievers import QdrantPointsRetriever
import sqlite3


# from langchain.chains import RetrievalQA
# from langchain_experimental.sql import SQLDatabaseChain, SQLDatabase, ChatOpenAI

# Load environment variables from .env file
# sqlite_db_path = "c:\\Users\\user\\Documents\\GitHub\\Final_Project\\simple-llm-docker\\data\\llm_agent.db"
from dotenv import load_dotenv
load_dotenv()

# Fallback: load secrets.toml if env vars are missing (useful on Cloud Run)
try:
    import toml  # lightweight parser
    secrets_path = os.path.join(os.path.dirname(__file__), "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            secrets = toml.load(secrets_path)
            for k, v in secrets.items():
                if v and not os.getenv(k):
                    os.environ[k] = v
        except Exception:
            pass
except Exception:
    pass
# os.environ.get("SQLITE_DB_PATH", "llm_agent.db")
# sqlite_db_path = "C:\\Users\\user\\Documents\\GitHub\\Final_Project\\simple-llm-docker\\data\\llm_agent.db"
# os.environ.get("SQLITE_DB_PATH", "llm_agent.db")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Feature flags / controls
DISABLE_INGEST = os.getenv("DISABLE_INGEST", "0").strip().lower() in ("1", "true", "yes", "on")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# CSV loading will happen in startup_event to ensure proper initialization in Docker
# sqlite_db_path will be set at runtime

# Helper function to verify database (can be called after startup)
def show_data_from_sqlite():
    sqlite_db_path = os.getenv("SQLITE_DB_PATH", "olist.db")
    conn = sqlite3.connect(sqlite_db_path)
    query = "SELECT * FROM products LIMIT 5;"
    df = pd.read_sql(query, conn)
    print(f"Data from SQLite database '{sqlite_db_path}': 'products' table:")
    print(df)
    conn.close()
#load data to sqlite database
# Function to load sample data into SQLite database

#         user="root",
#         password="ijj4swt",
#         database="llm_agent_db"
#     )
#     # Query data into DataFrame
#     query = "SELECT * FROM chat_history"
#     df = pd.read_sql(query, conn)
#     print("Data from MySQL database:")
#     print(df)
#     conn.close()

# load data to mysql database
# Function to load sample data into MySQL database
#call app_dbsql.py to load data
# import app_dbsql
# def import_csv_to_mysql(csv_file_path, mysql_config, table_name):
#     app_dbsql.import_csv_to_mysql(csv_file_path, mysql_config, table_name)

# def load_data_to_mysql():
#     # Connect to MySQL database
#     # conn = _mysql_connector.connect(
#     #     host="localhost",   
#     #     user="root",
#     #     password="password",
#     #     database="llm_agent_db"
#     # )
#     # cursor = conn.cursor()
#     # Create table if not exists
#     # cursor.execute("""
#     #     CREATE TABLE IF NOT EXISTS chat_history (
#     #         id INT AUTO_INCREMENT PRIMARY KEY,
#     #         user_message TEXT,
#     #         agent_response TEXT,
#     #         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#     #     )
#     # """)
#     # # Load sample data into DataFrame
#     # data = {
#     #     "user_message": ["Hello", "How are you?", "What is AI?"],
#     #     "agent_response": ["Hi there!", "I'm a bot, I don't have feelings.", "AI stands for Artificial Intelligence."]
#     # }
#     # df = pd.DataFrame(data)
#     # # Insert data into MySQL table
#     # for _, row in df.iterrows():
#     #     cursor.execute(
#     #         "INSERT INTO chat_history (user_message, agent_response) VALUES (%s, %s)",
#     #         (row['user_message'], row['agent_response'])
#     #     )
#     # conn.commit()
#     # cursor.close()
#     # conn.close()
#     # logger.info("Sample data loaded into MySQL database.")


# ================= PYDANTIC MODELS =================
class ChatRequest(BaseModel):
    message: str
    agent: Optional[Literal["auto", "sql", "qdrant"]] = "auto"
    session_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "message": "ringkasan review parfum",
                "agent": "auto"
            }
        }


# Initialize FastAPI app
app = FastAPI(
    title="LLM Agent Service with Dual RAG Agents",
    description="LLM Agent service with SQL and Qdrant RAG agents for comprehensive data analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
agent = None
# placeholders for runtime-initialized components
db = None
llm = None
sql_chain = None
qdrant_client = None
vectorstore = None
vectorstore_products = None
retriever = None
agent_a = None
rag_chain = None
embeddings = None
reviews_df = None
# Qdrant defaults (can be overridden via env)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_URL", "http://host.docker.internal:6338")
QDRANT_URL = os.getenv("QDRANT_URL", "https://acb9e0ed-c7e4-4abc-9495-1382817b533e.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "olist_reviews")
headers = {"Content-Type": "application/json"}
if QDRANT_API_KEY:
    headers["Authorization"] = f"Bearer {QDRANT_API_KEY}"



@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup - optimized for fast Cloud Run startup"""
    global agent, db, llm, toolkit, sql_chain, qdrant_client, vectorstore, retriever, agent_a, rag_chain, embeddings, QDRANT_API_KEY, QDRANT_URL, headers, reviews_df, sql_rag_agent, qdrant_rag_agent

    logger.info("🚀 Starting FastAPI application...")
    
    # Skip heavy initialization on Cloud Run - do lazy loading instead
    if os.getenv("DISABLE_INGEST") == "1":
        logger.info("⚡ DISABLE_INGEST=1: Skipping heavy CSV/collection loading for fast Cloud Run start")
        # But still initialize lightweight components
        try:
            # Initialize LLM and embeddings (fast)
            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("✅ LLM and embeddings initialized")
            
            # Initialize SQL Database (fast - connects to existing olist.db)
            try:
                sqlite_db_path = os.getenv("SQLITE_DB_PATH", "olist.db")
                db_uri = f"sqlite:///{sqlite_db_path}"
                
                if os.path.exists(sqlite_db_path):
                    db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
                    sql_chain = SimpleSQLQueryChain(llm, db)
                    sql_rag_agent = SQLRagAgent(db, llm, sql_chain)
                    logger.info("✅ SQL RAG agent initialized")
                else:
                    logger.warning(f"⚠️  SQLite database not found at {sqlite_db_path}")
                    db = None
                    sql_chain = None
                    sql_rag_agent = None
            except Exception as se:
                logger.warning(f"⚠️  SQL initialization failed: {se}")
                db = None
                sql_chain = None
                sql_rag_agent = None
            
            # Initialize Qdrant client (fast - no collection loading)
            try:
                qdrant_client = QdrantClient(
                    url=QDRANT_URL, 
                    timeout=3,
                    prefer_grpc=False,
                    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
                )
                
                # Initialize vectorstore (points to existing collection, no upload)
                from langchain_qdrant import QdrantVectorStore
                vectorstore = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=QDRANT_COLLECTION,
                    embedding=embeddings
                )
                logger.info("✅ Qdrant vectorstore initialized")
                
                # Initialize RAG agents with existing vectorstore
                qdrant_rag_agent = QdrantRagAgent(vectorstore, llm, embeddings)
                logger.info("✅ Qdrant RAG agent initialized")
            except Exception as qe:
                logger.warning(f"⚠️  Qdrant initialization failed (will use lazy init): {qe}")
                qdrant_rag_agent = None
            
            logger.info("✅ FastAPI Application Ready (fast startup mode)")
        except Exception as e:
            logger.exception(f"❌ Fast startup initialization failed: {e}")
            logger.info("⚠️  Continuing with minimal initialization")
        return
    
    # Initialize SimpleAgent (uses OpenAI key)
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        agent = SimpleAgent(openai_api_key=api_key)
        logger.info("✅ SimpleAgent initialized")
    except Exception as e:
        logger.warning(f"⚠️  SimpleAgent initialization failed: {e}")
        agent = None

    # Initialize SQL Database (SQLite3) and LLM
    try:
        # Use SQLite3 database (file-based)
        sqlite_db_path = os.getenv("SQLITE_DB_PATH", "olist.db")
        db_uri = f"sqlite:///{sqlite_db_path}"
        
        # Check if database already exists and has data
        db_exists = os.path.exists(sqlite_db_path)
        needs_csv_load = not db_exists
        
        if db_exists:
            # Quick check if tables exist
            try:
                conn = sqlite3.connect(sqlite_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
                has_tables = cursor.fetchone() is not None
                conn.close()
                needs_csv_load = not has_tables
            except:
                needs_csv_load = True
        
        if needs_csv_load:
            logger.info(f"Loading CSV data into SQLite database: {sqlite_db_path}")
            
            # Load only essential tables quickly
            csv_files = {
                "products": "products.csv",
                "orders": "orders.csv",
                "customers": "customers.csv"
            }
            
            loaded_count = 0
            for table_name, csv_filename in csv_files.items():
                csv_path = os.path.join(os.path.dirname(__file__), "isi olist db", csv_filename)
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        conn = sqlite3.connect(sqlite_db_path)
                        df.to_sql(table_name, conn, if_exists="replace", index=False)
                        conn.close()
                        logger.info(f"✅ Loaded {len(df)} rows into '{table_name}'")
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"❌ Failed to load {csv_filename}: {e}")
            
            logger.info(f"Loaded {loaded_count}/{len(csv_files)} essential tables")
        else:
            logger.info(f"Database already exists at {sqlite_db_path}, skipping CSV load")
        
        # Ensure additional tables exist
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT,
                agent_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"SQLite3 database initialized at {sqlite_db_path}")
        
        # Initialize LangChain SQLDatabase
        db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Lightweight SQL generation chain implemented locally to avoid version mismatch
        sql_chain = SimpleSQLQueryChain(llm, db)
        logger.info("SQL Database and chain initialized with SQLite3")
    except Exception as e:
        logger.exception("Failed to initialize SQL components: %s", e)
        db = None
        sql_chain = None

    # Initialize Qdrant and vectorstore (with shorter timeout for Cloud Run)
    try:
        logger.info(f"Initializing Qdrant vector store...")
        collection_name = QDRANT_COLLECTION
        
        # Initialize embeddings first (fast operation)
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        # Check if Qdrant is reachable (reduced timeout for faster startup)
        try:
            qdrant_client = QdrantClient(
                url=QDRANT_URL, 
                timeout=3,  # Very short timeout for startup
                prefer_grpc=False,
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
            )
            collections_response = qdrant_client.get_collections()
            qdrant_connected = True
            logger.info(f"✅ Qdrant connected")
        except Exception as e:
            qdrant_connected = False
            logger.warning(f"⚠️  Could not connect to Qdrant during startup: {e}")
            logger.warning("Application will continue without Qdrant (lazy init on first use)")
            qdrant_client = None
            vectorstore = None
            # Skip Qdrant initialization but continue with RAG agents setup
        
        # Check if collection already exists (only if connected)
        if qdrant_connected:
            collection_exists = any(c.name == collection_name for c in collections_response.collections)
            
            if collection_exists:
                logger.info(f"📦 Collection '{collection_name}' already exists. Loading existing collection...")
                # Load existing collection WITHOUT re-uploading documents
                from langchain_qdrant import QdrantVectorStore
                vectorstore = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=collection_name,
                    embedding=embeddings
                )
                logger.info(f"✅ Loaded existing collection: {collection_name}")
            else:
                if DISABLE_INGEST:
                    logger.warning(
                        f"DISABLE_INGEST is set. Skipping creation of missing collection '{collection_name}'."
                    )
                    vectorstore = None
                else:
                    logger.info(f"🆕 Collection '{collection_name}' does not exist. Creating with documents from CSV...")
                # Load reviews CSV and create documents
                try:
                    reviews_csv_path = os.path.join(os.path.dirname(__file__), "isi olist db", "order_reviews.csv")
                    df_reviews = pd.read_csv(reviews_csv_path)
                    reviews_df = df_reviews.copy()
                    logger.info(f"Loaded {len(df_reviews)} reviews from CSV")
                    
                    # Create documents from reviews (chunk by review)
                    from langchain_core.documents import Document
                    chunked_documents = []
                    for idx, row in df_reviews.iterrows():
                        # Combine review title and comment
                        review_text = f"Review Title: {row.get('review_comment_title', 'N/A')}\n"
                        review_text += f"Review: {row.get('review_comment_message', 'N/A')}\n"
                        review_text += f"Score: {row.get('review_score', 'N/A')}"
                        
                        metadata = {
                            "review_id": str(row.get('review_id', '')),
                            "order_id": str(row.get('order_id', '')),
                            "review_score": int(row.get('review_score', 0)) if pd.notna(row.get('review_score')) else 0,
                            "source": "olist_reviews"
                        }
                        
                        doc = Document(page_content=review_text, metadata=metadata)
                        chunked_documents.append(doc)
                    
                    logger.info(f"Created {len(chunked_documents)} document chunks")
                    
                    # Create collection with documents
                    from langchain_qdrant import QdrantVectorStore
                    qdrant_kwargs = {
                        "documents": chunked_documents,
                        "embedding": embeddings,
                        "collection_name": collection_name,
                        "url": QDRANT_URL,
                        "prefer_grpc": False,
                        "force_recreate": False,
                    }
                    
                    if QDRANT_API_KEY and QDRANT_URL and QDRANT_URL.lower().startswith("https"):
                        qdrant_kwargs["api_key"] = QDRANT_API_KEY
                    elif QDRANT_API_KEY:
                        logger.warning("QDRANT_API_KEY is set but QDRANT_URL is not HTTPS. Not sending API key.")
                    
                    vectorstore = QdrantVectorStore.from_documents(**qdrant_kwargs)
                    logger.info(f"✅ Successfully stored {len(chunked_documents)} documents to collection: {collection_name}")
                    
                except FileNotFoundError:
                    logger.warning(f"Reviews CSV not found at {reviews_csv_path}. Collection will be empty.")
                    vectorstore = None
                except Exception as doc_error:
                    logger.exception(f"Error loading documents: {doc_error}")
                    vectorstore = None
            
            logger.info("✅ Qdrant vector store initialized")

            # Initialize products semantic collection (optional)
            try:
                products_collection = os.getenv("QDRANT_PRODUCTS_COLLECTION", "olist_products_semantic")
                exists_products = any(c.name == products_collection for c in qdrant_client.get_collections().collections)
                from langchain_qdrant import QdrantVectorStore as QdrantVS
                if exists_products:
                    vectorstore_products = QdrantVS(
                        client=qdrant_client,
                        collection_name=products_collection,
                        embeddings=embeddings,
                    )
                    logger.info(f"✅ Loaded products semantic collection: {products_collection}")
                else:
                    if DISABLE_INGEST:
                        logger.warning(f"Products collection '{products_collection}' missing; ingestion disabled.")
                        vectorstore_products = None
                    else:
                        # Build simple documents from SQLite products
                        try:
                            sqlite_db_path = os.getenv("SQLITE_DB_PATH", "olist.db")
                            conn = sqlite3.connect(sqlite_db_path)
                            dfp = pd.read_sql_query("SELECT product_id, product_category_name, product_name_length, product_description_length FROM products", conn)
                            conn.close()
                            docs = []
                            for _, r in dfp.iterrows():
                                text = f"Category: {r.get('product_category_name','')} | Name length: {r.get('product_name_length','')} | Description length: {r.get('product_description_length','')}"
                                meta = {
                                    "product_id": str(r.get("product_id", "")),
                                    "product_category_name": r.get("product_category_name", ""),
                                    "product_name_length": r.get("product_name_length", None),
                                    "product_description_length": r.get("product_description_length", None),
                                }
                                docs.append(Document(page_content=text, metadata=meta))
                            vectorstore_products = QdrantVS.from_documents(
                                documents=docs,
                                embedding=embeddings,
                                collection_name=products_collection,
                                url=QDRANT_URL,
                                prefer_grpc=False,
                                api_key=QDRANT_API_KEY if (QDRANT_API_KEY and QDRANT_URL.lower().startswith("https")) else None,
                            )
                            logger.info(f"✅ Created products semantic collection '{products_collection}' with {len(dfp)} items")
                        except Exception as pe:
                            logger.warning(f"Failed to create products collection: {pe}")
                            vectorstore_products = None
            except Exception as e_prod:
                logger.warning(f"Products semantic init skipped: {e_prod}")
    except Exception as e:
        logger.exception(f"❌ Error initializing Qdrant vector store: {e}")
        qdrant_client = None
        vectorstore = None

    # Initialize RAG Agents (must be inside startup_event where variables are defined)
    if llm and db and sql_chain:
        sql_rag_agent = SQLRagAgent(db, llm, sql_chain)
        logger.info("✅ SQL RAG Agent initialized")
    else:
        logger.warning("⚠️  SQL RAG Agent not initialized (missing llm, db, or sql_chain)")

    if vectorstore and llm and embeddings:
        qdrant_rag_agent = QdrantRagAgent(vectorstore, llm, embeddings)
        logger.info("✅ Qdrant RAG Agent initialized")
    else:
        logger.warning("⚠️  Qdrant RAG Agent not initialized (missing vectorstore, llm, or embeddings)")

    # Log completion of startup
    logger.info("=" * 50)
    logger.info("🎉 FastAPI Application Ready to Accept Requests!")
    logger.info("=" * 50)


# ================ LIGHTWEIGHT SQL QUERY CHAIN =================
class SimpleSQLQueryChain:
    """Minimal wrapper to generate a single SELECT SQL statement using LLM.

    This avoids depending on moving targets in langchain.chains.
    """
    def __init__(self, llm: ChatOpenAI, db: SQLDatabase):
        self.llm = llm
        self.db = db

    def invoke(self, inputs):
        question = inputs.get("question") if isinstance(inputs, dict) else str(inputs)
        try:
            schema = self.db.get_table_info()
        except Exception:
            schema = ""
        prompt = f"""
You are a SQL assistant for SQLite. Given the schema and the user's question, output ONLY a single valid SELECT SQL query. No narration.

Schema:
{schema}

Question:
{question}

SQL:
""".strip()
        return self.llm.predict(prompt)


# ================= AGENTIC RAG SYSTEMS =================

# Category Synonyms for better search
CATEGORY_SYNONYMS = {
    "jam tangan": "relogio", "watch": "relogio",
    "parfum": "perfumaria", "perfume": "perfumaria",
    "telepon": "telefonia", "phone": "telefonia",
    "tv": "tvs", "komputer": "informatica_acessorios",
    "computer": "informatica_acessorios",
    # furniture synonyms (Indonesian/English)
    "furnitur": "moveis_decoracao", "furniture": "moveis_decoracao",
    "mebel": "moveis_decoracao", "perabot": "moveis_decoracao",
    # art/arts synonyms
    "art": "artes", "arte": "artes",
}

def normalize_category(query: str) -> str:
    """Normalize category terms to dataset values."""
    lower = query.lower()
    for synonym, dataset_cat in CATEGORY_SYNONYMS.items():
        if synonym in lower:
            return dataset_cat
    return None


# ================= SQL RAG AGENT =================
class SQLRagAgent:
    """RAG Agent for SQL database queries and analysis."""
    
    def __init__(self, db, llm, sql_chain):
        self.db = db
        self.llm = llm
        self.sql_chain = sql_chain
        
    def query(self, question: str) -> str:
        """Execute SQL query based on natural language question."""
        if not self.db or not self.sql_chain:
            return "SQL database not initialized."
        
        try:
            # Clean markdown code blocks
            import re
            cleaned = question.strip()
            cleaned = re.sub(r"```sql|```", "", cleaned, flags=re.IGNORECASE).strip()
            
            # Generate SQL if natural language
            if not cleaned.upper().startswith("SELECT"):
                try:
                    generated_sql = self.sql_chain.invoke({"question": cleaned})
                    if isinstance(generated_sql, str):
                        cleaned = generated_sql
                    logger.info(f"Generated SQL: {cleaned}")
                except Exception as e:
                    return f"Error generating SQL: {str(e)}"
            
            # Clean markdown from generated SQL
            cleaned = re.sub(r"```sql|```", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"^SQLQuery:\s*", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"^\s*\n", "", cleaned).strip()
            
            # Execute SQL
            sqlite_db_path = os.getenv("SQLITE_DB_PATH", "olist.db")
            conn = sqlite3.connect(sqlite_db_path)
            df = pd.read_sql_query(cleaned, conn)
            conn.close()
            
            if df.empty:
                return "Query returned no results."
            
            # Format results
            result = f"SQL Query Results ({len(df)} rows):\n\n"
            result += df.to_string(index=False, max_rows=10)
            
            if len(df) > 10:
                result += f"\n\n... and {len(df) - 10} more rows"
            
            return result
            
        except Exception as e:
            logger.exception("SQL query error")
            return f"SQL Error: {str(e)}"
    
    def analyze(self, query: str) -> str:
        """Analyze data and provide insights."""
        raw_data = self.query(query)
        
        if "Error" in raw_data or "not initialized" in raw_data:
            return raw_data
        
        # Use LLM to generate insights
        prompt = f"""
Analyze the following SQL query results and provide business insights:

User Question: {query}

Data:
{raw_data}

Provide:
1. Key findings (in bullet points)
2. Statistical summary
3. Business recommendations

Keep the analysis concise and actionable.
"""
        try:
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            return f"Analysis error: {str(e)}"


# ================= QDRANT RAG AGENT =================
class QdrantRagAgent:
    """RAG Agent for Qdrant vector search and review analysis."""
    
    def __init__(self, vectorstore, llm, embeddings):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        
    def search(self, query: str, k: int = 5) -> str:
        """Search for relevant reviews."""
        if not self.vectorstore:
            return "Qdrant vector store not initialized."
        
        try:
            # Normalize category if present
            normalized_cat = normalize_category(query)
            if normalized_cat:
                logger.info(f"✅ Category identified: {query} -> {normalized_cat}")
            else:
                logger.info(f"ℹ️  No specific category detected in query: {query}")
            
            # Use raw Qdrant search since vectorstore doesn't populate page_content from existing collection
            # Get embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search using Qdrant client directly
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qdrant_client = self.vectorstore.client
            collection_name = self.vectorstore.collection_name
            
            # Build query filter if category is detected
            query_filter = None
            if normalized_cat:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="product_category",
                            match=MatchValue(value=normalized_cat)
                        )
                    ]
                )
                logger.info(f"🔍 Applying Qdrant filter for category: {normalized_cat}")
            
            search_results = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=k,
                with_payload=True
            ).points
            
            logger.info(f"Found {len(search_results)} results from Qdrant")
            if normalized_cat:
                logger.info(f"📊 Filtered by category: {normalized_cat}")
            
            # Convert Qdrant results to Document-like format
            from langchain_core.documents import Document
            results = []
            for hit in search_results:
                payload = hit.payload or {}
                # Use 'text' field as page_content
                page_content = payload.get('text', '')
                # Keep other fields as metadata
                metadata = {k: v for k, v in payload.items() if k != 'text'}
                metadata['_id'] = hit.id
                metadata['_score'] = hit.score
                results.append(Document(page_content=page_content, metadata=metadata))
            
            if not results:
                category_info = f" (category: {normalized_cat})" if normalized_cat else ""
                return f"No reviews found for: {query}{category_info}"
            
            # Filter by category if normalized
            if normalized_cat:
                filtered = []
                for doc in results:
                    # Check both field names: product_category (from olist_products) and product_category_name (from olist_reviews)
                    cat = doc.metadata.get("product_category", doc.metadata.get("product_category_name", ""))
                    if cat and normalized_cat.lower() in cat.lower():
                        filtered.append(doc)
                if filtered:
                    results = filtered
                    logger.info(f"📊 After category filter: {len(results)} results in category: {normalized_cat}")
                else:
                    logger.warning(f"⚠️  Category filter removed all results. Using unfiltered results.")
                    # Don't filter if it removes everything - the Qdrant filter already constrained results
            
            # Format response - adapted for merged product format
            category_header = f"[Searching in category: {normalized_cat}]\n\n" if normalized_cat else ""
            review_texts = [category_header] if normalized_cat else []
            for i, doc in enumerate(results[:5], 1):
                meta = doc.metadata
                # Handle both individual review format and merged product format
                if 'review_score' in meta:
                    # Individual review format
                    review_texts.append(
                        f"Review {i}:\n"
                        f"Score: {meta.get('review_score', 'N/A')}/5\n"
                        f"Content: {doc.page_content[:300]}...\n"
                    )
                else:
                    # Merged product format
                    review_texts.append(
                        f"Product {i}:\n"
                        f"Category: {meta.get('product_category', 'N/A')} ({meta.get('product_category_en', 'N/A')})\n"
                        f"Average Score: {meta.get('avg_review_score', 'N/A')}/5\n"
                        f"Number of Reviews: {meta.get('num_reviews', 'N/A')}\n"
                        f"Content: {doc.page_content[:500]}...\n"
                    )
            
            return "\n".join(review_texts)
            
        except Exception as e:
            logger.exception("Qdrant search error")
            return f"Search error: {str(e)}"
    
    def analyze(self, query: str) -> str:
        """Search reviews and provide sentiment analysis."""
        raw_reviews = self.search(query)
        logger.info(f"Search returned {len(raw_reviews)} chars: {raw_reviews[:200]}...")
        
        if "Error" in raw_reviews or "not initialized" in raw_reviews:
            return raw_reviews
        
        # Use LLM to analyze sentiment
        prompt = f"""
Analyze the following customer reviews and provide insights:

User Question: {query}

Reviews:
{raw_reviews}

Provide a structured analysis:
1. Overall Sentiment Summary
2. Common Positive Themes (if any)
3. Common Negative Themes (if any)
4. Key Insights and Recommendations

Use the same language as the user's question. Be concise and actionable.
"""
        try:
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            return f"Analysis error: {str(e)}"


# ================= INITIALIZE RAG AGENTS =================
# These will be initialized in startup_event
sql_rag_agent = None
qdrant_rag_agent = None

# ================= LEGACY TOOLS (backward compatibility) =================
@tool
def current_datetime(query: str = "") -> str:
    """Get current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_reviews(query: str) -> str:
    """Searches for customer reviews matching the query."""
    if not vectorstore:
        return "Vector store not initialized."
    try:
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return "No relevant reviews found."
        response = "\n\n".join([f"Review: {d.page_content}\nMetadata: {d.metadata}" for d in results])
        return response
    except Exception as e:
        return f"Error searching reviews: {e}"

@tool
def get_review_statistics(product_id: str = "") -> str:
    """Get review statistics for a specific product or all products."""
    try:
        global reviews_df
        if reviews_df is None:
            return "Reviews data not loaded."
        if product_id and 'product_id' in reviews_df.columns:
            filtered = reviews_df[reviews_df['product_id'] == product_id]
        else:
            filtered = reviews_df
        
        if len(filtered) == 0:
            return "No reviews found."
        
        if 'review_score' in filtered.columns:
            stats = f"""Review Statistics:
- Total Reviews: {len(filtered)}
- Average Rating: {filtered['review_score'].mean():.2f}
- Min Rating: {filtered['review_score'].min()}
- Max Rating: {filtered['review_score'].max()}"""
        else:
            stats = f"Total Reviews: {len(filtered)}"
        return stats
    except Exception as e:
        return f"Error getting statistics: {e}"

@app.get("/reviews/ask")
async def reviews_ask(q: str):
    """Ask the reviews agent a question using the review tools and vector store."""
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
        # Ensure vectorstore exists for search tool usefulness
        if vectorstore is None:
            raise HTTPException(status_code=503, detail="Vectorstore not initialized")
        # Lazily construct a lightweight agent if not created globally
        global review_agent, llm
        if llm is None:
            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        # The review_agent may be defined earlier conditionally; recreate if missing
        if 'review_agent' not in globals() or review_agent is None:
            try:
                from langgraph.prebuilt import create_react_agent
                tools = [search_reviews, current_datetime, get_review_statistics]
                # Newer create_react_agent signature does not accept 'prompt'; we will inject system message at call time.
                review_agent_local = create_react_agent(tools=tools, model=llm)
                globals()['review_agent'] = review_agent_local
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize review agent: {e}")
        # Prepend a system instruction to guide the agent
        system_msg = SystemMessage(content="You analyze customer reviews and use provided tools to answer succinctly.")
        result = review_agent.invoke({"messages": [system_msg, HumanMessage(content=q)]})
        answer = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
        return {"question": q, "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reviews agent error: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the agent if API key is available
review_agent = None
if OPENAI_API_KEY and vectorstore:
    review_agent = create_react_agent(
        tools=[search_reviews, current_datetime, get_review_statistics],
        model=llm,
    )
    
    def get_chat_bot_response(input_text, chat_history):
        """Get response from the chatbot agent."""
        try:
            if review_agent is None:
                return "Agent not initialized"
            result = review_agent.invoke(
                {"messages": chat_history + [HumanMessage(content=input_text)]}
            )
            return result["messages"][-1].content
        except Exception as e:
            logger.error(f"Error in chatbot response: {e}")
            return f"Sorry, I encountered an error: {e}"
else:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set. Please configure your API key.")
    if not vectorstore:
        logger.error("Vector store not initialized. Please check your Qdrant connection.")
    # try:
    #     QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    #     QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6338")
    #     headers = {"Content-Type": "application/json"}
    #     if QDRANT_API_KEY:
    #         headers["Authorization"] = f"Bearer {QDRANT_API_KEY}"

    #     qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, api_key=QDRANT_API_KEY)
    #     embeddings = OpenAIEmbeddings()
    #     vectorstore = Qdrant(client=qdrant_client, collection_name="olist_review", embeddings=embeddings)
    #     # retriever = vectorstore.as_retriever(client=qdrant_client, collection_name="olist_review", embedding_fn=embeddings, top_k=5)
    #     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
    #     # Setup an agent that can use the retriever
    #     ## add some tools
        

    #     qdrant_tool = Tool(
    #         name="QdrantRetriever",
    #         func=lambda q: retriever.get_relevant_documents(q),
    #         description="Use this to retrieve context from Qdrant vector DB"
    #     )
    #     tools = [qdrant_tool]
    #     if llm is None:
    #         # instantiate a lightweight llm if not available
    #         llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
    #         agent_a = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
    #         rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    #     logger.info("Qdrant vectorstore and RAG chain initialized")
    # except Exception as e:
    #     logger.exception("Failed to initialize Qdrant/vectorstore: %s", e)
    #     qdrant_client = None
    #     vectorstore = None
    #     retriever = None
    #     agent_a = None
    #     rag_chain = None

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "LLM Agent Service is running!",
        "status": "healthy",
        "endpoints": {
            "ask": "/ask",
            "sqlite": "/sqlite",
            "sqlite_raw": "/sqlite/raw",
            "qdrant": "/qdrant",
            "chat": "/chat",
            "health": "/health",
            "history": "/history"
        }
    }
@app.post("/upload")
async def upload_doc(file: UploadFile):
    """Upload a text file and store embeddings in Qdrant."""
    if DISABLE_INGEST:
        raise HTTPException(status_code=403, detail="Ingestion is disabled by configuration (DISABLE_INGEST)")
    content = await file.read()
    text = content.decode("utf-8")

    # Split into chunks (simple example)
    docs = [text[i:i+500] for i in range(0, len(text), 500)]
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vectorstore not initialized")
    vectorstore.add_texts(docs)

    return {"status": "uploaded", "chunks": len(docs)}

@app.get("/collections")
def list_collections():
    """List Qdrant collections."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    return qdrant_client.get_collections()

@app.post("/ask")
async def ask_question(question: str):
    """Ask a question using the RAG chain"""
    if not agent_a:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    response = agent_a.run(question)
    return {"question": question, "answer": response}

@app.post("/sqlite")
# def query_sqlite(q: str = "List top 5 customers by total orders"):
def query_sqlite(q: str = "List top 5 products from the products table"):
    if not sql_chain:
        raise HTTPException(status_code=503, detail="SQL chain not initialized")
    try:
        # Generate SQL with the chain
        sql_text = sql_chain.invoke({"question": q})
        if isinstance(sql_text, dict):
            # Some versions may return a dict
            sql_text = sql_text.get("sql") or sql_text.get("text") or ""
        if not sql_text:
            raise ValueError("Failed to generate SQL from question")

        # Sanitize common LLM artifacts (labels, code fences)
        s = str(sql_text).strip()
        # Remove triple backtick fences if present
        if s.startswith("```"):
            s = s.strip("`")
        s = s.replace("```sql", "").replace("```", "").strip()
        # Remove leading labels like 'SQLQuery:' or 'SQL:'
        for prefix in ["SQLQuery:", "SQL Query:", "SQL:"]:
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):].strip()
        # Ensure single statement and SELECT-only for safety
        if ";" in s:
            # Keep only first statement
            s = s.split(";")[0].strip()
        if not s.lower().startswith("select"):
            # Fall back to a safe default if the chain emitted narration
            s = "SELECT product_id, product_category_name FROM products LIMIT 5"

        # Execute SQL against SQLite and return rows
        conn = sqlite3.connect(os.getenv("SQLITE_DB_PATH", "olist.db"))
        df = pd.read_sql_query(s, conn)
        conn.close()

        # Build a small markdown table for Streamlit rendering
        def df_to_markdown(dframe: pd.DataFrame, max_rows: int = 10) -> str:
            if dframe.empty:
                return "No rows returned."
            d = dframe.head(max_rows)
            header = "| " + " | ".join(map(str, d.columns)) + " |\n"
            sep = "| " + " | ".join(["---"] * len(d.columns)) + " |\n"
            rows = "".join("| " + " | ".join(map(lambda x: str(x), row)) + " |\n" for row in d.values)
            return header + sep + rows

        result_md = df_to_markdown(df)
        return {
            "question": q,
            "sql": s,
            "rows": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "result": result_md
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQLite query error: {e}")

@app.post("/sqlite/raw")
def sqlite_raw(sql: str):
    """Execute a raw SELECT SQL safely and return rows.

    Notes:
    - Only single-statement SELECT queries are allowed.
    - Rejects semicolons to prevent multiple statements.
    - Returns columns, rows, and a small markdown preview for UIs.
    """
    try:
        if not sql or not isinstance(sql, str):
            raise HTTPException(status_code=400, detail="Missing 'sql' query parameter")
        sql_stripped = sql.strip()
        if ";" in sql_stripped:
            raise HTTPException(status_code=400, detail="Multiple statements are not allowed")
        if not sql_stripped.lower().startswith("select"):
            raise HTTPException(status_code=400, detail="Only SELECT statements are permitted")

        conn = sqlite3.connect(os.getenv("SQLITE_DB_PATH", "olist.db"))
        df = pd.read_sql_query(sql_stripped, conn)
        conn.close()

        def df_to_markdown(dframe: pd.DataFrame, max_rows: int = 10) -> str:
            if dframe.empty:
                return "No rows returned."
            d = dframe.head(max_rows)
            header = "| " + " | ".join(map(str, d.columns)) + " |\n"
            sep = "| " + " | ".join(["---"] * len(d.columns)) + " |\n"
            rows = "".join("| " + " | ".join(map(lambda x: str(x), row)) + " |\n" for row in d.values)
            return header + sep + rows

        return {
            "sql": sql_stripped,
            "rows": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "result": df_to_markdown(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQLite RAW query error: {e}")

@app.post("/qdrant")
def query_qdrant():
    # r = requests.get("https://acb9e0ed-c7e4-4abc-9495-1382817b533e.europe-west3-0.gcp.cloud.qdrant.io/collections/resume/points/search") 
        
    if not QDRANT_URL:
        raise HTTPException(status_code=503, detail="QDRANT_URL not configured")
    # Use Qdrant REST search endpoint for the configured collection
    # Ensure we don't double-prefix http:// if QDRANT_URL already includes the scheme
    base = QDRANT_URL.rstrip("/")
    url = f"{base}/collections/{QDRANT_COLLECTION}/points/search"
    body = {
        "vector": [0.01] * 1536,
        "limit": 5,
        "with_payload": True
    }
    r = requests.post(url, headers=headers, json=body)
    r.raise_for_status()
    return r.json()

@app.post("/qdrant/search")
def qdrant_search(q: str, k: int = 5):
    """Search Qdrant collection using embeddings and return top-k results.

    Uses Qdrant REST API directly to avoid client version/method mismatches.
    """
    # Validate configuration
    if not QDRANT_URL:
        raise HTTPException(status_code=503, detail="QDRANT_URL not configured")
    if not QDRANT_COLLECTION:
        raise HTTPException(status_code=503, detail="QDRANT_COLLECTION not configured")
    # Ensure embeddings available
    global embeddings
    if embeddings is None:
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as ee:
            raise HTTPException(status_code=503, detail=f"Embeddings init failed: {ee}")

    try:
        vec = embeddings.embed_query(q)
        # If the query mentions a known category, restrict search to it
        normalized_cat = normalize_category(q)
        if normalized_cat:
            logger.info(f"✅ /qdrant/search - Category filter: {q} -> {normalized_cat}")
        base = QDRANT_URL.rstrip("/")
        url = f"{base}/collections/{QDRANT_COLLECTION}/points/search"
        body = {
            "vector": vec,
            "limit": int(k),
            "with_payload": True,
            "with_vector": False
        }
        if normalized_cat:
            # Constrain results to the dataset category (Portuguese field)
            body["filter"] = {
                "must": [
                    {"key": "product_category", "match": {"value": normalized_cat}}
                ]
            }
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Normalize response
        results = []
        for item in data.get("result", []):
            results.append({
                "score": item.get("score"),
                "id": item.get("id"),
                "payload": item.get("payload", {})
            })
        return {
            "query": q,
            "k": k,
            "category_filter": normalized_cat,
            "results": results
        }
    except requests.HTTPError as he:
        raise HTTPException(status_code=he.response.status_code if he.response else 500, detail=f"Qdrant HTTP error: {he}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant search error: {e}")

@app.post("/products/search")
def products_search(q: str, k: int = 5):
    """Semantic search over SQLite products via Qdrant collection."""
    if not vectorstore_products:
        raise HTTPException(status_code=503, detail="Products vectorstore not initialized (collection missing or ingestion disabled)")
    try:
        results = vectorstore_products.similarity_search(q, k=k)
        return {
            "query": q,
            "k": k,
            "results": [
                {"text": d.page_content, "metadata": d.metadata} for d in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Products search error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "sql_agent_initialized": sql_rag_agent is not None,
        "qdrant_agent_initialized": qdrant_rag_agent is not None
    }

@app.get("/debug/config")
async def debug_config():
    """Debug endpoint to show active configuration (without exposing secret values)"""
    return {
        "qdrant_url_set": bool(QDRANT_URL),
        "qdrant_api_key_set": bool(QDRANT_API_KEY),
        "qdrant_collection": QDRANT_COLLECTION,
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "disable_ingest": DISABLE_INGEST,
        "category_synonyms": CATEGORY_SYNONYMS,
        "vectorstore_initialized": vectorstore is not None,
        "vectorstore_collection": vectorstore.collection_name if vectorstore else None,
        "qdrant_rag_agent_initialized": qdrant_rag_agent is not None,
        "sql_rag_agent_initialized": sql_rag_agent is not None,
    }

# @app.post("/favicon.ico")
# async def favicon():
#     # No favicon served; return 204 to silence browser requests in logs
#     return Response(status_code=204)

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Intelligent chat endpoint using dual RAG agents (SQL + Qdrant).
    
    Request Body:
        {
            "message": "your question here",
            "agent": "auto" | "sql" | "qdrant" (optional, default: "auto")
        }
    
    The system automatically routes queries to the appropriate agent:
    - SQL Agent: For quantitative analysis (counts, averages, statistics)
    - Qdrant Agent: For qualitative analysis (reviews, sentiments, opinions)
    - Auto: Intelligently decides based on query content
    
    Returns:
        dict: Response with agent type used and analysis
    """
    try:
        message = request.message
        agent_choice = request.agent.lower() if request.agent else "auto"
        session_id = request.session_id

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Ensure LLM available
        global llm
        if llm is None:
            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # Determine which agent to use
        use_sql = False
        use_qdrant = False

        if agent_choice == "sql":
            use_sql = True
        elif agent_choice == "qdrant":
            use_qdrant = True
        else:  # auto routing
            sql_keywords = [
                "berapa", "jumlah", "total", "count", "how many",
                "rata-rata", "average", "mean", "sum",
                "per kategori", "group by", "statistik", "statistics",
            ]
            qdrant_keywords = [
                "review", "ulasan", "pendapat", "opini", "opinion",
                "sentiment", "feedback", "bagus", "jelek",
                "pengalaman", "experience", "kata pelanggan",
            ]
            lower = message.lower()
            if any(k in lower for k in sql_keywords):
                use_sql = True
            elif any(k in lower for k in qdrant_keywords):
                use_qdrant = True
            else:
                # Default to qualitative if unsure
                use_qdrant = True

        # Build RAG contexts
        agents_used = []
        sql_context = None
        qdrant_context = None

        if use_sql:
            if sql_rag_agent:
                try:
                    # Use query() to retrieve raw tabular context for RAG
                    sql_context = sql_rag_agent.query(message)
                    agents_used.append("SQL")
                except Exception as e:
                    logger.exception("SQL RAG agent error")
                    sql_context = f"SQL agent error: {str(e)}"
                    agents_used.append("SQL")
            else:
                sql_context = "SQL RAG agent not initialized."

        if use_qdrant:
            if qdrant_rag_agent:
                try:
                    # Use search() to retrieve top review snippets for RAG
                    detected_category = normalize_category(message)
                    if detected_category:
                        logger.info(f"✅ /chat - Using category filter: {detected_category}")
                    qdrant_context = qdrant_rag_agent.search(message)
                    agents_used.append("Qdrant")
                except Exception as e:
                    logger.exception("Qdrant RAG agent error")
                    qdrant_context = f"Qdrant agent error: {str(e)}"
                    agents_used.append("Qdrant")
            else:
                qdrant_context = "Qdrant RAG agent not initialized."

        # Synthesize final answer using contexts when possible
        final_response = None
        if (sql_context and isinstance(sql_context, str)) or (qdrant_context and isinstance(qdrant_context, str)):
            try:
                sql_section = f"SQL Data Context:\n{sql_context}" if sql_context else ""
                qdrant_section = f"Review Context:\n{qdrant_context}" if qdrant_context else ""
                final_prompt = f"""
You are a helpful data and reviews analyst. Answer the user's question using ONLY the provided contexts. If a context is missing, say so briefly. Be concise and actionable.

User Question:
{message}

{sql_section}

{qdrant_section}

Provide:
- A direct answer in the user's language
- Key insights in bullet points
- If relevant, a short recommendation
""".strip()
                final_response = llm.predict(final_prompt)
            except Exception as e:
                logger.warning(f"LLM synthesis failed, falling back to per-agent responses: {e}")

        # Fallback: if synthesis failed, use per-agent analysis
        if not final_response:
            per_agent_responses = []
            if use_sql and sql_rag_agent:
                per_agent_responses.append({
                    "agent": "SQL",
                    "response": sql_rag_agent.analyze(message)
                })
            elif use_sql:
                per_agent_responses.append({
                    "agent": "SQL",
                    "response": sql_context or "SQL RAG agent not initialized."
                })
            if use_qdrant and qdrant_rag_agent:
                per_agent_responses.append({
                    "agent": "Qdrant",
                    "response": qdrant_rag_agent.analyze(message)
                })
            elif use_qdrant:
                per_agent_responses.append({
                    "agent": "Qdrant",
                    "response": qdrant_context or "Qdrant RAG agent not initialized."
                })

            if not per_agent_responses:
                raise HTTPException(status_code=503, detail="No RAG agents available. Please check system configuration.")

            if len(per_agent_responses) == 1:
                final_response = per_agent_responses[0]["response"]
            else:
                combined = [f"--- {r['agent']} Agent ---\n{r['response']}" for r in per_agent_responses]
                final_response = "\n\n=== COMBINED ANALYSIS ===\n\n" + "\n\n".join(combined)

        # Persist chat to SQLite (best-effort)
        try:
            sqlite_db_path = os.getenv("SQLITE_DB_PATH", "olist.db")
            conn = sqlite3.connect(sqlite_db_path)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO chat_history (user_message, agent_response) VALUES (?, ?)",
                (message if not session_id else f"[{session_id}] {message}", final_response),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to persist chat history: {e}")

        return {
            "user_message": message,
            "agent_response": final_response,
            "agents_used": agents_used,
            "agent_choice": agent_choice,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")



if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )

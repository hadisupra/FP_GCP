#!/usr/bin/env python3
"""Create keyword index on product_category field in Qdrant collection."""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
from dotenv import load_dotenv
import toml

load_dotenv()

# Load secrets.toml if available
secrets_path = os.path.join(os.path.dirname(__file__), "secrets.toml")
if os.path.exists(secrets_path):
    secrets = toml.load(secrets_path)
    for k, v in secrets.items():
        if v and not os.getenv(k):
            os.environ[k] = v

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "olist_products")

def create_category_index():
    """Create a keyword index on the product_category field."""
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60
        )
        
        print(f"Creating keyword index on 'product_category' in collection '{QDRANT_COLLECTION}'...")
        
        # Create index for product_category field
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="product_category",
            field_schema=PayloadSchemaType.KEYWORD
        )
        
        print("✅ Index created successfully!")
        
        # Verify index was created
        collection_info = client.get_collection(QDRANT_COLLECTION)
        print("\nCollection payload schema:")
        for field, schema in collection_info.payload_schema.items():
            print(f"  - {field}: {schema}")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        raise

if __name__ == "__main__":
    create_category_index()

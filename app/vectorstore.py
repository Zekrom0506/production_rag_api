from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from app.config import get_settings


def get_vectorstore():
    """
    Create (or connect to) the PGVector store on Supabase.
    Uses a local HuggingFace embedding model - no API key needed.
    """
    settings = get_settings()

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=settings.collection_name,
        connection=settings.database_url,
        use_jsonb=True,
    )

    return vectorstore
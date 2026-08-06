"""
Document ingestion script.
Run manually whenever you want to load new documents into the vector store.

Usage:
    uv run python -m app.ingestion
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.vectorstore import get_vectorstore

# Fixed set of categories the classifier is allowed to choose from.
# Keeping this closed (vs. free-text) makes the output predictable and
# easy to filter/report on later.
CATEGORIES = [
    "hr_policy",
    "finance_policy",
    "legal_contract",
    "technical_documentation",
    "invoice",
    "report",
    "other",
]

# How much of a document's text to show the classifier. Documents are
# usually identifiable from their first chunk of content, and keeping this
# short keeps classification fast and cheap.
CLASSIFY_CHAR_LIMIT = 1500


def get_classifier_llm() -> ChatGroq:
    """LLM used only for document classification during ingestion."""
    settings = get_settings()
    return ChatGroq(
        model=settings.fallback_model,
        api_key=settings.groq_api_key,
        temperature=0,
        timeout=30,
        max_retries=0,
    )


def classify_document(llm: ChatGroq, text: str) -> str:
    """
    Ask the LLM to label a document with one of the fixed CATEGORIES.
    Falls back to "other" if the LLM errors out or returns something
    unexpected, so a classification hiccup never blocks ingestion.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a document classifier. Read the document text and output "
         "ONLY one category label from this exact list, nothing else: "
         f"{', '.join(CATEGORIES)}."),
        ("human", "Document text:\n{text}\n\nCategory:"),
    ])

    try:
        chain = prompt | llm
        result = chain.invoke({"text": text[:CLASSIFY_CHAR_LIMIT]})
        label = result.content.strip().lower()
        return label if label in CATEGORIES else "other"
    except Exception:
        return "other"


def load_documents(data_dir: str = "./data"):
    """Load all .txt and .pdf files from the data directory."""
    docs = []
    data_path = Path(data_dir)

    for file_path in data_path.glob("**/*"):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())

    return docs


def classify_documents(docs: list, llm: ChatGroq) -> None:
    """
    Classify each loaded document and attach the result as
    metadata["category"], in place. Grouping by source file so a
    multi-page PDF is classified once (from its first page) rather
    than once per page.
    """
    docs_by_source: dict[str, list] = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        docs_by_source.setdefault(source, []).append(doc)

    for source, source_docs in docs_by_source.items():
        category = classify_document(llm, source_docs[0].page_content)
        print(f"  {source} -> {category}")
        for doc in source_docs:
            doc.metadata["category"] = category


def ingest_documents(data_dir: str = "./data"):
    print(f"Loading documents from {data_dir}...")
    docs = load_documents(data_dir)
    print(f"Loaded {len(docs)} documents.")

    if not docs:
        print("No documents found. Add .txt or .pdf files to the data/ folder.")
        return

    print("Classifying documents...")
    classifier_llm = get_classifier_llm()
    classify_documents(docs, classifier_llm)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    print("Connecting to vector store and embedding chunks (this may take a minute)...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    print(f"Done. Ingested {len(chunks)} chunks into the vector store.")


if __name__ == "__main__":
    ingest_documents()
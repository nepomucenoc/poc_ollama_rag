import argparse
import logging
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load env variables from .env
load_dotenv()

# Retrieve PostgreSQL credentials from environment variables
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5433")
PG_DB = os.getenv("PG_DB", "ragdb")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.1")

# Build the connection string
connection_string = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"


def load_kb_to_postgres(file_path: str):
    logging.info(f"Loading file: {file_path}")

    # Step 1: Load file content
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Step 2: Split content into chunks
    logging.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Step 3: Generate embeddings using Ollama
    logging.info("Generating embeddings with Ollama (llama3)...")
    embeddings = OllamaEmbeddings(
        model=LOCAL_MODEL,
        base_url="http://localhost:11434",
    )

    # Step 4: Store chunks with embeddings in PostgreSQL using pgvector
    logging.info("Storing vectors in PostgreSQL vector store...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=connection_string,
        collection_name=f"knowledge_base_{LOCAL_MODEL}"
    )

    logging.info("Knowledge base loaded and stored successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load markdown KB into PostgreSQL vector store")
    parser.add_argument("filepath", help="Path to the markdown file (e.g. ./kb/file.md)")
    args = parser.parse_args()

    load_kb_to_postgres(args.filepath)

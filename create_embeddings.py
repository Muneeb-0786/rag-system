#!/usr/bin/env python
# coding: utf-8

# Create Embeddings with Chroma (Persistent)

from helper_utils import word_wrap
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# === Step 1: Load PDF ===
reader = PdfReader("microsoft_annual_report_2022.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]
pdf_texts = [text for text in pdf_texts if text]

print("First page sample:")
print(word_wrap(pdf_texts[0]))

# === Step 2: Chunk Text ===
# Character-based splitting
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

print(f"\nCharacter splitting sample (chunk 10):")
print(word_wrap(character_split_texts[10]))
print(f"Total character chunks: {len(character_split_texts)}")

# Token-based splitting
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

print(f"\nToken splitting sample (chunk 10):")
print(word_wrap(token_split_texts[10]))
print(f"Total token chunks: {len(token_split_texts)}")

# === Step 3: Setup ChromaDB (Persistent) ===
import os
import shutil

# Define an absolute path for the database
persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
print(f"ChromaDB will be stored at: {persist_directory}")

# Remove directory if it exists to start fresh
if os.path.exists(persist_directory):
    print(f"Removing existing directory: {persist_directory}")
    shutil.rmtree(persist_directory)

# Create directory
os.makedirs(persist_directory, exist_ok=True)

# Use PersistentClient directly instead of Client with Settings
from chromadb import PersistentClient
chroma_client = PersistentClient(path=persist_directory)

embedding_function = SentenceTransformerEmbeddingFunction()

# Create or get collection
chroma_collection = chroma_client.get_or_create_collection(
    name="microsoft_annual_report_2022",
    embedding_function=embedding_function
)

# === Step 4: Add Documents to Chroma ===
metadata = {
    "source": "microsoft_annual_report_2022.pdf",
    "description": "Microsoft Annual Report 2022",
    "author": "Microsoft Corporation",
    "date": "2022-12-31"
}

# Use namespaced IDs to avoid collisions
ids = [f"msft2022_{i}" for i in range(len(token_split_texts))]

# Add to collection (embedding done automatically)
chroma_collection.add(
    documents=token_split_texts,
    metadatas=[metadata] * len(token_split_texts),
    ids=ids
)

# After adding documents to the collection
print(f"Embeddings created with {chroma_collection.count()} documents")
print(f"Checking directory: {persist_directory}")
if os.path.exists(persist_directory):
    files = os.listdir(persist_directory)
    print(f"Directory contents: {files}")
    if files:
        print("SUCCESS: Files were created successfully!")
    else:
        print("WARNING: Directory exists but is empty!")
else:
    print("ERROR: Directory was not created!")

# === Done ===
print(f"\nTotal documents embedded in Chroma: {chroma_collection.count()}")
print("\nEmbedding and persistence completed successfully!")

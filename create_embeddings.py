#!/usr/bin/env python
# coding: utf-8

# Create Embeddings with Chroma (Persistent) using Langchain

from helper_utils import word_wrap
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
import shutil

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

# === Step 3: Setup ChromaDB (Persistent) with Langchain ===
# Define an absolute path for the database
persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
print(f"ChromaDB will be stored at: {persist_directory}")

# Remove directory if it exists to start fresh
if os.path.exists(persist_directory):
    print(f"Removing existing directory: {persist_directory}")
    shutil.rmtree(persist_directory)

# Create directory
os.makedirs(persist_directory, exist_ok=True)

# Initialize the HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# === Step 4: Convert text chunks to Langchain Documents ===
documents = []
for i, text_chunk in enumerate(token_split_texts):
    metadata = {
        "source": "microsoft_annual_report_2022.pdf",
        "description": "Microsoft Annual Report 2022",
        "author": "Microsoft Corporation",
        "date": "2022-12-31",
        "chunk_id": f"msft2022_{i}"
    }
    doc = Document(page_content=text_chunk, metadata=metadata)
    documents.append(doc)

print(f"Created {len(documents)} Document objects")

# === Step 5: Create Chroma vector store with Langchain ===
collection_name = "microsoft_annual_report_2022"
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=persist_directory,
    collection_name=collection_name
)

# Persist the vectorstore
vectordb.persist()

# === Step 6: Verify Creation ===
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
print(f"\nTotal documents embedded in Chroma: {len(documents)}")
print("\nEmbedding and persistence completed successfully!")

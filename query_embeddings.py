#!/usr/bin/env python
# coding: utf-8

# Query Embeddings with Chroma and RAG

import os
from helper_utils import word_wrap
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# === Step 1: Connect to Persistent ChromaDB ===
# Use the same absolute path construction as in create_embeddings.py
persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
print(f"Connecting to ChromaDB at: {persist_directory}")

# Check if directory exists
if not os.path.exists(persist_directory):
    print(f"ERROR: Directory {persist_directory} does not exist!")
    print("Please run create_embeddings.py first to create the database.")
    exit(1)

# Use PersistentClient directly instead of Client with Settings
chroma_client = PersistentClient(path=persist_directory)

# Initialize the embedding function (same as used during creation)
embedding_function = SentenceTransformerEmbeddingFunction()

# Get the existing collection
collection_name = "microsoft_annual_report_2022"
try:
    chroma_collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    print(f"Connected to collection '{collection_name}' with {chroma_collection.count()} documents")
except Exception as e:
    print(f"Error connecting to collection: {e}")
    print("Available collections:", chroma_client.list_collections())
    exit(1)

# === Step 2: Query Functions ===
def retrieve_documents(query, n_results=5):
    """Retrieve relevant documents for a given query"""
    results = chroma_collection.query(query_texts=[query], n_results=n_results)
    return results['documents'][0]

def display_retrieved_documents(documents):
    """Display the retrieved documents in a readable format"""
    print("\n=== Retrieved Documents ===\n")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(word_wrap(doc))
        print("\n" + "-"*50 + "\n")

def rag(query, retrieved_documents, model="gemini-2.0-flash"):
    """Generate an answer using RAG with Gemini"""
    information = "\n\n".join(retrieved_documents)

    system_prompt = "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report. Answer the user's question using only the information provided."
    
    gemini = genai.GenerativeModel(model)
    response = gemini.generate_content(
        [
            system_prompt,
            f"Question: {query}. \n Information: {information}"
        ]
    )
    
    content = response.text
    return content

# === Step 3: Interactive Query Loop ===
def main():
    print("\nMicrosoft Annual Report 2022 Query System")
    print("Type 'exit' to quit")
    
    while True:
        print("\n" + "="*50)
        query = input("\nEnter your question: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(query)
        
        # Display retrieved documents
        display_retrieved_documents(retrieved_docs)
        
        # Generate answer with RAG
        answer = rag(query, retrieved_docs)
        
        print("\n=== Generated Answer ===\n")
        print(word_wrap(answer))

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8

# Query Embeddings with Chroma and RAG using Langchain

import os
from helper_utils import word_wrap
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# === Step 1: Connect to Persistent ChromaDB using Langchain ===
persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
print(f"Connecting to ChromaDB at: {persist_directory}")

# Check if directory exists
if not os.path.exists(persist_directory):
    print(f"ERROR: Directory {persist_directory} does not exist!")
    print("Please run create_embeddings.py first to create the database.")
    exit(1)

# Initialize the HuggingFace embeddings - must match what was used in creation
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load the existing vector store
collection_name = "microsoft_annual_report_2022"
try:
    vectordb = Chroma.from_documents(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    print(f"Connected to collection '{collection_name}'")
except Exception as e:
    print(f"Error connecting to collection: {e}")
    exit(1)

# === Step 2: Query Functions ===
def retrieve_documents(query, n_results=5):
    """Retrieve relevant documents for a given query"""
    docs = vectordb.similarity_search(query, k=n_results)
    return [doc.page_content for doc in docs]

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

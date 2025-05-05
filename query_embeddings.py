#!/usr/bin/env python
# coding: utf-8

# Query Embeddings with Chroma and RAG using Langchain

import os
from helper_utils import word_wrap
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# Load environment variables
_ = load_dotenv(find_dotenv())
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# === Step 1: Connect to Persistent ChromaDB using Langchain ===
persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
print(f"Connecting to ChromaDB at: {persist_directory}")
def retrieve_documents(query, n_results=5, max_return=5):
    """Retrieve relevant documents for a given query using multiple query expansion"""
    # Generate augmented queries
    augmented_queries = augment_query_generated(query)
    
    # Create a list of queries including the original
    queries = [query] + augmented_queries
    
    print("Using the following queries for retrieval:")
    for i, q in enumerate(queries):
        print(f"Query {i+1}: {word_wrap(q)}")
    print('-'*50)
    
    # Get results for all queries
    results = []
    for q in queries:
        docs = vectordb.similarity_search(q, k=n_results)
        results.append([doc.page_content for doc in docs])
    
    # Deduplicate the retrieved documents
    unique_documents = list(set(doc for documents in results for doc in documents))
   
    # Create pairs for cross-encoder scoring
    pairs = []
    for doc in unique_documents:
        pairs.append([query, doc])

    # Calculate cross-encoder scores
    scores = cross_encoder.predict(pairs)

    print("Cross-encoder scores:")
    for i, score in enumerate(scores):
        print(f"Document {i}: {score:.4f}")

    # Get indices sorted by score in descending order
    ranked_indices = np.argsort(scores)[::-1]
    
    print("Reranked order:")
    for i, idx in enumerate(ranked_indices):
        print(f"Rank {i+1}: Document index {idx} (Score: {scores[idx]:.4f})")

    # Rerank the documents and limit to max_return
    reranked_documents = [unique_documents[i] for i in ranked_indices[:max_return]]
    print(f"Returning top {max_return} documents based on reranking.")
    # display the reranked documents
    print("\n=== Reranked Documents ===\n")
    for i, doc in enumerate(reranked_documents):
        print(f"Document {i+1}:")
        print(word_wrap(doc))
        print("\n" + "-"*50 + "\n")
    # Display results for each query
    for i, documents in enumerate(results):
        print(f"Results for Query: {word_wrap(queries[i][:100])}...")
        print('')
        print("Retrieved documents:")
        for j, doc in enumerate(documents):
            print(f"Document {j+1}:")
            print(word_wrap(doc[:200]) + "...")
            print('')
        print('-'*100)
    
    return reranked_documents
    """Retrieve relevant documents for a given query using multiple query expansion"""
    # Generate augmented queries
    augmented_queries = augment_query_generated(query)
    
    # Create a list of queries including the original
    queries = [query] + augmented_queries
    
    print("Using the following queries for retrieval:")
    for i, q in enumerate(queries):
        print(f"Query {i+1}: {word_wrap(q)}")
    print('-'*50)
    
    # Get results for all queries
    results = []
    for q in queries:
        docs = vectordb.similarity_search(q, k=n_results)
        results.append([doc.page_content for doc in docs])
    
    # Deduplicate the retrieved documents
    unique_documents = list(set(doc for documents in results for doc in documents))
   
    # Create pairs for cross-encoder scoring
    pairs = []
    for doc in unique_documents:
        pairs.append([query, doc])

    # Calculate cross-encoder scores
    scores = cross_encoder.predict(pairs)

    print("Cross-encoder scores:")
    for i, score in enumerate(scores):
        print(f"Document {i}: {score:.4f}")

    # Get indices sorted by score in descending order
    ranked_indices = np.argsort(scores)[::-1]
    
    print("Reranked order:")
    for i, idx in enumerate(ranked_indices):
        print(f"Rank {i+1}: Document index {idx} (Score: {scores[idx]:.4f})")

    # Rerank the documents
    reranked_documents = [unique_documents[i] for i in ranked_indices]
    
    # Display results for each query
    for i, documents in enumerate(results):
        print(f"Results for Query: {word_wrap(queries[i][:100])}...")
        print('')
        print("Retrieved documents:")
        for j, doc in enumerate(documents):
            print(f"Document {j+1}:")
            print(word_wrap(doc[:200]) + "...")
            print('')
        print('-'*100)
    
    return reranked_documents
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
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    print(f"Connected to collection '{collection_name}'")
except Exception as e:
    print(f"Error connecting to collection: {e}")
    exit(1)

# === Step 2: Query Functions ===
def augment_query_generated(query, model="gemini-2.0-flash", num_variations=2):
    """Generate augmented queries using Gemini to help with retrieval"""
    system_prompt = """You are a helpful expert financial research assistant. 
    
    Given a user question about financial information in annual reports, please generate {num_variations} alternative ways to ask this question.
    
    These alternative formulations should:
    1. Capture different potential ways the information might be described in an annual report
    2. Use domain-specific financial terminology that might appear in the document
    3. Consider different aspects or angles of the original question
    4. Include potential keywords that would help in document retrieval
    
    Format your response as a numbered list with each query on a separate line.
    """
    
    gemini = genai.GenerativeModel(model)
    response = gemini.generate_content(
        [
            system_prompt.format(num_variations=num_variations),
            query
        ]
    )
    
    content = response.text
    # Parse the numbered list into separate queries
    augmented_queries = []
    for line in content.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() and '. ' in line):
            augmented_queries.append(line.split('. ', 1)[1])
    
    # Ensure we have at least one augmented query
    if not augmented_queries:
        augmented_queries = [content]
        
    return augmented_queries

def retrieve_documents(query, n_results=5):
    """Retrieve relevant documents for a given query using multiple query expansion"""
    # Generate augmented queries
    augmented_queries = augment_query_generated(query)
    
    # Create a list of queries including the original
    queries = [query] + augmented_queries
    
    print("Using the following queries for retrieval:")
    for i, q in enumerate(queries):
        print(f"Query {i+1}: {word_wrap(q)}")
    print('-'*50)
    
    # Get results for all queries
    results = []
    for q in queries:
        docs = vectordb.similarity_search(q, k=n_results)
        results.append([doc.page_content for doc in docs])
    
    # Deduplicate the retrieved documents
    unique_documents = set()
    for documents in results:
        for document in documents:
            unique_documents.add(document)
   
    pairs = []
    for doc in unique_documents:
        pairs.append([query, doc])


    # In[ ]:


    scores = cross_encoder.predict(pairs)


    # In[ ]:


    print("Scores:")
    for score in scores:
        print(score)


    # In[ ]:


    print("New Ordering:")
    for o in np.argsort(scores)[::-1]:
        print(o)


    # Display results for each query
    for i, documents in enumerate(results):
        print(f"Results for Query: {word_wrap(queries[i][:100])}...")
        print('')
        print("Retrieved documents:")
        for j, doc in enumerate(documents):
            print(f"Document {j+1}:")
            print(word_wrap(doc[:200]) + "...")
            print('')
        print('-'*100)
    
    return list(unique_documents)

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

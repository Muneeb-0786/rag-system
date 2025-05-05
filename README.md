# RAG (Retrieval-Augmented Generation) System

This project implements a Retrieval-Augmented Generation system for querying information from documents using vector embeddings and large language models.

## Overview

The RAG system allows users to:
- Create vector embeddings from documents
- Store these embeddings in a ChromaDB vector database
- Query the database with natural language questions
- Retrieve relevant document sections
- Generate accurate, document-grounded responses using Gemini API

## Project Structure

- `query_embeddings.py` - Main script for querying the vector database
- `helper_utils.py` - Utility functions for text formatting and processing
- `requirements.txt` - Required Python dependencies
- `chroma_db/` - Directory where vector embeddings are stored

## Setup Instructions

1. Clone this repository:

```bash
git clone <repository-url>
cd rag
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run the embedding creation script (if not already done):

```bash
python create_embeddings.py
```

5. Query the system:

```bash
python query_embeddings.py
```

## Usage

After starting `query_embeddings.py`, you'll be prompted to enter questions. The system will:

1. Convert your question to embeddings
2. Search the vector database for relevant document sections
3. Display the retrieved document pieces
4. Generate a comprehensive answer based on the retrieved information

Type 'exit' to quit the program.

## Requirements

- Python 3.8+
- ChromaDB
- Sentence Transformers
- Google Generative AI (Gemini)
- Other dependencies listed in requirements.txt

## License

[Add your license information here]

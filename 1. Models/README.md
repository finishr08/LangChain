# Models Directory

This directory contains examples and implementations of various Chat and Embedding models using LangChain.

## Directory Structure

### 1. ChatModels

Contains implementations of Chat Models.

- **`geminiModel.py`**: Implementation/Example using Google's Gemini model.
- **`huggingFace.py`**: Implementation/Example using Hugging Face models (e.g., TinyLlama).

### 2. EmbeddingModels

Contains implementations of Embedding Models for generating text embeddings.

- **`embedded_gemini.py`**: Generating embeddings using Google's Gemini.
- **`embedded_HF_documents.py`**: Generating embeddings for documents using Hugging Face.
- **`embedded_HF_query.py`**: Generating embeddings for queries using Hugging Face.

## Setup

Ensure you have the necessary environment variables set up in your `.env` file (e.g., `GOOGLE_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`).

## Usage

You can run individual scripts to test the models:

```bash
python "1.ChatModels/huggingFace.py"
python "2.EmbeddingModels/embedded_gemini.py"
# ... and so on
```

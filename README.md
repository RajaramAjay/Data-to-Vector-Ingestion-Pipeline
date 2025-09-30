# Data-to-Vector Ingestion Pipeline

## Overview
The Data-to-Vector Ingestion Pipeline is an enterprise-grade solution designed to empower businesses to build scalable vector databases from diverse data sources, including documents (PDFs, DOCX, TXT, CSV) and webpages. By transforming raw data into semantic embeddings stored in a FAISS vector database, it enables efficient, AI-driven semantic search and retrieval. The pipeline supports document chunking, deduplication, web scraping, and planned OCR integration for scanned PDFs, with robust error handling, logging, and performance optimization. Deployed as a Flask-based web application with a user-friendly front-end, it is ready for enterprise environments, including IIS deployment.

## Key Features
- **Multi-Source Data Ingestion**: Seamlessly process documents (PDFs, DOCX, TXT, CSV) and webpages into a unified vector database.
- **Semantic Embeddings**: Generate high-quality embeddings using Sentence Transformers (e.g., all-MiniLM-L6-v2) with GPU acceleration support.
- **FAISS Vector Storage**: Store and query embeddings efficiently with FAISS, supporting incremental updates and deduplication.
- **Document Chunking & Deduplication**: Split documents into semantically coherent chunks with SHA-256 hashing to eliminate duplicates.
- **Web Scraping**: Extract and process webpage content with configurable crawling limits and robust HTML parsing.
- **Planned OCR Integration**: Framework for Azure OCR to extract text from scanned PDFs (currently commented out for cost/privacy).
- **Robust Error Handling**: Comprehensive logging and error recovery for reliable operation in enterprise settings.
- **IIS Deployment**: Configuration support for deployment on Internet Information Services (IIS) via web.config.
- **Performance Optimization**: Multiprocessing for file loading and GPU-accelerated embedding generation for large-scale datasets.

## Use Cases
- **Enterprise Search**: Enable semantic search across internal documents and external web content for knowledge management.
- **Data Analytics**: Aggregate and index diverse data sources for AI-driven insights and decision-making.
- **Content Management**: Build intelligent repositories for documents and web data with fast, meaning-based retrieval.
- **AI Application Development**: Provide a foundation for RAG (Retrieval-Augmented Generation) systems and other AI workflows.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/data-to-vector-ingestion-pipeline.git
   cd data-to-vector-ingestion-pipeline
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure settings in `config/config.toml` (e.g., model paths, FAISS directory).
4. Run the Flask API:
   ```bash
   python main.py
   ```

## Directory Structure
- `main.py`: Flask API and application entry point.
- `src/`: Core logic for processing, embedding, and storage.
  - `azure_ocr.py`: Planned OCR for scanned PDFs.
  - `doc_chunker.py`: Document chunking and deduplication.
  - `embeddings.py`: Embedding generation with Sentence Transformers.
  - `file_processor.py`: File loading and processing.
  - `input_validation.py`: Input validation for files and URLs.
  - `web_processor.py`: Web page scraping and processing.
  - `has_images.py`: Detection of image-heavy PDFs.
  - `pipeline.py`: Orchestrates ingestion pipeline.
  - `interfaces.py`: API and data interfaces.
  - `faiss_store.py`: FAISS vector storage.
- `config/`: Application settings (e.g., `config.toml`).
- `templates/`: Front-end interface (`index.html`).
- `database/faiss/`: FAISS vector storage.
- `models/sentence_transformers/`: Pre-trained Sentence Transformer models.
- `logs/`: Log files for debugging.
- `logger_setup.py`: Centralized logging configuration.
- `web.config`: IIS deployment configuration for Windows server environments.

## Requirements
- Python 3.8+
- Flask, Sentence Transformers, FAISS, and other dependencies (see `requirements.txt`).

## Next Steps
After ingesting data into the FAISS vector database using this pipeline, you can perform semantic search and generate AI-driven responses with the [AI Search Pipeline](https://github.com/YourUsername/YourRepoName](https://github.com/RajaramAjay/AI-Search-Pipeline). The AI Search Pipeline uses the FAISS index created here to retrieve relevant documents and produce context-aware answers via an LLM.

Usage: Configure the AI Search Pipeline to point to the FAISS index (database/faiss) and query it via its Flask API or front-end UI.

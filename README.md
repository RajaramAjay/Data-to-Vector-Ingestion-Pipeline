# Data-to-Vector Ingestion Pipeline

## Overview
Data-to-Vector Ingestion Pipeline is a robust, enterprise-grade solution for building vector databases from diverse data sources, including documents (e.g., PDFs) and webpages. It enables efficient semantic search and retrieval through a streamlined pipeline featuring OCR, document chunking, web scraping, embedding generation with Sentence Transformers, and FAISS vector storage.

## Key Features
- **Document and Web Ingestion**: Process PDFs, scanned documents, and webpages seamlessly.
- **Embedding Generation**: Create semantic embeddings using Sentence Transformers (e.g., all-MiniLM-L6-v2).
- **Vector Storage**: Store and query embeddings efficiently with FAISS.
- **Pipeline Orchestration**: Automate data ingestion, processing, and storage.
- **Web Scraping**: Extract and process text from webpages for semantic search.
- **OCR Support**: Planned support for extracting text from scanned PDFs using Azure OCR.
- **Image Detection**: Identify image-heavy PDFs for optimized processing.
- **Input Validation**: Ensure robust validation for files and URLs.
- **Logging**: Centralized logging for monitoring and debugging.

## Use Cases
- Enterprise search systems for internal documents and web content.
- Knowledge management platforms requiring semantic retrieval.
- Data aggregation for AI-driven applications and analytics.

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

## Requirements
- Python 3.8+
- Flask, Sentence Transformers, FAISS, and other dependencies (see `requirements.txt`).

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. See `CONTRIBUTING.md` (to be added) for guidelines.

## License
MIT License
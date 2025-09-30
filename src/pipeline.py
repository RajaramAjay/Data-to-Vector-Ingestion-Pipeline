#src/pipeline.py
import time
from typing import Dict, Any, Tuple, List
from langchain.schema.document import Document

from src.utils import get_logger
# Get logger through the utility function
logger = get_logger(__name__)

# Import components
from src.input_validation import URLValidator, PathValidator
from src.file_processor import FileProcessor
from src.web_processor import WebProcessor
from src.doc_chunker import Chunker
from src.faiss_store import VectorStoreFAISS


class Pipeline:
    """
    Orchestrates the full document ingestion pipeline:
    - Validates input (URL or local path)
    - Loads & parses documents
    - Chunks content
    - Stores in FAISS vector store
    """
    def __init__(
        self,
        url_validator: URLValidator = None,
        path_validator: PathValidator = None,
        file_processor: FileProcessor = None,
        web_processor: WebProcessor = None,
        chunker: Chunker = None,
        vector_store: VectorStoreFAISS = None
    ):
        # Initialize components or use defaults
        self.url_validator = url_validator or URLValidator()
        self.path_validator = path_validator or PathValidator()
        self.file_processor = file_processor or FileProcessor()
        self.web_processor = web_processor or WebProcessor()
        self.chunker = chunker or Chunker()
        self.vector_store = vector_store or VectorStoreFAISS()

    def process(self, user_data: str) -> Tuple[List[Document], str, Dict[str, Any], float, int]:
        start_time = time.time()
        logger.info(f"Received input: {user_data}")
        
        try:
            # Step 1: Validate input type
            valid_url, _ = self.url_validator.validate(user_data)
            if valid_url:
                documents, num_files = self.web_processor.process(user_data)
                source_type = "website"
            else:
                valid_path, msg = self.path_validator.validate(user_data)
                if not valid_path:
                    raise ValueError(msg)
                documents, num_files = self.file_processor.process(user_data)
                source_type = "filepath"
            
            # Step 2: Check if documents are found
            if not documents:
                raise ValueError("No documents found in input.")
            
            # Step 3: Chunk documents and store in FAISS
            chunks = self.chunker.split_documents(documents)
            _, stats = self.vector_store.add_documents(chunks)
            
            processing_time = round(time.time() - start_time, 2)
            logger.info(f"Pipeline processing completed in {processing_time}s for {num_files} files")
            
            return documents, source_type, stats, processing_time, num_files
        
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise

# Create a factory function to return a configured instance
def create_pipeline() -> Pipeline:
    """
    Factory function to create a fully configured pipeline instance.
    Makes it easy to get a ready-to-use pipeline with default components.
    """
    return Pipeline()
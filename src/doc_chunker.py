#src/doc_chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import hashlib
from typing import List, Dict
from src.interfaces import DocumentChunker
import os, toml, sys

from src.utils import get_logger
# Get logger through the utility function
logger = get_logger(__name__)

# Load configuration
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
config = toml.load(config_path)

class Chunker(DocumentChunker):
    def __init__(self):
        # Initialize chunker with config values
        self.chunk_size = config['chunking']['chunk_size']
        self.chunk_overlap = config['chunking']['chunk_overlap']
        logger.info(f"Chunker initialized with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    @staticmethod
    def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
        """Adds unique chunk IDs and content hashes to documents, deduplicates identical chunks."""
        logger.info("Entered calculate_chunk_ids")
        try:
            last_page_id = None
            current_chunk_index = 0
            content_hash_map: Dict[str, str] = {}  # Store content hash to avoid duplicates
            deduplicated_chunks: List[Document] = []

            # Iterate over chunks to assign IDs and remove duplicates
            for chunk in chunks:
                source = chunk.metadata.get("source", "unknown_source")
                page = chunk.metadata.get("page", "0")
                current_page_id = f"{source}:{page}"

                # Increment chunk index for the same page
                if current_page_id == last_page_id:
                    current_chunk_index += 1
                else:
                    current_chunk_index = 0

                chunk_id = f"{current_page_id}:{current_chunk_index}"
                last_page_id = current_page_id

                # Generate content hash
                content = chunk.page_content.encode('utf-8')
                content_hash = hashlib.sha256(content).hexdigest()

                # Add metadata for chunk ID and content hash
                chunk.metadata["id"] = chunk_id
                chunk.metadata["content_hash"] = content_hash

                # Deduplicate based on content hash
                if content_hash not in content_hash_map:
                    content_hash_map[content_hash] = chunk_id
                    deduplicated_chunks.append(chunk)

            logger.info(f"Finished calculate_chunk_ids: {len(deduplicated_chunks)} unique chunks")
            return deduplicated_chunks

        except Exception as e:
            logger.error(f"Error in calculate_chunk_ids: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into chunks and assigns IDs while deduplicating."""
        logger.info("Entered split_documents")
        try:
            # Initialize the text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )

            # Split documents into chunks
            doc_chunks = splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(doc_chunks)} chunks")

            # Deduplicate chunks and assign chunk IDs
            final_chunks = self.calculate_chunk_ids(doc_chunks)
            logger.info(f"Final chunk count after deduplication: {len(final_chunks)}")
            # print(final_chunks)
            return final_chunks

        except Exception as e:
            logger.error(f"Error in split_documents: {e}")
            raise


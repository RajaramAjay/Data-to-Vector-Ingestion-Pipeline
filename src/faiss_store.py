#src/faiss_store.py
import os
import uuid
import sys
import toml
from typing import Dict, Tuple, List
from pathlib import Path
from src.embeddings import get_embedding_function
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from src.interfaces import VectorStore

# Logger setup
from src.utils import get_logger
# Get logger through the utility function
logger = get_logger(__name__)

# Load config
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
config = toml.load(config_path)

class VectorStoreFAISS(VectorStore):
    def __init__(self):
        self.FAISS_PATH = config['faiss']['path']
        self.embedding_function = get_embedding_function()

    def create_new_faiss_index(self, chunks: List[Document]) -> Tuple[FAISS, Dict[str, int]]:
        """
        Creates a new FAISS index and saves it to local path.
        """
        try:
            logger.info("Creating new FAISS index...")

            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.metadata["id"] for chunk in chunks]

            db = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_function,
                metadatas=metadatas,
                ids=ids
            )

            try:
                db.save_local(self.FAISS_PATH)
                logger.info(f"FAISS index saved to: {self.FAISS_PATH}")
            except Exception:
                temp_path = f"faiss_temp_{uuid.uuid4().hex[:8]}"
                db.save_local(temp_path)
                logger.warning(f"Failed to save FAISS to default path, saved to: {temp_path}")

            doc_count = len(db.docstore._dict) if hasattr(db, 'docstore') else len(chunks)

            stats = {
                "InitialDocChunk_count": 0,
                "AddedDocChunk_count": len(chunks),
                "TotalDocChunk_count": doc_count
            }

            return db, stats

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}")
            raise RuntimeError(f"Failed to create FAISS index: {str(e)}")
        
    def add_documents(self, chunks: List[Document]) -> Tuple[FAISS, Dict[str, int]]:
        """
        Adds documents to FAISS index with deduplication.
        Preserves existing data even on error.
        """
        try:
            index_exists = os.path.exists(self.FAISS_PATH) and os.path.isdir(self.FAISS_PATH)
            if index_exists:
                try:
                    logger.info("Loading existing FAISS index...")
                    db = FAISS.load_local(
                        self.FAISS_PATH,
                        self.embedding_function,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Get existing IDs and content hashes
                    existing_ids = set()
                    existing_hashes = {}
                    if hasattr(db, 'docstore') and hasattr(db.docstore, '_dict'):
                        for doc_id, doc in db.docstore._dict.items():
                            existing_ids.add(doc_id)
                            if doc.metadata and 'content_hash' in doc.metadata:
                                existing_hashes[doc.metadata['content_hash']] = doc_id
                    
                    initial_doc_count = len(existing_ids)
                    
                    # Filter out chunks with duplicate content hashes
                    new_chunks = [
                        chunk for chunk in chunks
                        if chunk.metadata.get("content_hash") not in existing_hashes
                    ]
                    
                    # Ensure IDs don't conflict with existing ones
                    for chunk in new_chunks:
                        if chunk.metadata["id"] in existing_ids:
                            # Generate a new unique ID if there's a conflict
                            new_id = f"{chunk.metadata['id']}_{uuid.uuid4().hex[:8]}"
                            logger.warning(f"ID conflict detected. Changed ID from {chunk.metadata['id']} to {new_id}")
                            chunk.metadata["id"] = new_id
                    
                    if new_chunks:
                        texts = [chunk.page_content for chunk in new_chunks]
                        metadatas = [chunk.metadata for chunk in new_chunks]
                        ids = [chunk.metadata["id"] for chunk in new_chunks]
                        
                        try:
                            db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                            db.save_local(self.FAISS_PATH)
                            logger.info(f"Added {len(new_chunks)} new chunks to FAISS index.")
                        except Exception as add_error:
                            logger.error(f"Failed to add new texts: {str(add_error)}")
                            # Don't create a new index, just return the existing one with stats
                            
                    final_doc_count = len(db.docstore._dict) if hasattr(db, 'docstore') else initial_doc_count
                    stats = {
                        "InitialDocChunk_count": initial_doc_count,
                        "AddedDocChunk_count": len(new_chunks),
                        "TotalDocChunk_count": final_doc_count
                    }
                    return db, stats
                    
                except Exception as e:
                    logger.error(f"Failed to load existing FAISS index: {str(e)}")
                    # Instead of creating a new index, try to backup and recover
                    backup_path = f"{self.FAISS_PATH}_backup_{uuid.uuid4().hex[:8]}"
                    try:
                        import shutil
                        shutil.copytree(self.FAISS_PATH, backup_path)
                        logger.info(f"Created backup of problematic index at: {backup_path}")
                        logger.info("Creating new index with current chunks only...")
                        return self.create_new_faiss_index(chunks)
                    except Exception as backup_error:
                        logger.error(f"Failed to create backup: {str(backup_error)}")
                        # As a last resort, create new index
                        return self.create_new_faiss_index(chunks)
            else:
                logger.info("No existing FAISS index found. Creating new one.")
                return self.create_new_faiss_index(chunks)
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {str(e)}")
            raise RuntimeError(f"Failed to add documents to FAISS: {str(e)}")

   
#src/file_processor.py
import os, sys
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader,
)
# from azure_loder import azurepdfloader
import multiprocessing
from src.interfaces import DocumentProcessor
from src.has_images import pdf_contains_images
from src.utils import get_logger
from src.azure_ocr import AzurePDFLoader
# Get logger through the utility function
logger = get_logger(__name__)

class FileProcessor(DocumentProcessor):
    def check_type(self, file_path: str) -> List[Document]:
        """Detects file type and loads the document using the appropriate loader."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
           
            file_extension = os.path.splitext(file_path)[1].lower()
                #if he pdf has images in it 
                # check for images  
                
            if file_extension == ".pdf":
                # has_images = pdf_contains_images(file_path)
                # if has_images:
                #     loader = AzurePDFLoader(pdf_path=file_path)
                # else:
                loader = PyPDFLoader(file_path)
    
            elif file_extension in {".docx", ".doc"}:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
           
            documents = loader.load()
           
            # Add metadata to track document source
            for doc in documents:
                doc.metadata["source"] = file_path
           
            return documents
       
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    def _process_file(self, file_path: str) -> List[Document]:
        """Helper function to process a single file for multiprocessing."""
        try:
            return self.check_type(file_path)
        except Exception as e:
            logger.warning(f"Skipping {file_path}: {e}")
            return []

    def process(self, input_path: str) -> List[Document]:
        """Processes a single file or all supported files in a given directory using multiprocessing."""
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Path not found: {input_path}")
            
            file_paths = []
            if os.path.isdir(input_path):  # If input is a folder, process all files
                for root, _, files in os.walk(input_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_paths.append(file_path)
            else:  # If input is a single file
                file_paths = [input_path]
            
            # Use multiprocessing to process files in parallel
            num_processes = max(1, multiprocessing.cpu_count() - 52)
            if num_processes <= 1 or len(file_paths) <= 1:
                # For single file or single process, don't use multiprocessing
                all_documents = []
                for file_path in file_paths:
                    try:
                        docs = self.check_type(file_path)
                        all_documents.extend(docs)
                    except Exception as e:
                        logger.warning(f"Skipping {file_path}: {e}")
            else:
                # Use multiprocessing for multiple files
                logger.info(f"Processing {len(file_paths)} files using {num_processes} processes")
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.map(self._process_file, file_paths)
                
                # Flatten the list of lists into a single list of documents
                all_documents = [doc for result in results for doc in result]
            
            return all_documents, len(file_paths)
       
        except Exception as e:
            logger.error(f"Error loading documents from path {input_path}: {e}")
            raise

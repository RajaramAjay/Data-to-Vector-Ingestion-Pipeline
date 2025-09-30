#src/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from langchain.schema.document import Document
# from langchain.schema.document import Document
# from langchain_core.embeddings import Embeddings

class Validator(ABC):
    @abstractmethod
    def validate(self, input_data: str):
        """Validates the input data and returns (is_valid, error_message or None)."""
        pass

class DocumentProcessor(ABC):
    @abstractmethod
    def process(self, source: str) :
        """
        Processes the given source (a file path or a URL) and returns a list of documents.
        """
        pass

class DocumentChunker(ABC):
    @abstractmethod
    def split_documents(self, documents: List[Document]) :
        """
        Splits documents into chunks using recursive character text splitting        """
        pass

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: List[Document]) -> Tuple[any, Dict[str, int]]:
        pass


#src/azure_ocr.py
import os
from src.utils import get_logger
logger = get_logger(__name__)

import toml
import io
import time
import random
import numpy as np
import concurrent.futures
from PIL import Image
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from msrest.exceptions import HttpOperationError
from langchain_core.documents import Document
from abc import ABC, abstractmethod

# Load config
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
config = toml.load(config_path)

class AzureDocumentLoaders(ABC):
    @abstractmethod
    def load(self):
        pass

class AzurePDFLoader(AzureDocumentLoaders):
    
    def __init__(self, pdf_path=None):
        self.endpoint = config['azure_ocr']['endpoint']
        self.subscription_key = config['azure_ocr']['subscription_key']
        self.poppler_path = config['azure_ocr']['poppler_path']
        self.pdf_path = pdf_path
        self.client = ComputerVisionClient(
            endpoint=self.endpoint,
            credentials=CognitiveServicesCredentials(self.subscription_key)
        )
        # Rate limiting settings
        self.max_retries = 5
        self.base_delay = 2  # seconds
        self.max_delay = 60  # seconds
        self.jitter = 0.1   # 10% jitter
    
    def convert_pdf_to_images(self, pdf_path, dpi=300, batch_size=4):
        """Convert PDF to images with parallel processing for large PDFs"""
        logger.info(f"Converting PDF to images: {pdf_path}")
        start_time = time.time()
        try:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=self.poppler_path,
                thread_count=os.cpu_count() or 4
            )
            for idx, img in enumerate(images, 1):
                logger.debug(f"Page {idx} image size: {img.size}")
            logger.info(f"Converted {len(images)} pages in {time.time() - start_time:.2f} seconds")
            return images
        except Exception as e:
            logger.error(f"Error in PDF conversion: {str(e)}")
            raise

    def retry_with_backoff(self, func, *args, **kwargs):
        """Execute a function with exponential backoff retry logic."""
        retry_count = 0
        while True:
            try:
                return func(*args, **kwargs)
            except HttpOperationError as e:
                if e.response.status_code == 429 or (e.response.status_code >= 500 and e.response.status_code < 600):
                    retry_count += 1
                    if retry_count > self.max_retries:
                        logger.error(f"Maximum retries ({self.max_retries}) exceeded. Giving up.")
                        raise
                    
                    # Parse retry-after header if available
                    retry_after = 1  # default 1 second
                    if "Retry-After" in e.response.headers:
                        try:
                            retry_after = int(e.response.headers["Retry-After"])
                        except (ValueError, TypeError):
                            pass
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.max_delay,
                        max(retry_after, self.base_delay * (2 ** (retry_count - 1)))
                    )
                    
                    # Add jitter
                    jitter_amount = delay * self.jitter
                    delay = delay + random.uniform(-jitter_amount, jitter_amount)
                    
                    logger.warning(f"Rate limit hit (429). Retrying in {delay:.2f} seconds (attempt {retry_count}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    # For other types of errors, raise immediately
                    logger.error(f"API error: {e.response.status_code} - {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise

    def analyze_image_features(self, image_bytes_io):
        logger.debug("Calling Azure CV API for visual analysis")
        start_time = time.time()
        features = [
            VisualFeatureTypes.description,
            VisualFeatureTypes.tags,
            VisualFeatureTypes.objects
        ]
        image_bytes_io.seek(0)
        
        # Use retry logic for API call
        result = self.retry_with_backoff(
            self.client.analyze_image_in_stream,
            image_bytes_io,
            visual_features=features
        )
        
        logger.debug(f"Visual analysis completed in {time.time() - start_time:.2f} seconds")
        return result
        
    def perform_ocr(self, image_bytes_io):
        logger.debug("Performing OCR using Azure Read API")
        start_time = time.time()
        image_bytes_io.seek(0)
        
        # Use retry logic for initial API call
        response = self.retry_with_backoff(
            self.client.read_in_stream,
            image_bytes_io,
            raw=True
        )
        
        operation_location = response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        polling_interval = 1
        max_polling_interval = 5
        attempts = 0
        
        while True:
            # Use retry logic for polling
            result = self.retry_with_backoff(
                self.client.get_read_result,
                operation_id
            )
            
            if result.status.lower() not in ['running', 'notstarted']:
                break
                
            time.sleep(min(polling_interval, max_polling_interval))
            polling_interval *= 1.5
            attempts += 1
            
        logger.debug(f"OCR completed in {attempts} attempts and {time.time() - start_time:.2f} seconds with status: {result.status}")
        return result

    def detect_tables(self, ocr_lines, tolerance=10):
        logger.debug("Detecting tabular structures from OCR lines")
        if not ocr_lines:
            logger.warning("No OCR lines provided for table detection")
            return []
        lines_by_y = sorted(ocr_lines, key=lambda line: line.bounding_box[1])
        rows = []
        current_row = [lines_by_y[0]]
        prev_y = lines_by_y[0].bounding_box[1]
        for line in lines_by_y[1:]:
            current_y = line.bounding_box[1]
            if abs(current_y - prev_y) <= tolerance:
                current_row.append(line)
            else:
                current_row.sort(key=lambda l: l.bounding_box[0])
                rows.append(current_row)
                current_row = [line]
                prev_y = current_y
        if current_row:
            current_row.sort(key=lambda l: l.bounding_box[0])
            rows.append(current_row)
        tables = []
        current_table = []
        for row in rows:
            if len(row) > 1:
                current_table.append(row)
            elif current_table:
                if len(current_table) > 1:
                    tables.append(current_table)
                current_table = []
        if current_table and len(current_table) > 1:
            tables.append(current_table)
        logger.debug(f"Detected {len(tables)} table(s)")
        return tables
        
    def process_single_page(self, page_num, image):
        logger.info(f"Processing Page {page_num}")
        page_start_time = time.time()
        try:
            analysis_bytes_io = io.BytesIO()
            ocr_bytes_io = io.BytesIO()
            image.save(analysis_bytes_io, format='PNG')
            image.save(ocr_bytes_io, format='PNG')
            
            # Process analysis and OCR serially to avoid too many concurrent requests
            analysis = self.analyze_image_features(analysis_bytes_io)
            ocr_result = self.perform_ocr(ocr_bytes_io)
            
            if analysis.description and analysis.description.captions:
                for caption in analysis.description.captions:
                    logger.debug(f"Caption: {caption.text} (confidence: {caption.confidence:.2f})")
            if ocr_result.status.lower() == 'succeeded':
                all_lines = [line for page in ocr_result.analyze_result.read_results for line in page.lines]
                logger.debug(f"Extracted {len(all_lines)} lines of text")
                tables = self.detect_tables(all_lines)
                for idx, table in enumerate(tables, 1):
                    logger.info(f"Table {idx} on Page {page_num}")
            logger.info(f"Page {page_num} processed in {time.time() - page_start_time:.2f} seconds")
            return page_num, analysis, ocr_result
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
            raise

    def process_pdf_parallel(self, pdf_path, max_workers=None):
        logger.info(f"Starting parallel PDF processing: {pdf_path}")
        total_start_time = time.time()
        
        # Adjust max_workers based on API limits
        # S1 tier typically allows 10 transactions per second
        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 4)  # Limit concurrent requests
            
        try:
            images = self.convert_pdf_to_images(pdf_path, dpi=300)
            all_results = {}
            # Process smaller batches to avoid rate limiting
            batch_size = min(max_workers, 4)  # Reduce batch size
            
            for i in range(0, len(images), batch_size):
                batch_start_time = time.time()
                batch_images = images[i:i+batch_size]
                batch_indices = range(i+1, i+len(batch_images)+1)
                
                # Add delay between batches to avoid rate limiting
                if i > 0:
                    time.sleep(1)  # Wait 1 second between batches
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_images), thread_name_prefix=f"PageBatch-{i}") as executor:
                    futures = {
                        executor.submit(self.process_single_page, idx, img): idx
                        for idx, img in zip(batch_indices, batch_images)
                    }
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            page_num, analysis, ocr_result = future.result()
                            all_results[page_num] = {
                                "analysis": analysis,
                                "ocr_result": ocr_result
                            }
                        except Exception as e:
                            page_num = futures[future]
                            logger.error(f"Error processing page {page_num}: {str(e)}")
                
                logger.info(f"Batch {i//batch_size + 1} processed in {time.time() - batch_start_time:.2f} seconds")
            
            logger.info(f"Successfully processed {len(all_results)} of {len(images)} pages")
            return all_results
        except FileNotFoundError:
            logger.error(f"❌ File not found: {pdf_path}")
        except PermissionError:
            logger.error(f"❌ Permission denied: {pdf_path}")
        except PDFInfoNotInstalledError:
            logger.error("❌ Poppler not found. Please install and add to PATH.")
        except Exception as e:
            logger.exception(f"❌ Unexpected error: {str(e)}")
        finally:
            logger.info(f"Total parallel processing time: {time.time() - total_start_time:.2f} seconds")
        return {}

    def convert_to_langchain_documents(self, all_results, pdf_path):
        """Convert OCR results to LangChain Document objects."""
        documents = []
        file_name = os.path.basename(pdf_path)
        
        for page_num in sorted(all_results.keys()):
            ocr_result = all_results[page_num]["ocr_result"]
            analysis = all_results[page_num]["analysis"]
            
            # Extract text from OCR results
            if ocr_result.status.lower() == 'succeeded':
                page_text = ""
                for page in ocr_result.analyze_result.read_results:
                    for line in page.lines:
                        page_text += line.text + "\n"
                
                # Extract metadata (e.g., captions, tags)
                metadata = {
                    "source": file_name,
                    "page_number": page_num,
                    "captions": [caption.text for caption in analysis.description.captions] if analysis.description and analysis.description.captions else [],
                    "tags": [tag.name for tag in analysis.tags] if analysis.tags else []
                }
                
                # Create LangChain Document
                doc = Document(
                    page_content=page_text.strip(),
                    metadata=metadata
                )
                documents.append(doc)
                logger.info(f"Created LangChain Document for Page {page_num}")
            else:
                logger.warning(f"OCR failed for Page {page_num}, skipping document creation")
        
        logger.info(f"Created {len(documents)} LangChain Documents")
        return documents

    def try_use_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU acceleration available: {torch.cuda.get_device_name(0)}")
                return True
            else:
                logger.info("GPU not available, using CPU processing")
                return False
        except ImportError:
            logger.info("PyTorch not installed, using CPU processing")
            return False
            
    def load(self, pdf_path=None):
        """Load a PDF and convert it to LangChain Documents."""
        if pdf_path is None:
            pdf_path = self.pdf_path
        
        if pdf_path is None:
            logger.error("PDF path not provided")
            return []
            
        # Check if GPU is available
        self.try_use_gpu()
        
        # Process the PDF
        all_results = self.process_pdf_parallel(pdf_path)
        
        # Convert results to LangChain Documents
        documents = self.convert_to_langchain_documents(all_results, pdf_path)
        
        return documents
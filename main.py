#main.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import uuid
import json
from typing import Dict, Any, Tuple
from multiprocessing import freeze_support
freeze_support()# Ensure compatibility with multiprocessing on Windows
from src.pipeline import create_pipeline
# Internal modules
from logger_setup import setup_logger, start_request_logging, end_request_logging

# Logger setup
logger = setup_logger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)


# Create pipeline instance
pipeline = create_pipeline()

# ------------------------- API Endpoints -------------------------
@app.route('/ingest', methods=['POST'])
def ingest_documents() -> Tuple[Dict[str, Any], int]:
    """Handles ingestion of documents from path or URL."""
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    
    # Start request-specific logging
    log_file = start_request_logging(request_id)
    
    try:
        if not request.is_json:
            logger.warning("Invalid request: Not JSON")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        user_data = data.get("data_input", "").strip()
        
        if not user_data:
            logger.warning("Missing 'data_input' in request")
            return jsonify({"error": "Missing 'data_input' field"}), 400
        
        # Run pipeline - the existing log calls will now go to our request-specific log file
        documents, source_type, stats, processing_time, num_files = pipeline.process(user_data)
        
        # Success response
        response = {
            "Message": "Chunked Documents loaded into FAISS",
            "Data_source": user_data,
            "Source_type": source_type,
            "Index_Status": json.dumps(stats, indent=2),
            "Processing_time_seconds": processing_time,
            "Files_processed": num_files,
            "Log_file": log_file  # Let the client know which log file was used
        }
        
        logger.info(f"Ingestion successful: {num_files} files processed.")
        return jsonify(response), 200
    
    except Exception as e:
        logger.exception(f"Unhandled error during ingestion: {e}")
        return jsonify({
            "status": "error",
            "error": "Internal server error",
            "message": str(e)
        }), 500
    
    finally:
        # Always end request logging
        end_request_logging()

@app.route('/')
def index() -> str:
    """Serves the HTML UI."""
    return render_template('index.html')

# ------------------------- Entry Point -------------------------
if __name__ == '__main__':
    logger.info("Starting Flask app on port 5007...")
    app.run(port=5007, debug=True, use_reloader=True)



# working but not multip...
# logger_setup.py
import os
import logging
from datetime import datetime
import toml
import threading

# Global variables
_log_dir = "logs"
_log_level = logging.INFO
_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_root_logger_configured = False
_thread_local = threading.local()

def _configure_root_logger():
    """Configure the root logger once"""
    global _root_logger_configured, _log_dir, _log_level, _log_format
    
    if not _root_logger_configured:
        try:
            # Load config
            config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), 'config', 'config.toml'))
            config = toml.load(config_path)
            log_config = config.get("logging", {})
            
            # Global log settings
            _log_dir = log_config.get("log_dir", "logs")
            _log_level = getattr(logging, log_config.get("log_level", "INFO").upper(), logging.INFO)
            _log_format = log_config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            
            # Create log directory if it doesn't exist
            os.makedirs(_log_dir, exist_ok=True)
            
            # Configure root logger for console output only
            root_logger = logging.getLogger()
            root_logger.setLevel(_log_level)
            
            # Clear existing handlers
            if root_logger.handlers:
                for handler in root_logger.handlers[:]:
                    root_logger.removeHandler(handler)
            
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(_log_format))
            root_logger.addHandler(console_handler)
            
            _root_logger_configured = True
            
        except Exception as e:
            print(f"Error initializing logger: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO, 
                              format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_logger(name=None):
    """Set up a basic logger that writes to console only"""
    _configure_root_logger()
    return logging.getLogger(name)

def start_request_logging(request_id=None):
    """
    Start logging for a new request.
    Call this at the beginning of an API endpoint or request handler.
    """
    # Generate a timestamp for this request
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate a unique request identifier if none provided
    if not request_id:
        request_id = f"req_{timestamp}"
    
    # Create a new log file for this request
    log_filename = os.path.join(_log_dir, f"log_{timestamp}_{request_id}.log")
    
    # Create a file handler for this request
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(logging.Formatter(_log_format))
    
    # Store in thread local storage
    _thread_local.current_handler = handler
    _thread_local.current_logfile = log_filename
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    return log_filename

def end_request_logging():
    """
    End logging for the current request.
    Call this at the end of an API endpoint or request handler.
    """
    if hasattr(_thread_local, 'current_handler'):
        handler = _thread_local.current_handler
        
        # Remove handler from all loggers
        root_logger = logging.getLogger()
        root_logger.removeHandler(handler)
        
        # Close the handler
        handler.close()
        
        # Clean up thread local storage
        del _thread_local.current_handler
        del _thread_local.current_logfile




#src/input_validation.py
from urllib.parse import urlparse
import os, sys
from typing import Tuple, Optional
from src.interfaces import Validator
from src.utils import get_logger
# Get logger through the utility function
logger = get_logger(__name__)

class URLValidator(Validator):
    def validate(self, input_data: str) -> Tuple[bool, Optional[str]]:
        try:
            result = urlparse(input_data)
            is_valid = all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
            if is_valid:
                logger.debug(f"URL '{input_data}' is valid.")
                return True, None
            else:
                logger.warning(f"URL '{input_data}' is invalid.")
                return False, "Invalid URL"
        except Exception as e:
            logger.error(f"Exception during URL validation: {e}")
            return False, str(e)

class PathValidator(Validator):
    def validate(self, input_data: str) -> Tuple[bool, Optional[str]]:
        try:
            if not os.path.exists(input_data):
                logger.warning(f"Path does not exist: {input_data}")
                return False, "Path does not exist"
            if not os.access(input_data, os.R_OK):
                logger.warning(f"Path is not readable: {input_data}")
                return False, "Path not readable"
            logger.debug(f"Path '{input_data}' is valid and readable.")
            return True, None
        except Exception as e:
            logger.error(f"Exception during path validation: {e}")
            return False, str(e)


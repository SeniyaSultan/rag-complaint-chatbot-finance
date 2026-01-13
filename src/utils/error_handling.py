"""
Production-ready error handling and validation utilities
"""
import logging
import sys
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from functools import wraps
from datetime import datetime
import json

from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/application.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class ErrorResponse(BaseModel):
    """Standardized error response model"""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = datetime.now().isoformat()
    traceback: Optional[str] = None

class ValidationResult(BaseModel):
    """Validation result container"""
    is_valid: bool
    errors: Optional[Dict[str, str]] = None
    cleaned_data: Optional[Dict[str, Any]] = None

class RetryConfig(BaseModel):
    """Configuration for retry logic"""
    max_attempts: int = 3
    wait_multiplier: float = 1.0
    wait_min: float = 1.0
    wait_max: float = 10.0
    retry_on_exceptions: tuple = (Exception,)

class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    def __init__(self, message: str, errors: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.errors = errors or {}
        self.message = message

class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass

class EmbeddingError(Exception):
    """Custom exception for embedding generation failures"""
    pass

def handle_exceptions(
    default_return: Any = None,
    log_level: str = "ERROR",
    include_traceback: bool = False
) -> Callable:
    """
    Decorator for comprehensive exception handling
    
    Args:
        default_return: Value to return on exception
        log_level: Logging level for exceptions
        include_traceback: Whether to include traceback in logs
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (DataValidationError, ValidationError) as e:
                # Handle validation errors specifically
                error_details = {}
                if hasattr(e, 'errors'):
                    error_details = e.errors
                elif hasattr(e, '__cause__') and hasattr(e.__cause__, 'errors'):
                    error_details = e.__cause__.errors
                
                error_msg = f"Validation failed for {func.__name__}: {str(e)}"
                logger.error(f"{error_msg}. Details: {error_details}")
                
                if include_traceback:
                    logger.error(traceback.format_exc())
                
                return ErrorResponse(
                    error_code="VALIDATION_ERROR",
                    message=error_msg,
                    details=error_details,
                    traceback=traceback.format_exc() if include_traceback else None
                )
                
            except (VectorStoreError, EmbeddingError) as e:
                # Handle domain-specific errors
                error_msg = f"Domain error in {func.__name__}: {str(e)}"
                logger.error(error_msg)
                
                if include_traceback:
                    logger.error(traceback.format_exc())
                
                return ErrorResponse(
                    error_code="DOMAIN_ERROR",
                    message=error_msg,
                    details={"function": func.__name__},
                    traceback=traceback.format_exc() if include_traceback else None
                )
                
            except Exception as e:
                # Handle all other exceptions
                error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
                getattr(logger, log_level.lower())(error_msg)
                
                if include_traceback:
                    getattr(logger, log_level.lower())(traceback.format_exc())
                
                return ErrorResponse(
                    error_code="INTERNAL_ERROR",
                    message=error_msg,
                    details={
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    } if log_level == "DEBUG" else None,
                    traceback=traceback.format_exc() if include_traceback else None
                ) if default_return is None else default_return
        
        return wrapper
    return decorator

def validate_data(
    data: Dict[str, Any],
    schema: Type[BaseModel],
    allow_partial: bool = False
) -> ValidationResult:
    """
    Validate data against a Pydantic schema
    
    Args:
        data: Data to validate
        schema: Pydantic model to validate against
        allow_partial: Whether to allow partial validation
    
    Returns:
        ValidationResult object
    """
    try:
        if allow_partial:
            # For partial validation, we need to filter the schema
            from pydantic import create_model
            partial_fields = {k: (Optional[v.type_], None) 
                            for k, v in schema.__fields__.items()}
            PartialSchema = create_model('PartialSchema', **partial_fields)
            cleaned = PartialSchema(**data).dict(exclude_none=True)
        else:
            cleaned = schema(**data).dict()
        
        return ValidationResult(
            is_valid=True,
            cleaned_data=cleaned
        )
        
    except ValidationError as e:
        errors = {}
        for error in e.errors():
            field = ".".join(str(loc) for loc in error['loc'])
            errors[field] = error['msg']
        
        return ValidationResult(
            is_valid=False,
            errors=errors
        )

def retry_on_failure(
    config: Optional[RetryConfig] = None,
    before_retry: Optional[Callable] = None
) -> Callable:
    """
    Decorator for retrying failed operations
    
    Args:
        config: Retry configuration
        before_retry: Function to call before each retry
    
    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(config.max_attempts),
            wait=wait_exponential(
                multiplier=config.wait_multiplier,
                min=config.wait_min,
                max=config.wait_max
            ),
            retry=retry_if_exception_type(config.retry_on_exceptions),
            before_sleep=before_retry
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

class ResourceManager:
    """
    Context manager for managing resources with proper cleanup
    """
    
    def __init__(self, resource, cleanup_func: Callable):
        self.resource = resource
        self.cleanup_func = cleanup_func
    
    def __enter__(self):
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.cleanup_func(self.resource)
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")
        
        if exc_type:
            logger.error(f"Exception in context manager: {exc_val}")
            return False  # Re-raise exception
        
        return True

def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log execution time of functions
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Completed {func.__name__} in {execution_time:.2f} seconds"
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}"
            )
            raise
    
    return wrapper

def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate file path with comprehensive checks
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
    
    Returns:
        True if valid, False otherwise
    """
    from pathlib import Path
    import os
    
    try:
        path = Path(file_path)
        
        # Check for null bytes (path traversal attempt)
        if '\x00' in str(file_path):
            raise ValueError("Path contains null byte")
        
        # Check for absolute path
        if path.is_absolute() and not file_path.startswith('/'):
            raise ValueError("Invalid absolute path")
        
        # Check path length
        if len(str(path)) > 4096:
            raise ValueError("Path too long")
        
        # Check if parent directory exists for write operations
        if not must_exist:
            parent = path.parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists (if required)
        if must_exist and not path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        # Check permissions
        if path.exists():
            if must_exist and not os.access(path, os.R_OK):
                raise PermissionError(f"No read permission: {file_path}")
            if not must_exist and path.parent.exists() and not os.access(path.parent, os.W_OK):
                raise PermissionError(f"No write permission for directory: {path.parent}")
        
        return True
        
    except (ValueError, FileNotFoundError, PermissionError) as e:
        logger.error(f"File path validation failed for {file_path}: {str(e)}")
        return False

def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 500
) -> Dict[str, Any]:
    """
    Create standardized error response
    """
    return {
        "error": {
            "code": error_code,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        },
        "status_code": status_code
    }

# Usage examples in other modules
if __name__ == "__main__":
    # Example usage
    @handle_exceptions(default_return={"error": "Failed"})
    @log_execution_time
    def risky_operation(data):
        if not data:
            raise DataValidationError("Data cannot be empty")
        return {"result": "success"}
    
    # Test the decorator
    print(risky_operation({}))  # Should return error response
    print(risky_operation({"key": "value"}))  # Should succeed
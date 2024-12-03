import logging
import streamlit as st
from functools import wraps
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def handle_db_errors(func):
    """Decorator for handling database operations errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Database error in {func.__name__}: {error_code} - {error_message}")
            st.error(f"Database error: {error_message}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            st.error("An unexpected error occurred")
            return None
    return wrapper

def handle_api_errors(func):
    """Decorator for handling API calls errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"API error in {func.__name__}: {str(e)}")
            st.error(f"API error: {str(e)}")
            return None
    return wrapper

class ApplicationError(Exception):
    """Base exception class for application-specific errors"""
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code
        logger.error(f"Application error {error_code}: {message}")

def init_database():
    """Initialize database connection with error handling"""
    from utils.db import DynamoDBManager
    try:
        db = DynamoDBManager()
        return db
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        st.error("Failed to connect to database. Please check your configuration.")
        return None 
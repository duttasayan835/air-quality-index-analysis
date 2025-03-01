"""
Input validation utilities for the Air Quality Index application.
These functions help validate user inputs to prevent security vulnerabilities.
"""

import re
import streamlit as st
import logging

# Configure logging
logger = logging.getLogger(__name__)

def validate_text_input(text, field_name="input", min_length=1, max_length=100, 
                        pattern=None, error_location=None):
    """
    Validate a text input field.
    
    Args:
        text (str): The text input to validate
        field_name (str): Name of the field for error messages
        min_length (int): Minimum allowed length
        max_length (int): Maximum allowed length
        pattern (str): Optional regex pattern the input must match
        error_location: Streamlit container to display error in (if None, uses st directly)
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not text or len(text.strip()) < min_length:
        error_msg = f"{field_name} must be at least {min_length} characters."
        if error_location:
            error_location.error(error_msg)
        else:
            st.error(error_msg)
        return False
    
    if len(text) > max_length:
        error_msg = f"{field_name} must be less than {max_length} characters."
        if error_location:
            error_location.error(error_msg)
        else:
            st.error(error_msg)
        return False
    
    if pattern and not re.match(pattern, text):
        error_msg = f"{field_name} contains invalid characters or format."
        if error_location:
            error_location.error(error_msg)
        else:
            st.error(error_msg)
        return False
    
    return True

def validate_numeric_input(value, field_name, min_val=None, max_val=None, error_location=None):
    """
    Validate that a numeric input is within acceptable ranges.
    
    Args:
        value: The input value to validate
        field_name: Name of the field (for error messages)
        min_val: Minimum acceptable value (optional)
        max_val: Maximum acceptable value (optional)
        error_location: Streamlit container to display error in (optional)
    
    Returns:
        tuple: (is_valid, value) where is_valid is a boolean and value is the validated value
    """
    try:
        # Convert to float if it's not already
        value = float(value)
        
        # Check minimum value
        if min_val is not None and value < min_val:
            error_msg = f"{field_name} must be at least {min_val}"
            if error_location:
                error_location.error(error_msg)
            logger.warning(f"Validation error: {error_msg}")
            return False, value
        
        # Check maximum value
        if max_val is not None and value > max_val:
            error_msg = f"{field_name} must be at most {max_val}"
            if error_location:
                error_location.error(error_msg)
            logger.warning(f"Validation error: {error_msg}")
            return False, value
        
        return True, value
    
    except (ValueError, TypeError):
        error_msg = f"{field_name} must be a valid number"
        if error_location:
            error_location.error(error_msg)
        logger.warning(f"Validation error: {error_msg}")
        return False, 0.0

def validate_location_input(location, error_location=None):
    """
    Validate a location input field.
    
    Args:
        location (str): The location input to validate
        error_location: Streamlit container to display error in (if None, uses st directly)
    
    Returns:
        bool: True if valid, False otherwise
    """
    # First check basic text validation
    if not validate_text_input(location, "Location", min_length=2, max_length=100, 
                              error_location=error_location):
        return False
    
    # Check for potentially dangerous characters for injection
    dangerous_chars = ['<', '>', ';', '&', '|', '`', '$', '#', '{', '}', '[', ']']
    for char in dangerous_chars:
        if char in location:
            error_msg = f"Location contains invalid character: {char}"
            if error_location:
                error_location.error(error_msg)
            else:
                st.error(error_msg)
            return False
    
    # Additional location-specific validation could go here
    # For example, matching against a regex pattern for city, state format
    
    return True

def validate_date_input(date, min_date=None, max_date=None, error_location=None):
    """
    Validate a date input.
    
    Args:
        date: The date to validate (datetime.date object)
        min_date: Minimum allowed date (if None, no minimum)
        max_date: Maximum allowed date (if None, no maximum)
        error_location: Streamlit container to display error in (if None, uses st directly)
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not date:
        error_msg = "Date is required."
        if error_location:
            error_location.error(error_msg)
        else:
            st.error(error_msg)
        return False
    
    if min_date and date < min_date:
        error_msg = f"Date must be on or after {min_date.strftime('%Y-%m-%d')}."
        if error_location:
            error_location.error(error_msg)
        else:
            st.error(error_msg)
        return False
    
    if max_date and date > max_date:
        error_msg = f"Date must be on or before {max_date.strftime('%Y-%m-%d')}."
        if error_location:
            error_location.error(error_msg)
        else:
            st.error(error_msg)
        return False
    
    return True

def sanitize_input(value):
    """
    Sanitize input to prevent injection attacks.
    
    Args:
        value: The input string to sanitize
    
    Returns:
        str: Sanitized string
    """
    if not isinstance(value, str):
        return value
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>\'";`]', '', value)
    
    # Log if sanitization changed the input
    if sanitized != value:
        logger.warning(f"Input sanitized: '{value}' -> '{sanitized}'")
    
    return sanitized

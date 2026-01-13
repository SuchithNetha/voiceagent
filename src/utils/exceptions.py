"""
Custom Exceptions for Sarah Voice Agent.

Having custom exceptions allows you to:
1. Handle specific error types differently (graceful degradation)
2. Provide user-friendly error messages
3. Log appropriate information for debugging
4. Maintain consistent error handling across the app

Production best practice: Always catch specific exceptions, not just `Exception`.
"""

from typing import Optional


class SarahBaseException(Exception):
    """
    Base exception for all Sarah-related errors.
    
    All custom exceptions inherit from this so you can:
    - Catch all Sarah errors with `except SarahBaseException`
    - Or catch specific ones like `except SearchEngineError`
    """
    
    def __init__(
        self, 
        message: str, 
        user_message: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Args:
            message: Technical message for logs
            user_message: Friendly message to speak to the user
            original_error: The underlying exception that caused this
        """
        super().__init__(message)
        self.message = message
        self.user_message = user_message or "I encountered a technical issue. Please try again."
        self.original_error = original_error
    
    def __str__(self):
        if self.original_error:
            return f"{self.message} (Caused by: {type(self.original_error).__name__}: {self.original_error})"
        return self.message


# --- CONFIGURATION ERRORS ---
class ConfigurationError(SarahBaseException):
    """Raised when there's a configuration issue (missing API keys, invalid settings)."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        user_message = "I'm having trouble with my configuration. Please contact support."
        super().__init__(message, user_message)
        self.config_key = config_key


# --- MODEL/AI ERRORS ---
class ModelLoadError(SarahBaseException):
    """Raised when AI models fail to load (STT, TTS, LLM)."""
    
    def __init__(self, model_name: str, original_error: Optional[Exception] = None):
        message = f"Failed to load model: {model_name}"
        user_message = "I'm having trouble starting up. Please try again in a moment."
        super().__init__(message, user_message, original_error)
        self.model_name = model_name


class TranscriptionError(SarahBaseException):
    """Raised when speech-to-text fails."""
    
    def __init__(self, original_error: Optional[Exception] = None):
        message = "Speech transcription failed"
        user_message = "I couldn't understand that. Could you please repeat?"
        super().__init__(message, user_message, original_error)


class SynthesisError(SarahBaseException):
    """Raised when text-to-speech fails."""
    
    def __init__(self, original_error: Optional[Exception] = None):
        message = "Speech synthesis failed"
        user_message = None  # Can't speak if TTS is broken!
        super().__init__(message, user_message, original_error)


# --- SEARCH/DATABASE ERRORS ---
class SearchEngineError(SarahBaseException):
    """Raised when the Superlinked search engine fails."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        user_message = "I'm having trouble searching right now. Let me try again."
        super().__init__(message, user_message, original_error)


class DataLoadError(SarahBaseException):
    """Raised when property data fails to load."""
    
    def __init__(self, file_path: str, original_error: Optional[Exception] = None):
        message = f"Failed to load data from: {file_path}"
        user_message = "I'm having trouble accessing my property database."
        super().__init__(message, user_message, original_error)
        self.file_path = file_path


# --- AGENT ERRORS ---
class AgentError(SarahBaseException):
    """Raised when the LangGraph agent fails."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        user_message = "I lost my train of thought. Could you repeat that?"
        super().__init__(message, user_message, original_error)


class ToolExecutionError(SarahBaseException):
    """Raised when a tool (like search_properties) fails during execution."""
    
    def __init__(self, tool_name: str, original_error: Optional[Exception] = None):
        message = f"Tool '{tool_name}' execution failed"
        user_message = "I ran into a problem while looking that up. Let me try differently."
        super().__init__(message, user_message, original_error)
        self.tool_name = tool_name


# --- NETWORK/API ERRORS ---
class APIError(SarahBaseException):
    """Raised when external API calls fail (Groq, etc.)."""
    
    def __init__(self, service_name: str, original_error: Optional[Exception] = None):
        message = f"API call to {service_name} failed"
        user_message = "I'm having trouble connecting to my brain. Just a moment."
        super().__init__(message, user_message, original_error)
        self.service_name = service_name


class RateLimitError(APIError):
    """Raised when we hit API rate limits."""
    
    def __init__(self, service_name: str, retry_after: Optional[int] = None):
        super().__init__(service_name)
        self.user_message = "I need a quick breather. Please try again in a few seconds."
        self.retry_after = retry_after

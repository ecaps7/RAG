"""Unified LLM service for the RAG agent."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from typing import Optional, Type
from pydantic import BaseModel

from ..config import get_config, init_model_with_config
from ..utils.logging import get_logger


class LLMServices:
    """Unified LLM service with singleton pattern."""
    
    _instance: Optional[LLMServices] = None
    
    def __new__(cls) -> LLMServices:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the LLM service."""
        if not hasattr(self, "_model"):
            self.logger = get_logger(self.__class__.__name__)
            self._model = None
            self._init_model()
    
    def _init_model(self):
        """Initialize the LLM model."""
        try:
            cfg = get_config()
            self._model = init_model_with_config(
                cfg.response_model_name,
                cfg.response_model_temperature
            )
            self.logger.info(f"Initialized LLM: {get_config().response_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def get_model(self) -> BaseChatModel:
        """Get the base LLM model.
        
        Returns:
            The base LLM model instance.
        """
        if self._model is None:
            self._init_model()
        return self._model
    
    def get_structured_model(self, schema: Type[BaseModel]) -> BaseChatModel:
        """Get an LLM model with structured output.
        
        Args:
            schema: The Pydantic model schema for structured output.
            
        Returns:
            An LLM model instance with structured output binding.
        """
        base_model = self.get_model()
        return base_model.with_structured_output(schema)


# Create singleton instance
llm_services = LLMServices()

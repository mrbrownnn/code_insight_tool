"""LLM Factory - Multi-provider initialization (Ollama, Gemini, Groq)."""
import os
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.schema.language_model import BaseLLM

from config import Settings


class LLMProvider(ABC):
    @abstractmethod
    def get_llm(self) -> BaseLLM:
        pass


class OllamaProvider(LLMProvider):
    def __init__(self, config: Settings):
        self.config = config
    
    def get_llm(self) -> BaseLLM:
        return ChatOllama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.llm_temperature,
        )

class OpenAIProvider(LLMProvider):
    def __init__(self, config: Settings):
        self.config = config
    
    def get_llm(self) -> BaseLLM:
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        return ChatOpenAI(
            model=self.config.llm_model,
            api_key=api_key,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
class GeminiProvider(LLMProvider):
    def __init__(self, config: Settings):
        self.config = config
    
    def get_llm(self) -> BaseLLM:
        api_key = self.config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        return ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            api_key=api_key,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )


class GroqProvider(LLMProvider):
    def __init__(self, config: Settings):
        self.config = config
    
    def get_llm(self) -> BaseLLM:
        api_key = self.config.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
        return ChatGroq(
            model=self.config.llm_model,
            api_key=api_key,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens,
        )


class LLMFactory:
    def __init__(self, config: Settings):
        self.config = config
        self._providers = {
            "ollama": OllamaProvider(config),
            "gemini": GeminiProvider(config),
            "groq": GroqProvider(config),
            "chatgpt": OpenAIProvider(config),
        }
    
    def get_llm(self, provider_name: Optional[str] = None) -> BaseLLM:
        if provider_name is None:
            provider_name = self._detect_provider()
        
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        return self._providers[provider_name].get_llm()
    
    def _detect_provider(self) -> str:
        model = self.config.llm_model.lower()
        switch: dict[str, list[str]] = {
            "ollama": ["ollama"],
            "gemini": ["gemini", "google"],
            "groq": ["groq"],
            "chatgpt": ["gpt", "openai"],
        }
        for provider, model_names in switch.items():
            if any(name in model for name in model_names):
                return provider
        raise ValueError(f"Unable to detect provider for model: {model}")


def create_llm_factory(config: Optional[Settings] = None) -> LLMFactory:
    if config is None:
        from config import settings
        config = settings
    return LLMFactory(config)
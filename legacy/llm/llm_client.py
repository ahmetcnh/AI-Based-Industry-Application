"""
LLM Client Module for AI Industrial Platform.

Provides a unified interface for multiple LLM providers including
Ollama (local), OpenAI, and Anthropic with automatic fallback.
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """
    Standardized LLM response.
    
    Attributes:
        content: Generated text content.
        model: Model used for generation.
        provider: Provider name.
        usage: Token usage information.
        finish_reason: Reason for completion.
    """
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a response."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local models."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Ollama API URL.
            timeout: Request timeout in seconds.
        """
        from config.settings import get_settings
        settings = get_settings()
        
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.routing_model = settings.OLLAMA_ROUTING_MODEL
        self.generation_model = settings.OLLAMA_GENERATION_MODEL
    
    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Ollama.
        
        Args:
            prompt: User prompt.
            system_prompt: System instructions.
            model: Model to use.
            **kwargs: Additional parameters.
        
        Returns:
            Generated text.
        """
        model = model or self.generation_model
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_tokens", 2048),
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
        
        except httpx.TimeoutException:
            logger.error("Ollama request timed out", model=model)
            raise
        except Exception as e:
            logger.error("Ollama generation failed", model=model, error=str(e))
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate streaming response using Ollama.
        
        Args:
            prompt: User prompt.
            system_prompt: System instructions.
            model: Model to use.
            **kwargs: Additional parameters.
        
        Yields:
            Text chunks.
        """
        model = model or self.generation_model
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("max_tokens", 2048),
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            logger.error("Ollama streaming failed", model=model, error=str(e))
            raise


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key.
            base_url: API base URL (for compatible endpoints).
        """
        from config.settings import get_settings
        settings = get_settings()
        
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = (base_url or settings.OPENAI_BASE_URL).rstrip("/")
        self.routing_model = settings.OPENAI_ROUTING_MODEL
        self.generation_model = settings.OPENAI_GENERATION_MODEL
    
    async def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return bool(self.api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using OpenAI."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        model = model or self.generation_model
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 0.9),
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error("OpenAI generation failed", model=model, error=str(e))
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming response using OpenAI."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        model = model or self.generation_model
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                content = data["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            logger.error("OpenAI streaming failed", model=model, error=str(e))
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key.
        """
        from config.settings import get_settings
        settings = get_settings()
        
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.base_url = "https://api.anthropic.com/v1"
        self.routing_model = settings.ANTHROPIC_ROUTING_MODEL
        self.generation_model = settings.ANTHROPIC_GENERATION_MODEL
    
    async def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return bool(self.api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response using Anthropic."""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
        
        model = model or self.generation_model
        
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2024-01-01"
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                return data["content"][0]["text"]
        
        except Exception as e:
            logger.error("Anthropic generation failed", model=model, error=str(e))
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming response using Anthropic."""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
        
        model = model or self.generation_model
        
        payload = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2024-01-01"
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    json=payload,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "content_block_delta":
                                    text = data.get("delta", {}).get("text", "")
                                    if text:
                                        yield text
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            logger.error("Anthropic streaming failed", model=model, error=str(e))
            raise


class LLMClient:
    """
    Unified LLM client with multi-provider support and automatic fallback.
    
    Provides a single interface for generating text using multiple LLM providers,
    with automatic fallback if the primary provider fails.
    
    Attributes:
        providers: Dict of initialized provider instances.
        provider_priority: List of provider names in priority order.
    """
    
    def __init__(self):
        """Initialize the LLM client with configured providers."""
        from config.settings import get_settings
        self.settings = get_settings()
        
        # Initialize providers
        self.providers: Dict[str, BaseLLMProvider] = {
            "ollama": OllamaProvider(),
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
        }
        
        self.provider_priority = self.settings.llm_providers_list
        self._available_providers: Optional[List[str]] = None
    
    async def _get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        if self._available_providers is not None:
            return self._available_providers
        
        available = []
        for name in self.provider_priority:
            if name in self.providers:
                provider = self.providers[name]
                if await provider.is_available():
                    available.append(name)
                    logger.info(f"Provider available: {name}")
                else:
                    logger.debug(f"Provider unavailable: {name}")
        
        self._available_providers = available
        return available
    
    def _get_model_for_type(
        self,
        provider: BaseLLMProvider,
        model_type: str
    ) -> str:
        """
        Get the appropriate model for a task type.
        
        Args:
            provider: LLM provider instance.
            model_type: Either "routing" or "generation".
        
        Returns:
            Model name to use.
        """
        if model_type == "routing":
            return provider.routing_model
        return provider.generation_model
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: str = "generation",
        model: Optional[str] = None,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the best available provider.
        
        Args:
            prompt: User prompt.
            system_prompt: System instructions.
            model_type: "routing" (fast) or "generation" (powerful).
            model: Specific model override.
            provider_name: Specific provider to use.
            **kwargs: Additional generation parameters.
        
        Returns:
            Generated text.
        
        Raises:
            RuntimeError: If no providers are available.
        """
        available = await self._get_available_providers()
        
        if not available:
            raise RuntimeError("No LLM providers available")
        
        # Use specific provider if requested
        if provider_name and provider_name in available:
            providers_to_try = [provider_name]
        else:
            providers_to_try = available
        
        last_error = None
        
        for name in providers_to_try:
            provider = self.providers[name]
            model_to_use = model or self._get_model_for_type(provider, model_type)
            
            try:
                logger.debug(
                    "Attempting generation",
                    provider=name,
                    model=model_to_use,
                    model_type=model_type
                )
                
                result = await provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model_to_use,
                    **kwargs
                )
                
                logger.debug("Generation successful", provider=name)
                return result
            
            except Exception as e:
                last_error = e
                logger.warning(
                    "Provider failed, trying next",
                    provider=name,
                    error=str(e)
                )
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: str = "generation",
        model: Optional[str] = None,
        provider_name: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate streaming text using the best available provider.
        
        Args:
            prompt: User prompt.
            system_prompt: System instructions.
            model_type: "routing" or "generation".
            model: Specific model override.
            provider_name: Specific provider to use.
            **kwargs: Additional parameters.
        
        Yields:
            Text chunks.
        
        Raises:
            RuntimeError: If no providers are available.
        """
        available = await self._get_available_providers()
        
        if not available:
            raise RuntimeError("No LLM providers available")
        
        if provider_name and provider_name in available:
            providers_to_try = [provider_name]
        else:
            providers_to_try = available
        
        for name in providers_to_try:
            provider = self.providers[name]
            model_to_use = model or self._get_model_for_type(provider, model_type)
            
            try:
                async for chunk in provider.generate_stream(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model_to_use,
                    **kwargs
                ):
                    yield chunk
                return
            
            except Exception as e:
                logger.warning(
                    "Streaming provider failed",
                    provider=name,
                    error=str(e)
                )
                continue
        
        raise RuntimeError("All streaming providers failed")
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_type: str = "routing",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response.
        
        Args:
            prompt: Prompt requesting JSON output.
            system_prompt: System instructions.
            model_type: Model type to use.
            **kwargs: Additional parameters.
        
        Returns:
            Parsed JSON as dictionary.
        """
        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\nRespond only with valid JSON."
        
        response = await self.generate(
            prompt=prompt,
            system_prompt=json_system,
            model_type=model_type,
            **kwargs
        )
        
        # Clean up response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(
                line for line in lines 
                if not line.startswith("```")
            )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("JSON parsing failed", response=response[:200], error=str(e))
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check health of all providers.
        
        Returns:
            Dict with provider health status.
        """
        health = {}
        
        for name, provider in self.providers.items():
            try:
                is_available = await provider.is_available()
                health[name] = {
                    "available": is_available,
                    "routing_model": provider.routing_model,
                    "generation_model": provider.generation_model
                }
            except Exception as e:
                health[name] = {
                    "available": False,
                    "error": str(e)
                }
        
        return health


if __name__ == "__main__":
    import asyncio
    
    async def test_llm():
        print("Initializing LLM client...")
        client = LLMClient()
        
        # Check health
        print("\nChecking provider health...")
        health = await client.check_health()
        for provider, status in health.items():
            print(f"  {provider}: {status}")
        
        # Test generation
        print("\nTesting generation...")
        try:
            response = await client.generate(
                prompt="What is 2 + 2? Reply with just the number.",
                system_prompt="You are a helpful assistant.",
                model_type="routing"
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Generation failed: {e}")
        
        # Test JSON generation
        print("\nTesting JSON generation...")
        try:
            response = await client.generate_json(
                prompt='Generate a JSON object with keys "name" and "value".',
                model_type="routing"
            )
            print(f"JSON Response: {response}")
        except Exception as e:
            print(f"JSON generation failed: {e}")
        
        print("\nLLM client test completed!")
    
    asyncio.run(test_llm())

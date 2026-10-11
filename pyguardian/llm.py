"""
llm.py — LLM Provider Interface & Local Ollama Backend for PyGuardian.

Responsibilities:
- Provide an abstract LLMProvider interface for language model backends.
- Implement OllamaProvider for local non-streaming generation over HTTP.
- Encapsulate provider configuration (model, base_url, timeout).
- Handle HTTP errors, timeouts, connection failures, and malformed responses cleanly.
- Maintain a strict communication boundary: higher-level AI code only interacts with LLMProvider.

Design Principles:
- Provider abstraction: NO direct coupling between PyGuardian and specific LLM providers.
- Security-first: NO prompt logging by default, NO credentials logging, NO external network calls unless configured.
- 0 AST operations, 0 migration reasoning, 0 file scanning, 0 patch generation, 0 Git commands.
- Pure communication layer only.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default Ollama configuration
DEFAULT_OLLAMA_MODEL: str = "llama3.2"
DEFAULT_OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS: float = 60.0
DEFAULT_PROVIDER_NAME: str = "ollama"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMProviderError(Exception):
    """Base exception for all LLM provider and generation errors in PyGuardian."""
    pass


class LLMConfigurationError(LLMProviderError):
    """Raised when LLM configuration parameters are invalid."""
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """
    Immutable configuration for an LLM provider backend.

    Attributes:
        model:    Model identifier (e.g. 'llama3.2', 'mistral', 'codellama').
        base_url: Base URL of the provider endpoint (e.g. 'http://localhost:11434').
        timeout:  Request timeout in seconds.
        provider: Provider backend identifier ('ollama').
    """

    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    provider: str = DEFAULT_PROVIDER_NAME

    def __post_init__(self) -> None:
        """Validate configuration fields."""
        if not isinstance(self.model, str) or not self.model.strip():
            raise LLMConfigurationError("LLM model name must be a non-empty string.")

        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise LLMConfigurationError("LLM base URL must be a non-empty string.")

        clean_url = self.base_url.strip()
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            raise LLMConfigurationError(
                f"LLM base URL must start with 'http://' or 'https://', got '{self.base_url}'."
            )

        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise LLMConfigurationError(
                f"LLM timeout must be a positive number of seconds, got {self.timeout}."
            )

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise LLMConfigurationError("LLM provider must be a non-empty string.")


def get_default_llm_config() -> LLMConfig:
    """
    Construct default LLMConfig, optionally reading environment overrides:
    - PYGUARDIAN_LLM_MODEL
    - PYGUARDIAN_OLLAMA_BASE_URL / PYGUARDIAN_LLM_BASE_URL
    - PYGUARDIAN_LLM_TIMEOUT
    - PYGUARDIAN_LLM_PROVIDER
    """
    model = os.environ.get("PYGUARDIAN_LLM_MODEL", DEFAULT_OLLAMA_MODEL)
    base_url = (
        os.environ.get("PYGUARDIAN_OLLAMA_BASE_URL")
        or os.environ.get("PYGUARDIAN_LLM_BASE_URL")
        or DEFAULT_OLLAMA_BASE_URL
    )

    timeout_env = os.environ.get("PYGUARDIAN_LLM_TIMEOUT")
    timeout = DEFAULT_TIMEOUT_SECONDS
    if timeout_env:
        try:
            timeout = float(timeout_env)
        except ValueError:
            raise LLMConfigurationError(
                f"Invalid PYGUARDIAN_LLM_TIMEOUT value: '{timeout_env}'"
            )

    provider = os.environ.get("PYGUARDIAN_LLM_PROVIDER", DEFAULT_PROVIDER_NAME)

    return LLMConfig(
        model=model,
        base_url=base_url,
        timeout=timeout,
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Abstract Provider Interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """
    Abstract interface for LLM backends in PyGuardian.

    Higher-level AI reasoning components interact exclusively with this interface.
    """

    @property
    @abstractmethod
    def config(self) -> LLMConfig:
        """The underlying configuration for this provider instance."""
        ...

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Send a string prompt to the model and return the generated completion text.

        Args:
            prompt: Non-empty prompt string to send to the LLM.

        Returns:
            Generated text completion.

        Raises:
            LLMProviderError: If the prompt is invalid, request fails, times out,
                              or response cannot be parsed.
        """
        ...


# ---------------------------------------------------------------------------
# Ollama Local Provider
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """
    Ollama local LLM provider communicating via Ollama's HTTP generate endpoint.

    Endpoint: POST {base_url}/api/generate
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize an OllamaProvider.

        Args:
            config:   Optional LLMConfig object.
            model:    Optional model override.
            base_url: Optional base_url override.
            timeout:  Optional timeout override in seconds.
        """
        if config is not None:
            c_model = model or config.model
            c_base = base_url or config.base_url
            c_timeout = timeout if timeout is not None else config.timeout
            self._config = LLMConfig(
                model=c_model,
                base_url=c_base,
                timeout=c_timeout,
                provider="ollama",
            )
        else:
            default_cfg = get_default_llm_config()
            c_model = model or default_cfg.model
            c_base = base_url or default_cfg.base_url
            c_timeout = timeout if timeout is not None else default_cfg.timeout
            self._config = LLMConfig(
                model=c_model,
                base_url=c_base,
                timeout=c_timeout,
                provider="ollama",
            )

    @property
    def config(self) -> LLMConfig:
        """The configuration used by this OllamaProvider."""
        return self._config

    @property
    def endpoint_url(self) -> str:
        """The full URL of Ollama's generate endpoint."""
        return f"{self._config.base_url.rstrip('/')}/api/generate"

    def generate(self, prompt: str) -> str:
        """
        Execute non-streaming completion against local Ollama instance.

        Args:
            prompt: Text prompt to generate completion for.

        Returns:
            Generated string response from Ollama.

        Raises:
            LLMProviderError: On validation, HTTP, connection, timeout, or parsing error.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMProviderError("Prompt must be a non-empty string.")

        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
        }

        url = self.endpoint_url
        logger.debug("Dispatching request to Ollama endpoint: %s (model=%s)", url, self._config.model)

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._config.timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.exceptions.Timeout as exc:
            raise LLMProviderError(
                f"Request to Ollama at '{url}' timed out after {self._config.timeout}s: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMProviderError(
                f"Could not connect to Ollama at '{url}'. Is the Ollama daemon running? Error: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMProviderError(
                f"Network request to Ollama at '{url}' failed: {exc}"
            ) from exc

        # Handle HTTP status codes
        if response.status_code == 404:
            raise LLMProviderError(
                f"Ollama resource or model '{self._config.model}' not found (HTTP 404) at '{url}'."
            )
        elif response.status_code == 429:
            raise LLMProviderError(
                f"Rate limit exceeded (HTTP 429) while querying Ollama at '{url}'."
            )
        elif not response.ok:
            raise LLMProviderError(
                f"Ollama returned HTTP error {response.status_code} from '{url}': {response.text[:200]}"
            )

        # Parse JSON response
        try:
            data = response.json()
        except (ValueError, Exception) as exc:
            raise LLMProviderError(
                f"Malformed JSON response received from Ollama at '{url}': {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMProviderError(
                f"Expected JSON object from Ollama response, got {type(data).__name__}."
            )

        if "response" not in data:
            raise LLMProviderError(
                f"Missing required 'response' field in Ollama response. Keys received: {list(data.keys())}"
            )

        result_text = data["response"]
        if not isinstance(result_text, str):
            raise LLMProviderError(
                f"Expected string 'response' from Ollama, got {type(result_text).__name__}."
            )

        return result_text


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

def create_llm_provider(
    config: Optional[LLMConfig] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> LLMProvider:
    """
    Factory function to instantiate an LLMProvider based on configuration.

    Args:
        config:   Optional LLMConfig object.
        provider: Provider identifier override ('ollama').
        model:    Model identifier override.
        base_url: Base URL override.
        timeout:  Timeout override in seconds.

    Returns:
        Configured LLMProvider instance.

    Raises:
        LLMConfigurationError: If the provider is unsupported or config is invalid.
    """
    target_provider = (
        provider
        or (config.provider if config else None)
        or os.environ.get("PYGUARDIAN_LLM_PROVIDER")
        or DEFAULT_PROVIDER_NAME
    ).lower().strip()

    if target_provider == "ollama":
        return OllamaProvider(
            config=config,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    raise LLMConfigurationError(
        f"Unsupported LLM provider: '{target_provider}'. Supported providers: ['ollama']."
    )

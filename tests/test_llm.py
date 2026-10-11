"""
test_llm.py — Unit tests for PyGuardian LLM Provider Interface & Ollama Provider.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch
import pytest
import requests

from pyguardian.llm import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    LLMConfig,
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    OllamaProvider,
    create_llm_provider,
    get_default_llm_config,
)


# ---------------------------------------------------------------------------
# Fake Provider for Interface Testing
# ---------------------------------------------------------------------------

class FakeLLMProvider(LLMProvider):
    """A minimal fake LLM provider for interface testing."""

    def __init__(self, response_text: str = "Fake generated migration suggestion.") -> None:
        self._config = LLMConfig(model="fake-model", base_url="http://localhost:9999", provider="fake")
        self._response_text = response_text
        self.received_prompts: list[str] = []

    @property
    def config(self) -> LLMConfig:
        return self._config

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise LLMProviderError("Prompt must be non-empty.")
        self.received_prompts.append(prompt)
        return self._response_text


# ---------------------------------------------------------------------------
# Interface Tests (1-2)
# ---------------------------------------------------------------------------

def test_1_abstract_llm_provider_cannot_be_instantiated():
    """Test 1 — LLMProvider is an abstract base class and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_2_fake_provider_implements_generate():
    """Test 2 — A concrete subclass can implement LLMProvider and execute generate()."""
    provider = FakeLLMProvider(response_text="Use pd.read_parquet instead.")
    result = provider.generate("How to migrate read_csv?")

    assert result == "Use pd.read_parquet instead."
    assert provider.received_prompts == ["How to migrate read_csv?"]
    assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# Configuration Tests (3-7)
# ---------------------------------------------------------------------------

def test_3_default_configuration_is_valid():
    """Test 3 — Default configuration produces valid standard values."""
    config = get_default_llm_config()

    assert config.model == DEFAULT_OLLAMA_MODEL
    assert config.base_url == DEFAULT_OLLAMA_BASE_URL
    assert config.timeout == DEFAULT_TIMEOUT_SECONDS
    assert config.provider == "ollama"


def test_4_explicit_model_configuration_works():
    """Test 4 — Explicit model configuration is stored correctly."""
    config = LLMConfig(model="codellama:13b")

    assert config.model == "codellama:13b"
    assert config.base_url == DEFAULT_OLLAMA_BASE_URL


def test_5_explicit_base_url_works():
    """Test 5 — Explicit base URL is accepted and normalized."""
    config = LLMConfig(base_url="http://192.168.1.50:11434")

    assert config.base_url == "http://192.168.1.50:11434"


def test_6_timeout_configuration_works():
    """Test 6 — Explicit timeout configuration is preserved."""
    config = LLMConfig(timeout=120.0)

    assert config.timeout == 120.0


def test_7_invalid_configurations_rejected():
    """Test 7 — Invalid configurations raise LLMConfigurationError."""
    # Empty model
    with pytest.raises(LLMConfigurationError, match="model"):
        LLMConfig(model="")

    # Whitespace model
    with pytest.raises(LLMConfigurationError, match="model"):
        LLMConfig(model="   ")

    # Empty base URL
    with pytest.raises(LLMConfigurationError, match="base URL"):
        LLMConfig(base_url="")

    # Invalid URL scheme
    with pytest.raises(LLMConfigurationError, match="http://"):
        LLMConfig(base_url="ftp://localhost:11434")

    # Non-positive timeout
    with pytest.raises(LLMConfigurationError, match="positive"):
        LLMConfig(timeout=0)

    with pytest.raises(LLMConfigurationError, match="positive"):
        LLMConfig(timeout=-10.0)

    # Empty provider
    with pytest.raises(LLMConfigurationError, match="provider"):
        LLMConfig(provider="")


def test_configuration_environment_variables(monkeypatch):
    """Test environment variable overrides for get_default_llm_config."""
    monkeypatch.setenv("PYGUARDIAN_LLM_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setenv("PYGUARDIAN_OLLAMA_BASE_URL", "http://ollama-server:11434")
    monkeypatch.setenv("PYGUARDIAN_LLM_TIMEOUT", "45.5")

    config = get_default_llm_config()

    assert config.model == "qwen2.5-coder:7b"
    assert config.base_url == "http://ollama-server:11434"
    assert config.timeout == 45.5


def test_configuration_invalid_env_timeout(monkeypatch):
    """Test invalid environment timeout value raises LLMConfigurationError."""
    monkeypatch.setenv("PYGUARDIAN_LLM_TIMEOUT", "not_a_number")

    with pytest.raises(LLMConfigurationError, match="Invalid PYGUARDIAN_LLM_TIMEOUT"):
        get_default_llm_config()


# ---------------------------------------------------------------------------
# OllamaProvider Tests (8-20)
# ---------------------------------------------------------------------------

@patch("pyguardian.llm.requests.post")
def test_8_successful_response_returns_generated_text(mock_post):
    """Test 8 — Successful Ollama API response extracts and returns the 'response' text."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "model": "llama3.2",
        "created_at": "2026-09-26T12:00:00Z",
        "response": "Replace `pandas.read_csv` with `pandas.read_parquet`.",
        "done": True,
    }
    mock_post.return_value = mock_resp

    provider = OllamaProvider()
    result = provider.generate("Migration prompt")

    assert result == "Replace `pandas.read_csv` with `pandas.read_parquet`."


def test_9_empty_prompt_is_rejected():
    """Test 9 — Empty or whitespace-only prompt raises LLMProviderError without network call."""
    provider = OllamaProvider()

    with pytest.raises(LLMProviderError, match="non-empty string"):
        provider.generate("")

    with pytest.raises(LLMProviderError, match="non-empty string"):
        provider.generate("   \n\t  ")

    with pytest.raises(LLMProviderError, match="non-empty string"):
        provider.generate(None)  # type: ignore[arg-type]


@patch("pyguardian.llm.requests.post")
def test_10_connection_error_becomes_llm_provider_error(mock_post):
    """Test 10 — Connection error is translated to LLMProviderError with descriptive message."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    provider = OllamaProvider()

    with pytest.raises(LLMProviderError, match="Could not connect to Ollama"):
        provider.generate("Test prompt")


@patch("pyguardian.llm.requests.post")
def test_11_timeout_becomes_llm_provider_error(mock_post):
    """Test 11 — Request timeout is translated to LLMProviderError."""
    mock_post.side_effect = requests.exceptions.Timeout("Read timed out")

    provider = OllamaProvider(timeout=10.0)

    with pytest.raises(LLMProviderError, match="timed out after 10.0s"):
        provider.generate("Test prompt")


@patch("pyguardian.llm.requests.post")
def test_12_http_404_becomes_llm_provider_error(mock_post):
    """Test 12 — HTTP 404 (model or endpoint not found) raises LLMProviderError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.ok = False
    mock_post.return_value = mock_resp

    provider = OllamaProvider(model="nonexistent-model")

    with pytest.raises(LLMProviderError, match="not found \\(HTTP 404\\)"):
        provider.generate("Test prompt")


@patch("pyguardian.llm.requests.post")
def test_13_http_429_becomes_llm_provider_error(mock_post):
    """Test 13 — HTTP 429 (rate limit) raises LLMProviderError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.ok = False
    mock_post.return_value = mock_resp

    provider = OllamaProvider()

    with pytest.raises(LLMProviderError, match="Rate limit exceeded \\(HTTP 429\\)"):
        provider.generate("Test prompt")


@patch("pyguardian.llm.requests.post")
def test_14_other_http_errors_become_llm_provider_error(mock_post):
    """Test 14 — HTTP 500 or other errors raise LLMProviderError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.ok = False
    mock_resp.text = "Internal Server Error"
    mock_post.return_value = mock_resp

    provider = OllamaProvider()

    with pytest.raises(LLMProviderError, match="HTTP error 500"):
        provider.generate("Test prompt")


@patch("pyguardian.llm.requests.post")
def test_15_malformed_json_becomes_llm_provider_error(mock_post):
    """Test 15 — Malformed JSON response body raises LLMProviderError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.side_effect = ValueError("Expecting value: line 1 column 1")
    mock_post.return_value = mock_resp

    provider = OllamaProvider()

    with pytest.raises(LLMProviderError, match="Malformed JSON"):
        provider.generate("Test prompt")


@patch("pyguardian.llm.requests.post")
def test_16_missing_response_field_becomes_llm_provider_error(mock_post):
    """Test 16 — Response missing 'response' field raises LLMProviderError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {"error": "unknown model"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider()

    with pytest.raises(LLMProviderError, match="Missing required 'response' field"):
        provider.generate("Test prompt")


@patch("pyguardian.llm.requests.post")
def test_17_correct_model_sent_in_request(mock_post):
    """Test 17 — The configured model name is correctly passed in payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {"response": "ok"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(model="mistral:instruct")
    provider.generate("Hello model")

    assert mock_post.called
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["model"] == "mistral:instruct"


@patch("pyguardian.llm.requests.post")
def test_18_correct_prompt_sent(mock_post):
    """Test 18 — The prompt argument is correctly sent in the payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {"response": "ok"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider()
    expected_prompt = "Migrate pandas.read_csv to 2.0"
    provider.generate(expected_prompt)

    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["prompt"] == expected_prompt
    assert call_kwargs["json"]["stream"] is False


@patch("pyguardian.llm.requests.post")
def test_19_correct_base_url_and_endpoint_used(mock_post):
    """Test 19 — The /api/generate endpoint on the base URL is used."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {"response": "ok"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(base_url="http://custom-host:8000")
    provider.generate("Test")

    call_args = mock_post.call_args
    assert call_args[0][0] == "http://custom-host:8000/api/generate"


@patch("pyguardian.llm.requests.post")
def test_20_timeout_passed_correctly(mock_post):
    """Test 20 — Configured timeout is passed to requests.post."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {"response": "ok"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(timeout=75.0)
    provider.generate("Test")

    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["timeout"] == 75.0


# ---------------------------------------------------------------------------
# Security & Behavior Tests (21-23)
# ---------------------------------------------------------------------------

@patch("pyguardian.llm.requests.post")
def test_21_prompt_is_not_written_to_logs_by_default(mock_post, caplog):
    """Test 21 — Prompts containing sensitive source data are not logged at INFO/WARNING levels."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {"response": "Migrated successfully"}
    mock_post.return_value = mock_resp

    secret_prompt = "CONFIDENTIAL_SOURCE_CODE_DEF_API_KEY = 9999"

    with caplog.at_level(logging.INFO):
        provider = OllamaProvider()
        provider.generate(secret_prompt)

    assert secret_prompt not in caplog.text


def test_22_provider_does_not_mutate_input():
    """Test 22 — Provider preserves configuration immutability."""
    config = LLMConfig(model="llama3.2", base_url="http://localhost:11434", timeout=30.0)
    provider = OllamaProvider(config)

    assert provider.config == config
    assert provider.config.model == "llama3.2"


def test_23_no_real_network_call_on_initialization():
    """Test 23 — Initializing OllamaProvider makes zero network calls."""
    # Instantiating should not trigger any socket connection or HTTP requests
    provider = OllamaProvider(model="custom", base_url="http://unreachable-host:9999")
    assert provider.config.model == "custom"
    assert provider.endpoint_url == "http://unreachable-host:9999/api/generate"


# ---------------------------------------------------------------------------
# Factory Tests (24-25)
# ---------------------------------------------------------------------------

def test_24_factory_creates_ollama_provider():
    """Test 24 — create_llm_provider('ollama') creates an OllamaProvider instance."""
    provider = create_llm_provider(provider="ollama", model="qwen2.5-coder")

    assert isinstance(provider, OllamaProvider)
    assert provider.config.model == "qwen2.5-coder"
    assert provider.config.provider == "ollama"


def test_25_factory_unsupported_provider_raises_clear_error():
    """Test 25 — Unsupported provider name raises LLMConfigurationError."""
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider: 'openai'"):
        create_llm_provider(provider="openai")

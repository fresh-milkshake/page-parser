import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from src.config.paths import DEFAULT_SETTINGS_FILE


@dataclass
class ProviderConfig:
    """
    Represents the configuration for a single vision provider.

    Attributes:
        name: The provider's name (e.g., "openai").
        model: The model identifier to use (e.g., "gpt-4o").
        base_url: The base URL for the provider's API.
        api_key_env: The name of the environment variable holding the API key.
        api_key_type: The type of API key source (default: "env").
    """

    name: str
    model: str
    base_url: str
    api_key_env: str
    api_key_type: str
    _api_key: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        """
        Retrieve the API key for this provider.

        Returns:
            The API key as a string if available, otherwise None.
        """
        return self._api_key

    def is_configured(self) -> bool:
        """
        Check if the provider is properly configured.

        Returns:
            True if the provider has a model, base_url, and API key; False otherwise.
        """
        return bool(self.api_key and self.base_url and self.model)

    def __repr__(self) -> str:
        """
        Return a string representation of the ProviderConfig.

        Returns:
            A string summarizing the provider configuration.
        """
        return (
            f"ProviderConfig(name={self.name!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key_env={self.api_key_env!r}, "
            f"configured={self.is_configured()})"
        )

    def unsafe_repr(self) -> str:
        """
        Return a string representation of the ProviderConfig with all fields.
        """
        return (
            f"ProviderConfig(name={self.name!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key_env={self.api_key_env!r}, "
            f"configured={self.is_configured()}, api_key={self.api_key!r})"
        )


@dataclass
class VisionSettings:
    """
    Dataclass for vision-related settings.

    Attributes:
        provider: The default provider configuration (ProviderConfig object).
        retries: Number of retries for vision API calls.
        timeout: Timeout in seconds for vision API calls.
        providers: Mapping of provider names to ProviderConfig objects.
    """

    provider: ProviderConfig
    retries: int
    timeout: int
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)


@dataclass
class FiltrationSettings:
    """
    Dataclass for filtration settings.

    Attributes:
        chart_labels: List of labels used to identify charts/figures in images.
    """

    chart_labels: List[str] = field(default_factory=list)


@dataclass
class ProcessingSettings:
    """
    Dataclass for processing settings.

    Attributes:
        ocr_lang: The language code for OCR processing (e.g., "eng").
    """

    ocr_lang: str = "eng"
    zoom_factor: int = 2


@dataclass
class AppSettings:
    """
    Dataclass for the complete application settings.

    Attributes:
        vision: VisionSettings containing vision provider settings.
        filtration: FiltrationSettings for chart/figure filtering.
        processing: ProcessingSettings for OCR and other processing.
    """

    vision: VisionSettings
    filtration: FiltrationSettings
    processing: ProcessingSettings


class Settings:
    """
    Application settings loader and accessor.

    Loads configuration from a TOML file and supports environment variable overrides.
    Provides access to vision, filtration, and processing settings, as well as
    provider-specific configuration.

    This class implements a singleton pattern to ensure only one instance
    exists throughout the application lifecycle.
    """

    _instance: Optional["Settings"] = None
    _initialized: bool = False

    def __new__(cls, config_path: Path) -> "Settings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Path) -> None:
        """
        Initialize the settings singleton.

        Args:
            config_path: Path to the TOML configuration file.
        """
        if not self._initialized:
            self.config_path = config_path
            self._config: Optional[Dict[str, Any]] = None
            self._load_config()
            self._initialized = True

    def _load_config(self) -> None:
        """
        Load configuration from the TOML file specified by self.config_path.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the TOML file is invalid.
        """
        try:
            with open(self.config_path, "rb") as file:
                self._config = tomllib.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML configuration: {e}")

    def _get_env_override(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a setting from the environment, falling back to a default.

        Args:
            key: The setting key (e.g., "vision_provider").
            default: The default value if the environment variable is not set.

        Returns:
            The value from the environment or the default.
        """
        env_key = f"PAGE_PARSER_{key.upper()}"
        return os.getenv(env_key, default)

    def _parse_provider_config(self, name: str, data: Dict[str, Any]) -> ProviderConfig:
        """
        Parse a single provider's configuration from a dictionary.

        Args:
            name: The provider's name.
            data: The dictionary containing provider configuration.

        Returns:
            A ProviderConfig instance.
        """
        model: str = data.get("model", "")
        base_url: str = data.get("base_url", "")
        api_key_config: Any = data.get("api_key", {})

        if isinstance(api_key_config, dict):
            api_key_type: str = api_key_config.get("type", "env")
            api_key_env: str = api_key_config.get("name", "")
        else:
            api_key_type = "env"
            api_key_env = str(api_key_config) if api_key_config else ""

        if api_key_type == "env":
            api_key = self._get_env_override(api_key_env)
        elif api_key_type == "inline":
            api_key = api_key_config.get("key", "")
        elif api_key_type == "file":
            path = api_key_config.get("path", "")
            if not path:
                raise ValueError("API key file path is required for file type")
            with open(path, "r") as file:
                api_key = file.read().strip()
        else:
            raise ValueError(f"Invalid API key type: {api_key_type}")

        return ProviderConfig(
            name=name,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            api_key_type=api_key_type,
            _api_key=api_key,
        )

    def _parse_providers(
        self, providers_section: Dict[str, Any]
    ) -> Dict[str, ProviderConfig]:
        """
        Parse all providers from the configuration section.

        Args:
            providers_section: Dictionary mapping provider names to their configs.

        Returns:
            A dictionary mapping provider names to ProviderConfig instances.
        """
        return {
            provider_name: self._parse_provider_config(provider_name, provider_data)
            for provider_name, provider_data in providers_section.items()
        }

    @property
    def vision(self) -> VisionSettings:
        """
        Access vision settings, with support for environment variable overrides.

        Returns:
            VisionSettings containing provider, retries, timeout, and providers.

        Raises:
            RuntimeError: If configuration is not loaded.
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        vision_section: Dict[str, Any] = self._config.get("vision", {})

        provider_name: str = self._get_env_override(
            "vision_provider", vision_section.get("provider", "")
        )

        providers_section: Dict[str, Any] = vision_section.get("providers", {})
        providers: Dict[str, ProviderConfig] = self._parse_providers(providers_section)

        # Get the provider config for the specified provider name
        provider_config = providers.get(provider_name)
        if not provider_config:
            # Create a default provider config if the specified provider doesn't exist
            provider_config = ProviderConfig(
                name=provider_name,
                model="",
                base_url="",
                api_key_env="",
                api_key_type="env",
            )

        retries: int = int(
            self._get_env_override("vision_retries", vision_section.get("retries", 3))
        )
        timeout: int = int(
            self._get_env_override("vision_timeout", vision_section.get("timeout", 10))
        )

        return VisionSettings(
            provider=provider_config,
            retries=retries,
            timeout=timeout,
            providers=providers,
        )

    @property
    def filtration(self) -> FiltrationSettings:
        """
        Access filtration settings, with support for environment variable overrides.

        Returns:
            FiltrationSettings containing chart_labels.

        Raises:
            RuntimeError: If configuration is not loaded.
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        filtration_section: Dict[str, Any] = self._config.get("filtration", {})

        chart_labels_env = self._get_env_override("filtration_chart_labels")
        if chart_labels_env:
            chart_labels: List[str] = [
                label.strip() for label in chart_labels_env.split(",")
            ]
        else:
            chart_labels: List[str] = filtration_section.get("chart_labels", [])

        return FiltrationSettings(chart_labels=chart_labels)

    @property
    def processing(self) -> ProcessingSettings:
        """
        Access processing settings, with support for environment variable overrides.

        Returns:
            ProcessingSettings containing ocr_lang.

        Raises:
            RuntimeError: If configuration is not loaded.
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        processing_section: Dict[str, Any] = self._config.get("processing", {})

        ocr_lang: str = self._get_env_override(
            "processing_ocr_lang", processing_section.get("ocr_lang", "eng")
        )

        zoom_factor: int = int(
            self._get_env_override("processing_zoom_factor", processing_section.get("zoom_factor", 2))
        )

        return ProcessingSettings(ocr_lang=ocr_lang, zoom_factor=zoom_factor)

    def get_settings(self) -> AppSettings:
        """
        Retrieve all application settings as a dataclass.

        Returns:
            AppSettings containing vision, filtration, and processing settings.
        """
        return AppSettings(
            vision=self.vision,
            filtration=self.filtration,
            processing=self.processing,
        )

    def get_settings_as_dict(self) -> dict:
        """
        Retrieve all application settings as a dictionary.

        Returns:
            Dictionary containing vision, filtration, and processing settings as dicts.
        """

        def provider_to_dict(provider: ProviderConfig) -> dict:
            return {
                "name": provider.name,
                "model": provider.model,
                "base_url": provider.base_url,
                "api_key_env": provider.api_key_env,
                "api_key_type": provider.api_key_type,
                "configured": provider.is_configured(),
            }

        vision = self.vision
        filtration = self.filtration
        processing = self.processing

        return {
            "vision": {
                "provider": vision.provider.name,
                "retries": vision.retries,
                "timeout": vision.timeout,
                "providers": {
                    name: provider_to_dict(provider)
                    for name, provider in vision.providers.items()
                },
            },
            "filtration": {
                "chart_labels": filtration.chart_labels,
            },
            "processing": {
                "ocr_lang": processing.ocr_lang,
                "zoom_factor": processing.zoom_factor,
            },
        }

    def get_provider(self, name: Optional[str] = None) -> Optional[ProviderConfig]:
        """
        Retrieve a specific provider's configuration.

        Args:
            name: The provider's name. If None, uses the default provider.

        Returns:
            The ProviderConfig for the specified provider, or None if not found.
        """
        providers = self.vision.providers
        provider_name = name or self.vision.provider.name
        return providers.get(provider_name)

    def get_configured_providers(self) -> List[str]:
        """
        Get a list of provider names that are properly configured.

        Returns:
            List of provider names with valid configuration.
        """
        return [
            name
            for name, provider in self.vision.providers.items()
            if provider.is_configured()
        ]

    def reload(self) -> None:
        """
        Reload the configuration from the TOML file.

        This clears the cached configuration and reloads from disk.
        """
        self._config = None
        self._load_config()

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        This should only be used in tests to ensure clean state.
        """
        cls._instance = None
        cls._initialized = False


def get_settings(config_path: Optional[Path] = None) -> Settings:
    """
    Get the singleton Settings instance.

    Args:
        config_path: Path to the TOML settings file. If None, uses DEFAULT_SETTINGS_FILE.

    Returns:
        Settings: The singleton settings instance.
    """
    if config_path is None:
        config_path = DEFAULT_SETTINGS_FILE
    return Settings(config_path)


def load_settings(path: Optional[Path] = None) -> dict:
    """
    Load and parse the application settings from a TOML file.

    This function is provided for backward compatibility and returns the
    settings as a dictionary.

    Args:
        path: Path to the TOML settings file. If None, uses DEFAULT_SETTINGS_FILE.

    Returns:
        dict: Parsed application settings as a dictionary.
    """
    settings = get_settings(path)
    return settings.get_settings_as_dict()

# config_loader.py
import os
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv(".env")
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Replace environment variable placeholders
        return self._resolve_env_vars(config)
    
    def _resolve_env_vars(self, config: Dict) -> Dict:
        """Recursively resolve environment variable placeholders in config"""
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value.get(k, {})
            if value == {}:
                return default
        return value
    
    def validate_required_keys(self, required_keys: list) -> None:
        """Validate that required configuration keys are present"""
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

# Global config instance
config = Config()
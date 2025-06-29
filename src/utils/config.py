import toml as tomllib  # use 'import tomli as tomllib' if Python < 3.11
from pathlib import Path
from functools import lru_cache
import os

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"

@lru_cache()
def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    path = path or Path(os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "rb") as f:
        return tomllib.load(path)

def get_paths():
    cfg = load_config()
    return cfg["paths"]

def get_settings():
    """
    Settings : General
    """
    cfg = load_config()
    return cfg["settings"]

def get_settings_prep():
    """
    Settings : preprocess
    """
    cfg = load_config()
    return cfg["settings_prep"]

def get_settings_stage():
    """
    Settings : staging
    """
    cfg = load_config()
    return cfg["settings_stage"]

def get_settings_bootsrap():
    """
    Settings : bootstrap
    """
    cfg = load_config()
    if not isinstance(cfg["settings_bootstrap"]["channels"], list):
        raise TypeError("Expected Bootsrap Settings - channels to be a list (e.g [0, 1, 5] )")
    return cfg["settings_bootstrap"]

def get_cut_settings():
    """
    Settings : cut files
    """
    cfg = load_config()
    return cfg["cut_settings"]

def setup_logger():
    pass

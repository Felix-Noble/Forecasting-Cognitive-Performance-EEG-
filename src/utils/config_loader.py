from pathlib import Path
from functools import lru_cache

def find_project_root(marker_file_name: str = "pyproject.toml") -> Path:
    """
    Traverses up the directory tree from the current script's location
    to find the project root, identified by the presence of a marker file.

    Args:
        marker_file_name: The name of the file that marks the project root
                          (e.g., 'project.toml', 'pyproject.toml', '.git').

    Returns:
        A pathlib.Path object representing the project root directory.

    Raises:
        FileNotFoundError: If the marker file cannot be found by traversing up.
    """
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir, *current_dir.parents]:
        if (parent / marker_file_name).exists():
            return parent
    raise FileNotFoundError(f"Project root marker '{marker_file_name}' not found "
                            f"in {current_dir} or any parent directories.")

@lru_cache()
def _load_config(path: Path | None = None) -> dict:
    """
    Loads and caches the TOML configuration from the specified path.
    """
    if path is None:
        path = find_project_root() / "config" / "config.toml"
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import toml as tomllib
    except ImportError:
        import tomli as tomllib
    
    return tomllib.load(path)

def get_paths():
    """Load paths relative to project root"""
    cfg = _load_config()
    raw_paths = cfg.get("paths", {})
    root = find_project_root()
    resolved_paths = {}

    for name, path_str in raw_paths.items():
        path_obj = Path(path_str)
        if path_obj.is_absolute():
            # If the path in the config is already absolute, use it directly.
            resolved_paths[name] = path_obj
        else:
            # Otherwise, join it with the project root to make it absolute.
            resolved_paths[name] = root / path_obj

    return resolved_paths

def get_settings():
    """Load settings : general"""
    cfg = _load_config()
    return cfg["settings"]

def get_settings_prep():
    """Load pipeline settings : preprocess"""
    cfg = _load_config()
    return cfg["pipeline"]["preprocessing"]

def get_settings_epoch():
    """ Load pipeline settings : epoch"""
    cfg = _load_config()
    return cfg["pipeline"]["epoch"]

def get_settings_bootstrap():
    """Load pipeline settings : bootstrap"""
    cfg = _load_config()
    if not isinstance(cfg["pipeline"]["bootstrap"]["channels"], list):
        raise TypeError("Expected Bootsrap Settings - channels to be a list (e.g [0, 1, 5] )")
    return cfg["pipeline"]["bootstrap"]


if __name__ == '__main__':
   
    paths = get_paths()

    # Now you can use the paths anywhere in your project
    print(f"Project Root: {find_project_root()}")
    print("-" * 20)
    for name, path in paths.items():
        print(f"{name:<15}: {path}")
        # print(f"{'is absolute?':>15}  {path.is_absolute()}")
        # print()
    
import yaml
from pathlib import Path
from types import SimpleNamespace


def load_config(config_path=None, file_name='config.yaml', encoding='utf-8', flat=True):
    """
    Load YAML configuration file with cross-platform path handling.

    Parameters
    ----------
    config_path : str or Path, optional
        Either a directory path (when used with file_name) or full file path.
        If None, searches in current directory with default file_name.
    file_name : str, default='config.yaml'
        Configuration file name (used when config_path is a directory).
    encoding : str, default='utf-8'
        File encoding to use.
    flat : bool, default=True
        If True, flatten nested sections and return a SimpleNamespace
        for dot access (config.batch_size). If False, return raw dict.

    Returns
    -------
    SimpleNamespace or dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the configuration file doesn't exist.
    yaml.YAMLError
        If the YAML file is malformed.
    """
    if config_path is not None:
        config_path = Path(config_path)

    if config_path is None:
        file_path = Path.cwd() / file_name
    elif config_path.is_dir():
        file_path = config_path / file_name
    else:
        file_path = config_path

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")

    if not flat:
        return raw

    # Flatten nested sections into a single namespace
    flat_dict = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat_dict.update(value)
        else:
            flat_dict[key] = value

    return SimpleNamespace(**flat_dict)
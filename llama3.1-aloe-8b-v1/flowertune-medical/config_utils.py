"""Configuration utilities for overriding settings with environment variables."""
import ast
import os
from logging import WARNING
from typing import Any, MutableMapping

from flwr.common.logger import log


def _try_parse_value(value: str) -> Any:
    """Attempt to parse a string value into a Python literal (e.g., int, bool, float)."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, MemoryError):
        # If parsing fails, return the original string
        return value


def override_config_with_env_vars(
    config: MutableMapping[str, Any], prefix: str
) -> MutableMapping[str, Any]:
    """Override configuration with environment variables.

    This function scans for environment variables starting with a given prefix,
    transforms their names into dot-separated config keys, and overrides the
    values in the provided config dictionary.

    Args:
        config (MutableMapping[str, Any]): The configuration dictionary to override.
        prefix (str): The prefix for environment variables to consider.

    Returns:
        MutableMapping[str, Any]: The updated configuration dictionary.
    """
    for var_name, value in os.environ.items():
        if var_name.startswith(prefix):
            log(
                WARNING,
                "Overriding config with environment variable: %s=%s",
                var_name,
                value,
            )

            # 1. Remove prefix and convert to lowercase
            key_str = var_name[len(prefix) :].lower()

            # 2. Split by double underscore for nesting levels
            key_parts = key_str.split("__")

            # 3. For each part, replace single underscores with hyphens
            final_key_parts = [part.replace("_", "-") for part in key_parts]

            # 4. Join parts with a dot to create the final flat key
            final_key = ".".join(final_key_parts)

            # 5. Parse the value to its likely type (int, float, bool, etc.)
            parsed_value = _try_parse_value(value)

            # 6. Override the value in the config dictionary
            config[final_key] = parsed_value

    return config


def replace_keys(
    input_dict: MutableMapping[str, Any], match: str = "-", target: str = "_"
) -> MutableMapping[str, Any]:
    """Recursively replace characters in dictionary keys.

    Args:
        input_dict: 원본 딕셔너리.
        match: 치환 대상 문자열(기본 "-").
        target: 치환 후 문자열(기본 "_").

    Returns:
        키 내 match가 target으로 치환된 새 딕셔너리.
    """
    new_dict: MutableMapping[str, Any] = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, MutableMapping):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict

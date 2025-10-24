"""
Utility Functions (유틸리티 함수 모듈)

Common utility functions for finding closest values by timestamp
"""

from typing import Dict, Any, Optional


def find_closest_scalar(data_dict: Dict[int, float], target_timestamp: int) -> Optional[float]:
    """
    Find closest scalar value by timestamp

    Args:
        data_dict: Dictionary mapping timestamp to scalar value
        target_timestamp: Target timestamp (nanoseconds)

    Returns:
        Closest scalar value or None if data_dict is empty
    """
    if not data_dict:
        return None

    closest_ts = min(data_dict.keys(), key=lambda ts: abs(ts - target_timestamp))
    return data_dict[closest_ts]


def find_closest_dict(
    data_dict: Dict[int, Dict[str, Any]],
    target_timestamp: int
) -> Optional[Dict[str, Any]]:
    """
    Find closest dictionary value by timestamp

    Args:
        data_dict: Dictionary mapping timestamp to dictionary value
        target_timestamp: Target timestamp (nanoseconds)

    Returns:
        Closest dictionary or None if data_dict is empty
    """
    if not data_dict:
        return None

    closest_ts = min(data_dict.keys(), key=lambda ts: abs(ts - target_timestamp))
    return data_dict[closest_ts]


def extract_dict_values(
    data_dict: Optional[Dict[str, Any]],
    fields: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Helper function to extract multiple fields from dictionary with default values

    Args:
        data_dict: Source dictionary (can be None)
        fields: {field_name: default_value} mapping

    Returns:
        Dictionary with extracted values or defaults
    """
    if not data_dict:
        return {field: default for field, default in fields.items()}

    return {field: data_dict.get(field, default) for field, default in fields.items()}

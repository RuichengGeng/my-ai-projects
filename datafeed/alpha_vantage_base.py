"""
Base Alpha Vantage API client shared across datafeed providers.
"""

import os
import json
import requests

API_BASE_URL = "https://www.alphavantage.co/query"


def get_api_key() -> str:
    """Retrieve the Alpha Vantage API key from environment."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")
    return api_key


class AlphaVantageRateLimitError(Exception):
    """Raised when the Alpha Vantage API rate limit is exceeded."""


def make_api_request(function_name: str, params: dict) -> dict:
    """Call the Alpha Vantage API and return the parsed JSON response.

    Args:
        function_name: Alpha Vantage function name (e.g. "WTI", "COPPER").
        params: Query parameters for this specific endpoint.

    Returns:
        Parsed JSON dictionary.

    Raises:
        AlphaVantageRateLimitError: When the API rate limit is exceeded.
        requests.HTTPError: On HTTP errors.
    """
    api_params = params.copy()
    api_params["function"] = function_name
    api_params["apikey"] = get_api_key()

    response = requests.get(API_BASE_URL, params=api_params)
    response.raise_for_status()

    data = response.json()

    # Alpha Vantage returns error messages under "Information", "Note", or "Error Message"
    if isinstance(data, dict):
        info_msg = data.get("Information", "")
        note_msg = data.get("Note", "")
        error_msg = data.get("Error Message", "")
        combined = f"{info_msg} {note_msg} {error_msg}"
        if "rate limit" in combined.lower() or "api key" in combined.lower():
            raise AlphaVantageRateLimitError(
                f"Alpha Vantage rate limit exceeded: {combined.strip()}"
            )
        if error_msg:
            raise ValueError(f"Alpha Vantage API error: {error_msg}")

    return data


def make_api_request_csv(function_name: str, params: dict) -> str:
    """Call the Alpha Vantage API and return the raw CSV response.

    Args:
        function_name: Alpha Vantage function name.
        params: Query parameters (datatype=csv is added automatically).

    Returns:
        Raw CSV string.

    Raises:
        AlphaVantageRateLimitError: When the API rate limit is exceeded.
    """
    api_params = params.copy()
    api_params["function"] = function_name
    api_params["apikey"] = get_api_key()
    api_params["datatype"] = "csv"

    response = requests.get(API_BASE_URL, params=api_params)
    response.raise_for_status()

    response_text = response.text

    # When datatype=csv, a JSON response indicates an error.
    # Alpha Vantage returns errors as JSON even when datatype=csv is requested.
    try:
        error_data = json.loads(response_text)
        # Empty JSON object `{}` or dict with no error keys = rate-limited demo key
        if isinstance(error_data, dict):
            info_msg = error_data.get("Information", "")
            note_msg = error_data.get("Note", "")
            error_msg = error_data.get("Error Message", "")
            combined = f"{info_msg} {note_msg} {error_msg}"
            if "rate limit" in combined.lower() or "api key" in combined.lower():
                raise AlphaVantageRateLimitError(
                    f"Alpha Vantage rate limit exceeded: {combined.strip()}"
                )
            if error_msg:
                raise ValueError(f"Alpha Vantage API error: {error_msg}")
            # Got valid JSON but no recognized error — likely demo restriction
            if not info_msg and not note_msg and not error_msg:
                raise AlphaVantageRateLimitError(
                    "Alpha Vantage API returned unexpected JSON (likely demo key restriction). "
                    "Claim a free API key at https://www.alphavantage.co/support/#api-key"
                )
    except json.JSONDecodeError:
        pass  # valid CSV, not JSON error

    return response_text

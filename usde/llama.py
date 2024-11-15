from typing import Optional, Dict, Any
import requests
import polars as pl
from typing import List


class DefiLlamaAPIError(Exception):
    """Custom exception for DefiLlama API errors"""

    pass


class StableLlamaAPI:
    """A Python wrapper for the DefiLlama API"""

    BASE_DOMAIN = "llama.fi"

    def __init__(
        self,
        name: str = "stablecoins",
        endpoint: str = "stablecoins",
        timeout: int = 30,
    ):
        """
        Initialize DefiLlama API client

        Args:
            name (str): Type of data to fetch (e.g., "stablecoins")
            timeout (int): Request timeout in seconds
        """
        self.name = name
        self.timeout = timeout
        self.base_url = f"https://{self.name}.{self.BASE_DOMAIN}"

    def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Make HTTP request to DefiLlama API

        Args:
            endpoint (str): API endpoint
            params (dict, optional): Query parameters

        Returns:
            dict: JSON response

        Raises:
            DefiLlamaAPIError: If the API request fails
        """
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise DefiLlamaAPIError(f"API request failed: {str(e)}") from e

    def get_stablecoin_by_id(self, id: int):
        data = self._make_request(endpoint=f"stablecoin/{id}")
        return data

    def _handle_error(self, message: str) -> None:
        """
        Handle errors consistently throughout the class

        Args:
            message (str): Error message to log/print
        """
        # In a production environment, you might want to use proper logging here
        print(message)

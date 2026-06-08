"""
Alpha Vantage commodities data provider.

Covers all commodity endpoints:
  - Gold & Silver: spot prices, historical prices
  - Energy: WTI crude, Brent crude, Natural Gas
  - Metals: Copper, Aluminum
  - Agriculture: Wheat, Corn, Cotton, Sugar, Coffee
  - Index: Global Commodities Index

API reference: https://www.alphavantage.co/documentation/#commodities
"""

from datetime import datetime
from typing import Literal

import pandas as pd
from io import StringIO

from .alpha_vantage_base import make_api_request, make_api_request_csv

# ── interval types ──────────────────────────────────────────────────────
EnergyInterval = Literal["daily", "weekly", "monthly"]
MetalsAgriInterval = Literal["monthly", "quarterly", "annual"]
GoldSilverInterval = Literal["daily", "weekly", "monthly"]


class CommoditiesProvider:
    """Provides commodity price data from Alpha Vantage."""

    # ── Gold & Silver ────────────────────────────────────────────────

    def get_gold_silver_spot(self, symbol: Literal["GOLD", "SILVER"]) -> dict:
        """Get current spot price for gold or silver.

        Args:
            symbol: "GOLD" or "SILVER".

        Returns:
            JSON dict with spot price data.
        """
        return make_api_request("GOLD_SILVER_SPOT", {"symbol": symbol})

    def get_gold_silver_history(
        self,
        symbol: Literal["GOLD", "SILVER"],
        interval: GoldSilverInterval = "daily",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get historical prices for gold or silver.

        Args:
            symbol: "GOLD" or "SILVER".
            interval: "daily", "weekly", or "monthly".
            start_date: Filter results from this date (yyyy-mm-dd).
            end_date: Filter results until this date (yyyy-mm-dd).

        Returns:
            DataFrame with columns: date, open, high, low, close.
        """
        csv_data = make_api_request_csv(
            "GOLD_SILVER_HISTORY",
            {"symbol": symbol, "interval": interval},
        )
        return self._csv_to_dataframe(csv_data, start_date, end_date, rename_cols={
            "timestamp": "date",
            "open (usd)": "open",
            "high (usd)": "high",
            "low (usd)": "low",
            "close (usd)": "close",
        })

    # ── Energy ───────────────────────────────────────────────────────

    def _get_energy_commodity(
        self,
        function_name: str,
        interval: EnergyInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        csv_data = make_api_request_csv(
            function_name, {"interval": interval},
        )
        rename = {"timestamp": "date", "value": "price"}
        return self._csv_to_dataframe(csv_data, start_date, end_date, rename_cols=rename)

    def get_wti(
        self,
        interval: EnergyInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get WTI crude oil prices.

        Args:
            interval: "daily", "weekly", or "monthly".
            start_date: Filter from this date (yyyy-mm-dd).
            end_date: Filter until this date (yyyy-mm-dd).

        Returns:
            DataFrame with date and price columns.
        """
        return self._get_energy_commodity("WTI", interval, start_date, end_date)

    def get_brent(
        self,
        interval: EnergyInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get Brent crude oil prices."""
        return self._get_energy_commodity("BRENT", interval, start_date, end_date)

    def get_natural_gas(
        self,
        interval: EnergyInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get natural gas prices."""
        return self._get_energy_commodity("NATURAL_GAS", interval, start_date, end_date)

    # ── Metals & Agriculture ─────────────────────────────────────────

    def _get_metals_agri_commodity(
        self,
        function_name: str,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        csv_data = make_api_request_csv(
            function_name, {"interval": interval},
        )
        rename = {"timestamp": "date", "value": "price"}
        return self._csv_to_dataframe(csv_data, start_date, end_date, rename_cols=rename)

    def get_copper(
        self,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get copper prices."""
        return self._get_metals_agri_commodity("COPPER", interval, start_date, end_date)

    def get_aluminum(
        self,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get aluminum prices."""
        return self._get_metals_agri_commodity("ALUMINUM", interval, start_date, end_date)

    def get_wheat(
        self,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get wheat prices."""
        return self._get_metals_agri_commodity("WHEAT", interval, start_date, end_date)

    def get_corn(
        self,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get corn prices."""
        return self._get_metals_agri_commodity("CORN", interval, start_date, end_date)

    def get_cotton(
        self,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get cotton prices."""
        return self._get_metals_agri_commodity("COTTON", interval, start_date, end_date)

    def get_sugar(
        self,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get sugar prices."""
        return self._get_metals_agri_commodity("SUGAR", interval, start_date, end_date)

    def get_coffee(
        self,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get coffee prices."""
        return self._get_metals_agri_commodity("COFFEE", interval, start_date, end_date)

    # ── Global Index ─────────────────────────────────────────────────

    def get_all_commodities(
        self,
        interval: MetalsAgriInterval = "monthly",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get the Global Price Index of All Commodities.

        Args:
            interval: "monthly", "quarterly", or "annual".
            start_date: Filter from this date (yyyy-mm-dd).
            end_date: Filter until this date (yyyy-mm-dd).

        Returns:
            DataFrame with date and price (index value) columns.
        """
        return self._get_metals_agri_commodity(
            "ALL_COMMODITIES", interval, start_date, end_date,
        )

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _csv_to_dataframe(
        csv_data: str,
        start_date: str | None = None,
        end_date: str | None = None,
        rename_cols: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Parse a CSV response into a DataFrame with optional date filtering."""
        if not csv_data or not csv_data.strip():
            return pd.DataFrame()

        df = pd.read_csv(StringIO(csv_data))

        if rename_cols:
            df.rename(columns=rename_cols, inplace=True)

        # Parse date column and filter
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])

        if start_date:
            df = df[df[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[date_col] <= pd.to_datetime(end_date)]

        return df.reset_index(drop=True)

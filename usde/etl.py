import polars as pl


import requests
import json
import datetime
import sqlalchemy
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from usde.llama import StableLlamaAPI


class Extractor(StableLlamaAPI):
    def __init__(self, id: int = 146):
        super().__init__()
        self.id = id

    def extract_data(self) -> Dict:
        """Public method to extract data from source"""
        return self._request_data()

    def _request_data(self) -> Dict:
        """Internal method to handle API request"""
        return self.get_stablecoin_by_id(self.id)


class Transformer:
    @staticmethod
    def transform_data(df: pl.DataFrame) -> pl.DataFrame:
        """Public method for complete data transformation"""
        pass

    @staticmethod
    def create_dataframe(raw_data: Dict) -> pl.DataFrame:
        """Public method to create initial DataFrame"""
        chain_dfs = pl.DataFrame()
        for chain_name, chain_data in raw_data["chainBalances"].items():
            df = pl.DataFrame(chain_data["tokens"])
            df = Transformer._process_chain_data(df, chain_name)
            df = Transformer._standardize_columns(df, chain_dfs)
            chain_dfs = chain_dfs.vstack(df)
        return chain_dfs

    @staticmethod
    def _process_chain_data(df: pl.DataFrame, chain_name: str) -> pl.DataFrame:
        """Internal method for processing chain data"""
        df = df.with_columns(
            (pl.col("date") * 1000)
            .cast(pl.Datetime(time_unit="ms"))
            .dt.date()
            .alias("date")
        )

        for column in df.columns:
            if isinstance(df[column].dtype, pl.Struct):
                df = Transformer._flatten_struct(df, column)

        return df.with_columns(pl.lit(chain_name).alias("chain"))

    @staticmethod
    def _flatten_struct(df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Internal helper method for flattening struct columns"""
        dtype = df[column].dtype

        if not isinstance(dtype, pl.Struct):
            return df

        # Get all field names as strings
        fields = [field.name for field in dtype.fields]

        # Create expressions to extract each field
        new_columns = [
            pl.col(column).struct.field(field_name).alias(f"{column}_{field_name}")
            for field_name in fields
        ]

        # Add new columns
        df = df.with_columns(new_columns)

        # Remove original struct column
        df = df.drop(column)

        # Recursively flatten any new struct columns
        for field_name in fields:
            new_col_name = f"{column}_{field_name}"
            if isinstance(df[new_col_name].dtype, pl.Struct):
                df = DataTransformer._flatten_struct(df, new_col_name)

        return df

    @staticmethod
    def _standardize_columns(
        df: pl.DataFrame, reference_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Standardize column types and ensure consistent columns across DataFrames"""

        # Helper function moved inside class
        def is_numeric(dtype):
            return isinstance(
                dtype,
                (
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.UInt8,
                    pl.UInt16,
                    pl.UInt32,
                    pl.UInt64,
                    pl.Float32,
                    pl.Float64,
                ),
            )

        # Convert numeric columns to float64
        numeric_cols = [col for col in df.columns if is_numeric(df[col].dtype)]
        df = df.with_columns([pl.col(col).cast(pl.Float64) for col in numeric_cols])

        if reference_df is not None and not reference_df.is_empty():
            # Add missing columns with null values
            missing_cols = set(reference_df.columns) - set(df.columns)
            if missing_cols:
                df = df.with_columns(
                    [
                        pl.lit(None).cast(reference_df[col].dtype).alias(col)
                        for col in missing_cols
                    ]
                )
            # Select columns in the same order
            df = df.select(reference_df.columns)

        return df


class DataLoader(ABC):
    """Abstract base class for data loading"""

    @abstractmethod
    def create_tables(self):
        pass

    @abstractmethod
    def load_data(self, transformed_df: pl.DataFrame):
        pass


class SQLiteLoader(DataLoader):
    def __init__(self, database_location: str):
        self.database_location = database_location
        self.engine = sqlalchemy.create_engine(database_location)

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        with self.engine.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS my_played_tracks(
                    song_name VARCHAR(200),
                    artist_name VARCHAR(200),
                    played_at VARCHAR(200),
                    timestamp VARCHAR(200),
                    CONSTRAINT primary_key_constraint PRIMARY KEY (played_at)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fav_artist(
                    timestamp VARCHAR(200),
                    ID VARCHAR(200),
                    artist_name VARCHAR(200),
                    count VARCHAR(200),
                    CONSTRAINT primary_key_constraint PRIMARY KEY (ID)
                )
            """
            )

    def load_data(self, transformed_df: pl.DataFrame):
        """Load transformed data into SQLite"""

        try:
            transformed_df.to_sql(
                "fav_artist", self.engine, index=False, if_exists="append"
            )
        except Exception as e:
            print(f"Error loading transformed data: {e}")


class ETLPipeline:
    def __init__(self, loader: DataLoader):
        self.extractor = Extractor()
        self.transformer = Transformer()
        self.loader = loader

    def run(self):
        """Execute the complete ETL pipeline"""
        try:
            # Extract
            raw_data = self.extractor.extract_data()

            # Transform
            raw_df = self.transformer.create_dataframe(raw_data)
            if self.transformer.validate_data(raw_df):
                transformed_df = self.transformer.transform_data(raw_df)

                # Load
                self.loader.create_tables()
                self.loader.load_data(raw_df, transformed_df)
                print("ETL process completed successfully")

        except Exception as e:
            print(f"ETL process failed: {e}")

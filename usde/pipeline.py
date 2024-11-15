import polars as pl


import requests
import json
import datetime
import sqlalchemy
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from usde.llama import StableLlamaAPI


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

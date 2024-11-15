import json
from usde.etl import USDEExtractor

if __name__ == "__main__":
    extractor = USDEExtractor()
    df = extractor.create_dataframe()
    print(df)

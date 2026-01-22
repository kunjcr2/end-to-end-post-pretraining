import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import RobustScaler

class DataProcess:

    """
    DataProcess class is used to preprocess the data
    """

    def __init__(self, path="dataset.csv"):
        """
        Initialize the data process
        """
        self.path = path
        self.df = pd.read_csv(path)

        self.logger = logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.scalar = RobustScaler()
    
    def getData(self):
        """
        Returns the data
        """
        self.logger.info("Getting the data")
        return self._preprocess(self.df)

    def _preprocess(self, df):
        """
        Preprocess the data
        """
        self.logger.info("Preprocessing the data")

        df.drop(columns=["Unnamed: 0"], inplace=True)
        df["timestamp"] = df["timestamp"].map(lambda x: pd.to_datetime(x))
        df["is_weekend"] = df["timestamp"].map(lambda x: int(x.dayofweek >= 5))
        df["slot"] = df["timestamp"].map(lambda x: x.hour*2 + (x.minute == 30))

        df = self._getSinCos(df, "slot", 48)
        df = self._getSinCos(df, "timestamp", 12)

        df["lag_1"] = df["value"].shift(1)
        df["lag_48"] = df["value"].shift(48)
        df["delta_1"] = df["value"] - df["lag_1"]
        df["delta_48"] = df["value"] - df["lag_48"]

        df = self._scaling(df, "value")
        df = self._scaling(df, "delta_1")
        df = self._scaling(df, "delta_48")

        df.dropna(inplace=True)
        df.drop(columns=["timestamp"], inplace=True)

        mean = df["value"].mean()
        std = df["value"].std()

        df["anomaly"] = (
            (df["value"] > (mean + 2 * std)) |
            (df["value"] < (mean - 2 * std))
        ).astype(int)

        df = self._scaling(df)

        self.logger.info("Data preprocessing completed")

        return df

    def _getSinCos(self, df, col, period):
        """
        Get sin and cos values
        """
        self.logger.info("Getting sin and cos values")

        df[col] = np.sin(2 * np.pi * df[col] / period)
        df[col] = np.cos(2 * np.pi * df[col] / period)

        return df

    def _scaling(self, df, col):
        """
        Scale the data
        """
        self.logger.info("Scaling the data")

        self.scalar.fit(df[col])
        df[col] = self.scalar.transform(df[col])

        return df

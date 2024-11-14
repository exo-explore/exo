# inference.py

import json
import os
import numpy as np
from zipfile import ZipFile
from typing import Optional, Tuple
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
import pickle
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from .updater import (
    download_binance_daily_data,
    download_binance_current_day_data,
    download_coingecko_data,
    download_coingecko_current_day_data
)# Import your configuration settings

class CoinPredictionInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader: ShardDownloader):
        self.shard = None
        self.shard_downloader = shard_downloader
        self.model = 'LinearRegression'
        self.training_price_data_path = '~/Downloads/data.csv'
        self.trained_model_path = '~/Downloads/'
        self.token = 'ETH'  # To keep track of the current token
        self.timeframe = '1D'  # Default timeframe```
        self.region = 'US'  # Default region
        self.data_provider = 'binance'  # Default data provider

    async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> (bool):
        return False

    async def infer_prompt(
        self,
        request_id: str,
        shard: Shard,
        prompt: str,
        image_str: Optional[str] = None,
        inference_state: Optional[str] = None
    ) -> Tuple[np.ndarray, str, bool]:
        await self.ensure_shard(shard)
        
        # Extract token symbol from prompt
        token = prompt.strip()
        
        # Update token if it has changed
        if self.token != token:
            self.token = token

        try:
            predicted_price = self.get_inference()
            output_data = np.array([[predicted_price]])
            return output_data, inference_state or '', True  # True indicates completion
        except Exception as e:
            print(f"Error during inference: {e}")
            return np.array([]), inference_state or '', True

    async def ensure_shard(self, shard: Shard):
        self.update_data()
        self.shard = shard

    def get_inference(self):
        # Load current data
        
        current_day_data = download_binance_current_day_data(f"{self.token}USDT", self.region)
        
        X_new = self.load_frame(current_day_data, self.timeframe)
        print(X_new.tail())
        print(X_new.shape)

        #self.ensure_shard()
        current_price_pred = self.model.predict(X_new)
        return current_price_pred[0]

    def update_data(self):
        """Download price data, format data and train model."""
        os.makedirs(os.path.dirname(self.trained_model_path), exist_ok=True)
        if self.data_provider == "binance":
            files = sorted([x for x in os.listdir(self.trained_model_path) if x.startswith(f"{self.token}USDT")])
        elif self.data_provider == "coingecko":
            files = sorted([x for x in os.listdir(self.trained_model_path) if x.endswith(".json")])
        self.format_data(files, self.data_provider, self.training_price_data_path)
        self.train_model(self.timeframe, self.model, self.training_price_data_path, self.trained_model_path)

    def format_data(self, files, data_provider, data_path):
        if not files:
            print("Already up to date")
            return
        
        if data_provider == "binance":
            files = sorted([x for x in os.listdir(data_path) if x.startswith(f"{self.token}USDT")])
        elif data_provider == "coingecko":
            files = sorted([x for x in os.listdir(data_path) if x.endswith(".json")])

        # No files to process
        if len(files) == 0:
            return

        price_df = pd.DataFrame()
        if data_provider == "binance":
            for file in files:
                zip_file_path = os.path.join(data_path, file)

                if not zip_file_path.endswith(".zip"):
                    continue

                myzip = ZipFile(zip_file_path)
                with myzip.open(myzip.filelist[0]) as f:
                    line = f.readline()
                    header = 0 if line.decode("utf-8").startswith("open_time") else None
                df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
                df.columns = [
                    "start_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "end_time",
                    "volume_usd",
                    "n_trades",
                    "taker_volume",
                    "taker_volume_usd",
                ]
                df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
                df.index.name = "date"
                price_df = pd.concat([price_df, df])

                price_df.sort_index().to_csv(data_path)
        elif data_provider == "coingecko":
            for file in files:
                with open(os.path.join(data_path, file), "r") as f:
                    data = json.load(f)
                    df = pd.DataFrame(data)
                    df.columns = [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close"
                    ]
                    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.drop(columns=["timestamp"], inplace=True)
                    df.set_index("date", inplace=True)
                    price_df = pd.concat([price_df, df])

                price_df.sort_index().to_csv(data_path)

    def load_frame(self, frame, timeframe):
        print(f"Loading data...")
        df = frame.loc[:,['open','high','low','close']].dropna()
        df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric)
        df['date'] = frame['date'].apply(pd.to_datetime)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

    def train_model(self, timeframe, MODEL, data_path, trained_model_path):
        # Load the price data
        price_data = pd.read_csv(data_path)
        df = self.load_frame(price_data, timeframe)

        print(df.tail())

        y_train = df['close'].shift(-1).dropna().values
        X_train = df[:-1]

        print(f"Training data shape: {X_train.shape}, {y_train.shape}")

        # Define the model
        if MODEL == "LinearRegression":
            model = LinearRegression()
        elif MODEL == "SVR":
            model = SVR()
        elif MODEL == "KernelRidge":
            model = KernelRidge()
        elif MODEL == "BayesianRidge":
            model = BayesianRidge()
        # Add more models here
        else:
            raise ValueError("Unsupported model")
        
        # Train the model
        model.fit(X_train, y_train)

        # create the model's parent directory if it doesn't exist
        os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)

        # Save the trained model to a file
        with open(trained_model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Trained model saved to {trained_model_path}")


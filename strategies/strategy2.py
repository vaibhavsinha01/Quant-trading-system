import numpy as np
import pandas as pd
from SmartApi import SmartConnect
from datetime import datetime, timedelta
from indicators.volatility import VolatilityIndicator
from indicators.trend import TrendIndicators
from indicators.momentum import MomentumIndicator
from indicators.volume import VolumeIndicator
from broker.angel import AngelBrokerWrapper
import creds
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# here use all the indicators that are there - momentum,trend,volume,volatility,extra,candles to train for the last 100 candles - if posssible apply neural network(dl),xgboostclassifer(ml).

class Strategy2:
    def __init__(self):
        self.api_key = creds.api_key
        self.username = creds.username
        self.password = creds.password
        self.token = creds.token
        self.broker = AngelBrokerWrapper(api_key=self.api_key, username=self.username, password=self.password, token=self.token)
        self.trendI = TrendIndicators()
        self.momenI = MomentumIndicator()
        self.volatI = VolatilityIndicator()
        self.volumI = VolumeIndicator()
        self.model = None

    def fetch_data(self):
        try:
            current_time = datetime.now()
            market_open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            start_time = market_open_time - timedelta(days=creds.use_data_for_x_days)
            self.df = self.broker.get_candle_data(
                creds.exchange,
                creds.token_id,
                creds.timeframe,
                start_time.strftime("%Y-%m-%d %H:%M"),
                current_time.strftime("%Y-%m-%d %H:%M")
            )
            print("The data is fetched and the current df is \n")
            print(self.df)
            print("\n")
            # Ensure renaming is assigned back to self.df
            self.df = self.df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            if self.df is None or self.df.empty:
                print("Fetched data is empty. Check data source.")
            else:
                print(f"Data fetched successfully: {self.df.tail()}")

        except Exception as e:
            print(f"Error in fetch_data: {e}")

    def feature_engineering(self):
        # Add indicators based on the entire dataframe for feature extraction
        self.df = self.volatI.directional_volatility(self.df)
        self.df['cci'] = self.momenI.cci(self.df)
        self.df['tsv'] = self.volumI.tsv(self.df)
        self.df['macd_ema'], self.df['macd_signal'], self.df['histogram'] = self.trendI.macd(self.df)
        self.df['rsi'] = self.momenI.rsi(self.df)
        self.df['support'],self.df['resistance'],self.df['trend'] = self.trendI.trend_lines(self.df)
        self.df['middlebb'],self.df['upperbb'],self.df['lowerbb'],self.df['bandwidthbb'],self.df['percent_bb'] = self.volatI.bbands(self.df)
        print(self.df)

    def prepare_ml_data(self):
        # Calculate the percentage change and shift it by 1 to predict the next value
        self.df['pct_change'] = np.sign(self.df['close'].pct_change())
            
        # Shift the target by 1 row (next period's percentage change) to predict the next movement
        self.df['target'] = self.df['pct_change'].shift(-1)

        # Drop rows with missing values (due to the shift operation)
        self.df.dropna(axis=0, inplace=True)

        # Features and target split
        features = ['macd_ema', 'macd_signal', 'cci', 'tsv', 'histogram','rsi','support','resistance','middlebb','upperbb','lowerbb','bandwidthbb','percent_bb']
        X = self.df[features]
        y = self.df['target']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets (with the time sequence intact)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        return X_train, X_test, y_train, y_test


    def objective(self, trial):
        # Optuna optimization for Random Forest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       random_state=42)

        # Get data split for training and testing
        X_train, X_test, y_train, y_test = self.prepare_ml_data()
        
        # Fit the model
        model.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy

    def optimize_model(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=50)
        
        print(f"Best trial: {study.best_trial.params}")
        return study.best_trial.params

    def train_ml_model(self):
        # Train the model with optimized parameters from Optuna
        best_params = self.optimize_model()
        self.model = RandomForestClassifier(**best_params)
        X_train, X_test, y_train, y_test = self.prepare_ml_data()
        self.model.fit(X_train, y_train)

    def generate_signals(self):
        if self.model is None:
            print("Model is not trained yet!")
            return

        # Get the last row as the feature set for prediction
        last_candle = self.df.iloc[-1][['macd_ema', 'macd_signal', 'cci', 'tsv', 'histogram','rsi','support','resistance','middlebb','upperbb','lowerbb','bandwidthbb','percent_bb']].values.reshape(1, -1)
        
        # Predict the signal using the last candle (last row) data
        predicted_signal = self.model.predict(last_candle)
        
        print(f"Predicted signal for the last candle: {predicted_signal[0]}")

        # Store the predicted signal in the dataframe for analysis
        self.df.loc[self.df.index[-1], 'predicted_signal'] = predicted_signal[0]

    def execute_signals(self):
        if 'predicted_signal' not in self.df.columns:
            print("No predicted signal available!")
            return

        if self.df['predicted_signal'].iloc[-1] > 0:
            print("A buy signal will be placed")
            # self.broker.place_order()
        elif self.df['predicted_signal'].iloc[-1] < 0:
            print("A sell signal will be placed")
            # self.broker.place_order()
        else:
            print("No trade would be currently placed")

    def run(self):
        self.fetch_data()
        self.feature_engineering()
        self.train_ml_model()  # Train the model
        self.generate_signals()
        self.execute_signals()

if __name__ == "__main__":
    s1 = Strategy2()
    s1.run()

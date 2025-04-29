import pandas as pd
import numpy as np

class PortfolioManager:
    def __init__(self):
        pass

    def compute_daily_returns(self, price_df):
        return price_df.pct_change().dropna()

    def compute_beta(self, stock_returns, market_returns):
        cov = np.cov(stock_returns, market_returns)
        return cov[0, 1] / cov[1, 1]  # Cov(stock, market) / Var(market)

    def allocate_inverse_volatility(self, price_data_dict, market_df):
        """
        price_data_dict: dict of { 'TCS': df, 'INFY': df, ... } with each df having ['date', 'close']
        market_df: df with ['date', 'close'] of NIFTY or any index
        """
        market_df = market_df.copy()
        market_df = market_df[['date', 'close']].rename(columns={'close': 'market_close'})
        market_df['date'] = pd.to_datetime(market_df['date'])

        weights = {}
        betas = {}
        volatilities = {}

        for ticker, df in price_data_dict.items():
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            merged = pd.merge(df, market_df, on='date', how='inner')

            stock_returns = merged['close'].pct_change().dropna()
            market_returns = merged['market_close'].pct_change().dropna()

            # Align lengths
            min_len = min(len(stock_returns), len(market_returns))
            stock_returns = stock_returns[-min_len:]
            market_returns = market_returns[-min_len:]

            beta = self.compute_beta(stock_returns, market_returns)
            volatility = stock_returns.std()

            betas[ticker] = beta
            volatilities[ticker] = volatility

        # Inverse volatility weights
        inv_vol = {ticker: 1/vol for ticker, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        weights = {ticker: iv / total_inv_vol for ticker, iv in inv_vol.items()}

        return weights, betas, volatilities


if __name__ == "__main__":
    # Simulated Data
    dates = pd.date_range(start='2023-01-01', periods=200)

    def generate_stock_df(name, noise=1.0):
        np.random.seed(hash(name) % 2**32)
        prices = np.cumsum(np.random.randn(200) * noise + 0.2) + 100
        return pd.DataFrame({'date': dates, 'close': prices})

    stock_data = {
        'TCS': generate_stock_df('TCS', noise=1.2),
        'INFY': generate_stock_df('INFY', noise=1.0),
        'RELIANCE': generate_stock_df('RELIANCE', noise=1.5),
    }

    nifty_df = generate_stock_df('NIFTY', noise=0.9)

    pm = PortfolioManager()
    weights, betas, vols = pm.allocate_inverse_volatility(stock_data, nifty_df)

    print("== Portfolio Allocation Based on Inverse Volatility ==")
    for ticker in weights:
        print(f"{ticker}: Weight = {weights[ticker]:.2%}, Beta = {betas[ticker]:.2f}, Volatility = {vols[ticker]:.4f}")

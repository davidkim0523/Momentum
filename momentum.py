import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class Momentum():
    def __init__(self, prices, lookback_window, n_selection, cost):   
        self.returns = self.get_returns(prices)        
        self.abs_mom = self.absolute_momentum(prices, lookback_window)
        self.rel_mom = self.relative_momentum(prices, lookback_window, n_selection)
        self.dual_mom = self.dual_momentum(prices, lookback_window, n_selection)
        self.ma = self.ma(prices, lookback_window)

    def get_returns(self, prices):
        return prices.pct_change().dropna()
        
    def absolute_momentum(self, prices, lookback):
        log_returns = np.log(prices / prices.shift(lookback))
        long_signal = (log_returns > 0).applymap(self.bool_converter)
        short_signal = -(log_returns < 0).applymap(self.bool_converter)
        signal = long_signal + short_signal
        return signal
    
    def relative_momentum(self, prices, lookback, n_selection):
        log_returns = np.log(prices / prices.shift(lookback))
        rank = log_returns.rank(axis=1, ascending=False)
        long_signal = (rank <= n_selection).applymap(self.bool_converter)
        short_signal = -(rank >= len(rank.columns) - n_selection + 1).applymap(self.bool_converter)
        signal = long_signal + short_signal 
        return signal
    
    def dual_momentum(self, prices, lookback, n_selection):
        abs_signal = self.absolute_momentum(prices, lookback)
        rel_signal = self.relative_momentum(prices, lookback, n_selection)
        
        signal = (abs_signal == rel_signal).applymap(self.bool_converter) * abs_signal

        return signal

    def inverse_volatility(self, returns, lookback):
        vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        inv_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        weight = pd.DataFrame(index=returns.index, columns=returns.columns)

        for ticker in returns.columns:
            vol[ticker] = returns[ticker].rolling(lookback).std() * np.sqrt(lookback)
            inv_vol[ticker] = 1 / vol[ticker]

        for ticker in returns.columns:
            weight[ticker] = inv_vol[ticker] / inv_vol.sum(axis=1)

        return weight

    def cost_structure(self, signal, cost):
        cost_df = (signal.diff() != 0).applymap(self.bool_converter).iloc[1:, :] * cost
        return cost_df

    def ma(self, prices, lookback):

        moving_avg = prices.rolling(lookback).mean()
        long_signal = (prices > moving_avg).applymap(self.bool_converter)
        short_signal = -(prices < moving_avg).applymap(self.bool_converter)
        signal = long_signal + short_signal

        return signal

    def bool_converter(self, bool_var):
        if bool_var == True:
            result = 1
        elif bool_var == False:
            result = 0
        return result

    def straddle_delta(self, returns, lookback_window):
        mu = returns.rolling(lookback_window).mean()
        sigma = returns.rolling(lookback_window).std()
        tstat = mu / sigma * np.sqrt(lookback_window)
        confidence = 2 * (tstat.apply(norm.cdf)) - 1
        return confidence.fillna(0)



if __name__ == "__main__":
    
    url = 'https://raw.githubusercontent.com/davidkim0523/Momentum/main/Data.csv'
    df = pd.read_csv(url).dropna()
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Date'])
    df = df.resample('W').mean()

    lookback_window = 4
    cost = 0.000
    n_selection = int(len(df.columns)/2)
    
    momentum = Momentum(df, lookback_window, n_selection, cost)

    returns = momentum.returns
    abs_signal = momentum.abs_mom
    rel_signal = momentum.rel_mom
    dual_signal = momentum.dual_mom
    straddle_delta = momentum.straddle_delta(returns, lookback_window)

    emv_weight = momentum.inverse_volatility(returns, lookback_window).fillna(0)
    ew_weight = 1 / len(returns.columns)
    abs_cost = momentum.cost_structure(abs_signal, cost)
    rel_cost = momentum.cost_structure(rel_signal, cost)
    dual_cost = momentum.cost_structure(dual_signal, cost)
    straddle_delta_cost = momentum.cost_structure(straddle_delta, cost)

    plt.plot((1 + (dual_signal.shift(1) * emv_weight.shift(1) * (returns - dual_cost)).sum(axis=1)).cumprod() - 1, label='EMV Dual Mom')
    plt.plot((1 + (abs_signal.shift(1) * emv_weight.shift(1) * (returns - abs_cost)).sum(axis=1)).cumprod() - 1, label='EMV Absolute Mom')
    plt.plot((1 + (rel_signal.shift(1) * emv_weight.shift(1) * (returns - rel_cost)).sum(axis=1)).cumprod() - 1, label='EMV Relative Mom')
#    plt.plot((1 + (dual_signal.shift(1) * ew_weight * (returns - dual_cost)).sum(axis=1)).cumprod() - 1, label='EW Dual Mom')
#    plt.plot((1 + (abs_signal.shift(1) * ew_weight * (returns - abs_cost)).sum(axis=1)).cumprod() - 1, label='EW Absolute Mom')
#    plt.plot((1 + (rel_signal.shift(1) * ew_weight * (returns - rel_cost)).sum(axis=1)).cumprod() - 1, label='EW Relative Mom')

    plt.legend()
    plt.show()

# Machine Learning Engineer Nanodegree
# Capstone Proposal

25 Dec 2019

Stock market prediction has been identified as a very important practical problem in the economic field. Over 80% of trading in stock market and FOREX market was performed by trading algorithms rather than humans. In the crypto-currency market, algorithmic trading is also a hot topic among investors. However, timely and accurate prediction of the market is generally regarded as one of the most challenging problems, since the environment is highly affected by volatile political-economic factors, such as legislative acts, political unrest, and mass irrational panic.

There are many studies regarding algorithmic trading in financial markets taking advantage of machine learning, where recurrent neural network (RNN) and reinforcement learning (RL) are being popular in recent years. In this study, a Bitcoin price predictor based on long short-term memory (LSTM, a variant of RNN) is presented.

## Problem Statement

The goal is to build a price predictor of Bitcoin price.

## Datasets and Inputs

In this study, data of the BitMEX XBTZ19 contract, which is a Bitcoin futures contract expiring in December 2019, is used to train and test the predictor. Input data of the predictor is time series data $\{x_t|t=1,2,\dots,T\}$, where $x_t$ is a vector of trading data in a specific minute, containing open-high-low-close prices, volume, and Volume-weighted average price (VWAP).

Data could be fetched from BitMEX's API (`https://www.bitmex.com/api/explorer/#!/Trade/Trade_getBucketed`) without charge of fee.

## Solution Statement

Given data of a time range with length of N, $\{x_{i-N+1}\dots x_i\}$, the predictor outputs the VWAP of the next minute $y_i=v_{i+1}$, where $v_i$ is an item of the vector $x_i$ indicating the VWAP $x_i=(\dots, v_i)$. The predictor can be seen as a function $\hat y_i=f(x_{i-N+1}\dots x_i)$ mapping from the history trading data to the next VWAP.

## Benchmark Model

A linear regression model will be used as the benchmark to compare against the LSTM model.

## Evaluation Metrics

The mean squared error (MSE) between labels $y$ and predictions $\hat y$ will be used to evaluate both of the benchmark model and the solution model. For a given $N$ and a time-series test data-set, all consequetive sub-sequence of the time-series with length $N$ will be used and equally contibute to the final MSE.

## Project Design

The data could be fetched from BitMEX's official API. The details are listed as follows.

```
endpoint:   https://www.bitmex.com/api/v1/trade/bucketed
parameters:
    symbol:  "XBTZ19"
    binSize: "1m"
```

The API returns 12 fields for each time-step (or minute in this case). We only need 5 of them as features for the machine learning model, which are high price, low price, close price, volume, VWAP. The open price returned by BitMEX API is defined as the close price of the last time step, instead of the price of the first trade during the specific time, and therefore will not be taken as a feature. Other fields, such as instrument symbol, are irrelevant and will be abandoned.

Technical indicators, like moving average convergence/divergence (MACD), relative strength index (RSI), will be used as additional features.

<!-- [^Bigiotti]: Bigiotti, Alessandro; Navarra, Alfredo (October 19, 2018), "Optimizing Automated Trading Systems", Advances in Intelligent Systems and Computing, Springer International Publishing, pp. 254–261, doi:10.1007/978-3-030-02351-5_30, ISBN 978-3-030-02350-8 -->

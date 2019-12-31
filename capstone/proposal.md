# Machine Learning Engineer Nanodegree

# Capstone Proposal

25 December 2019

The stock market prediction has been identified as a significant practical problem in the economic field. Over 80% of trading in the stock market and FOREX market was performed by trading algorithms rather than humans [^bigiotti]. In the crypto-currency market, algorithmic trading is also a hot topic among investors. However, timely and accurate prediction of the market is generally regarded as one of the most challenging problems, since the environment is profoundly affected by volatile political-economic factors, such as legislative acts, political unrest, and mass irrational panic.

There are many studies regarding algorithmic trading in financial markets based on machine learning, where recurrent neural network (RNN) and reinforcement learning (RL) are being popular in recent years. In this study, a Bitcoin price predictor based on long short-term memory (LSTM, a variant of RNN) is presented.

## Problem Statement

The goal is to build a price predictor of Bitcoin price.

## Datasets and Inputs

In this study, data of the BitMEX XBTZ19 contract, which is a Bitcoin futures contract expiring in December 2019, is used to train and test the predictor. Input data of the predictor is time-series data $\{x_t|t=1,2,\dots,T\}$, where $x_t$ is a vector of trading data in a specific minute, containing open-high-low prices, volume, and Volume-weighted average price (VWAP).

Data could be fetched from BitMEX's API (`https://www.bitmex.com/api/explorer/#!/Trade/Trade_getBucketed`) without charge of fee.

## Solution Statement

Given data of a time range with length of N, $\{x_{i-N+1}\dots x_i\}$, the predictor outputs the VWAP of the next minute $y_i=v_{i+1}$, where $v_i$ is an item of the vector $x_i$ indicating the VWAP $x_i=(\dots, v_i)$. The predictor can be seen as a function $\hat y_i=f(x_{i-N+1}\dots x_i)$ mapping from the history trading data to the next VWAP.

## Benchmark Model

A linear regression model will be used as the benchmark to compare against the LSTM model.

## Evaluation Metrics

The mean squared error (MSE) between labels $y$ and predictions $\hat y$ will be used to evaluate both of the benchmark model and the solution model. For a given N and a time-series test dataset, all consecutive sub-sequence of the time-series with length $N$ will be used and equally contribute to the final MSE.

## Project Design

The data could be fetched from BitMEX's official API. The details are listed as follows.

```
endpoint:   https://www.bitmex.com/api/v1/trade/bucketed
parameters:
    symbol:  "XBTZ19"
    binSize: "1m"
```

The API returns 12 fields for each time-step (or minute in this case). We only need 5 of them as features for the machine learning model, which are high price, low price, close price, volume, VWAP. The open price returned by BitMEX API is defined as the close price of the last time step instead of the price of the first trade during the specific time, and therefore will not be taken as a feature. Other fields, such as instrument symbol, are irrelevant and will be abandoned.

Technical indicators, like moving average convergence/divergence (MACD), relative strength index (RSI), will be used as additional features.

Consisting of five original columns and technical indicator columns as features, the entire time-series will be split into three parts for training, validation, and testing, with lengths proportionate to 6:2:2.

The training dataset will be sent to train the benchmark model and the solution model. Both models are implemented with PyTorch and trained with AWS SageMaker.

The benchmark model consists of two linear regression layers:
$$x_1=xA_1+b_1$$
$$y=ReLU(x_1)A_2+b_2$$

The solution model consists of one LSTM layer and one linear layer. The LSTM layer is formally formulated as follows.

\begin{array}{ll} \\
    i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
    f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
    g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
    o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
    c_t = f_t * c_{(t-1)} + i_t * g_t \\
    h_t = o_t * \tanh(c_t) \\
\end{array}

where where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell state at time `t`, :math:`x_t` is the input at time `t`, and :math:`i_t`, :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell, and output gates, respectively. :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

After training, the MSE between the original and predicted VWAP would be calculated. Besides, a chart of real and predicted VWAP will be plotted.

<!-- [^bigiotti]: Bigiotti, Alessandro; Navarra, Alfredo (October 19, 2018), "Optimizing Automated Trading Systems", Advances in Intelligent Systems and Computing, Springer International Publishing, pp. 254–261, doi:10.1007/978-3-030-02351-5_30, ISBN 978-3-030-02350-8 -->

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def macd(prices: pd.Series, n_fast=12, n_slow=26) -> pd.DataFrame:
    EMAfast = prices.ewm(span=n_fast, min_periods=n_slow).mean()
    EMAslow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = EMAfast - EMAslow
    MACDsign = MACD.ewm(span=9, min_periods=9).mean()
    MACDdiff = MACD - MACDsign
    return pd.DataFrame(
        {
            "macd_" + str(n_fast) + "_" + str(n_slow): MACD,
            "macd_sign_" + str(n_fast) + "_" + str(n_slow): MACDsign,
            "macd_diff_" + str(n_fast) + "_" + str(n_slow): MACDdiff,
        }
    )


def rsi(high: pd.Series, low: pd.Series, n=14) -> pd.Series:
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 < low.shape[0]:
        UpMove = high.iloc[i + 1] - high.iloc[i]
        DoMove = low.iloc[i] - low.iloc[i + 1]
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    PosDI = pd.Series(UpI, index=low.index).ewm(span=n, min_periods=n).mean()
    NegDI = pd.Series(DoI, index=low.index).ewm(span=n, min_periods=n).mean()
    RSI = pd.Series(PosDI / (PosDI + NegDI), name="rsi_" + str(n))
    return RSI


def preprocess(df: pd.DataFrame, input_dir: str) -> None:
    df_x = pd.DataFrame(
        {
            "vwap_log": df.vwap.apply(pd.np.log),
            "high_log": df.high.apply(pd.np.log),
            "low_log": df.low.apply(pd.np.log),
            "close_log": df.close.apply(pd.np.log),
            "trades_log": (df.trades + 1).apply(pd.np.log),
            "home_notional_log": (df.homeNotional + 1).apply(pd.np.log),
            "foreign_notional_log": (df.foreignNotional + 1).apply(pd.np.log),
        }
    )
    df_x.vwap_log.fillna(method="ffill", inplace=True)
    df_x = df_x.join(macd(df.vwap))
    df_x = df_x.join(rsi(df.high, df.low))
    while pd.isna(df_x.iloc[0].macd_sign_12_26):
        df_x = df_x.iloc[1:]

    scaler = StandardScaler()
    df_x = pd.DataFrame(
        scaler.fit_transform(df_x.to_numpy()), columns=df_x.columns, index=df_x.index
    )

    df_y = pd.Series(df_x.vwap_log.shift(-1), name="next")

    trai_x, vali_x = train_test_split(df_x, test_size=0.4, shuffle=False)
    vali_x, test_x = train_test_split(vali_x, test_size=0.5, shuffle=False)
    trai_y, vali_y = train_test_split(df_y, test_size=0.4, shuffle=False)
    vali_y, test_y = train_test_split(vali_y, test_size=0.5, shuffle=False)

    trai_x.to_csv(os.path.join(input_dir, "trai_x.csv"))
    vali_x.to_csv(os.path.join(input_dir, "vali_x.csv"))
    test_x.to_csv(os.path.join(input_dir, "test_x.csv"))
    trai_y.to_csv(os.path.join(input_dir, "trai_y.csv"), header=True)
    vali_y.to_csv(os.path.join(input_dir, "vali_y.csv"), header=True)
    test_y.to_csv(os.path.join(input_dir, "test_y.csv"), header=True)
    pickle.dump(scaler, open(os.path.join(input_dir, "scaler.pkl"), "wb"))

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def macd(df: pd.DataFrame, n_fast=12, n_slow=26) -> pd.DataFrame:
    EMAfast = df["vwap"].ewm(span=n_fast, min_periods=n_slow).mean()
    EMAslow = df["vwap"].ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = pd.Series(EMAfast - EMAslow, name="macd_" + str(n_fast) + "_" + str(n_slow))
    MACDsign = pd.Series(
        MACD.ewm(span=9, min_periods=9).mean(),
        name="macd_sign_" + str(n_fast) + "_" + str(n_slow),
    )
    MACDdiff = pd.Series(
        MACD - MACDsign, name="macd_diff_" + str(n_fast) + "_" + str(n_slow)
    )
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df


def rsi(df: pd.DataFrame, n=14) -> pd.DataFrame:
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 < df.shape[0]:
        UpMove = df.iloc[i + 1].high - df.iloc[i].high
        DoMove = df.iloc[i].low - df.iloc[i + 1].low
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
    UpI = pd.Series(UpI, index=df.index)
    DoI = pd.Series(DoI, index=df.index)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name="rsi_" + str(n))
    df = df.join(RSI)
    return df


def preprocess(df: pd.DataFrame, input_dir: str) -> None:
    df_x = df.drop(["symbol", "volume", "open", "lastSize"], axis="columns")
    while pd.isna(df_x.iloc[0].vwap):
        df_x = df_x.iloc[1:]
    df_x["vwap"].fillna(method="ffill", inplace=True)
    df_x = macd(df_x)
    df_x = rsi(df_x)
    while pd.isna(df_x.iloc[0].macd_sign_12_26):
        df_x = df_x.iloc[1:]

    scaler = StandardScaler()
    df_x = pd.DataFrame(
        scaler.fit_transform(df_x.values), columns=df_x.columns, index=df_x.index
    )

    df_y = pd.Series(df_x["vwap"].shift(-1), name="next")
    df_x = df_x[:-1]
    df_y = df_y[:-1]

    x_trai, x_vali = train_test_split(df_x, test_size=0.4, shuffle=False)
    x_vali, x_test = train_test_split(x_vali, test_size=0.5, shuffle=False)
    y_trai, y_vali = train_test_split(df_y, test_size=0.4, shuffle=False)
    y_vali, y_test = train_test_split(y_vali, test_size=0.5, shuffle=False)

    x_trai.to_csv(os.path.join(input_dir, "trai_x.csv"))
    x_vali.to_csv(os.path.join(input_dir, "vali_x.csv"))
    x_test.to_csv(os.path.join(input_dir, "test_x.csv"))
    y_trai.to_csv(os.path.join(input_dir, "trai_y.csv"), header=True)
    y_vali.to_csv(os.path.join(input_dir, "vali_y.csv"), header=True)
    y_test.to_csv(os.path.join(input_dir, "test_y.csv"), header=True)
    pickle.dump(scaler, open(os.path.join(input_dir, "scaler.pkl"), "wb"))

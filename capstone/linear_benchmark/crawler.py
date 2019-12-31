import os
import urllib.request
import urllib.parse

import pandas as pd


def fetch_from_bitmex(subject: str, param: dict) -> pd.DataFrame:
    url = f"https://www.bitmex.com/api/v1/{subject}?{urllib.parse.urlencode(param)}"
    res = urllib.request.urlopen(url)
    df = pd.read_json(res.read(), convert_dates=True)
    return df


def fetch_bucketed_from_bitmex(symbol: str, bin_size: str) -> pd.DataFrame:
    """
    symbol:   Instrument symbol
    bin_size: [1m,5m,1h,1d]. Time interval to bucket by.
    """
    cache_file_path = os.path.join("./cache_crawler", f"{symbol}_{bin_size}.csv")
    try:
        df = pd.read_csv(cache_file_path, index_col="timestamp", parse_dates=True)
        cursor = df.shape[0]
    except FileNotFoundError:
        df = None
        cursor = 0
    while True:
        df_p = fetch_from_bitmex(
            "trade/bucketed",
            {"symbol": symbol, "binSize": bin_size, "count": 1000, "start": cursor},
        )
        if df_p.shape[0] == 0:
            break
        df_p.set_index("timestamp", inplace=True)
        if df is None:
            df_p.to_csv(cache_file_path)
        else:
            df_p.to_csv(cache_file_path, mode="a", header=False)
        df = pd.concat([df, df_p])
        cursor = df.shape[0]
        print(".", end="")
    print()
    return df

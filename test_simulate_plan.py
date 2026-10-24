"""تست واحد شبیه‌ساز: ورود باید open کندلِ بعد از سیگنال باشد، نه close همان کندل."""
import numpy as np
import pandas as pd

from walk_forward_trade_plan import simulate_plan


def _df(n=30):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.full(n, 100.0)
    return pd.DataFrame({
        "open": np.full(n, 100.0), "high": np.full(n, 100.5), "low": np.full(n, 99.5),
        "close": close, "ATR": np.full(n, 1.0), "ADX_14": np.full(n, 40.0),
        "funding_rate": np.zeros(n), "symbol": "TEST"}, index=idx)


def test_entry_is_next_open():
    df = _df()
    df.iloc[6, df.columns.get_loc("open")] = 101.0   # open کندل بعد از سیگنال
    up = np.zeros(len(df)); up[5] = 1.0              # سیگنال در کندل 5
    dn = np.zeros(len(df))
    tr = simulate_plan(df, up, dn, 0.5, 0.5, 30, 1.5, 2.0, 4, 0.0, False)
    assert tr, "باید معامله ساخته شود"
    assert tr[0]["entry_time"] == df.index[6]
    assert tr[0]["entry"] == 101.0


def test_no_entry_on_last_candle():
    df = _df()
    up = np.zeros(len(df)); up[-1] = 1.0             # سیگنال در آخرین کندل: کندل بعدی وجود ندارد
    tr = simulate_plan(df, up, np.zeros(len(df)), 0.5, 0.5, 30, 1.5, 2.0, 4, 0.0, False)
    assert tr == []


if __name__ == "__main__":
    test_entry_is_next_open(); test_no_entry_on_last_candle(); print("✅ تست‌ها گذشت")

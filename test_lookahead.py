"""
test_lookahead.py
-----------------
تست خودکار look-ahead: هر ویژگی در کندل t باید با «داده تا t» برابر باشد.
روش: ویژگی‌ها را یک‌بار روی کل داده و یک‌بار روی داده‌ی بریده‌شده تا t می‌سازیم؛
اگر مقدار ردیف t فرق کرد → آن ستون از آینده استفاده می‌کند.

اجرا:
    python test_lookahead.py --symbol BTCUSDT --interval 1h --rows 3000
(به data_cache/{symbol}_{interval}.csv نیاز دارد)
"""
import argparse
import sys

import numpy as np
import pandas as pd

from features import build_features, get_feature_columns


def check(df_raw: pd.DataFrame, cut_points, **kw) -> dict:
    full = build_features(df_raw, **kw)
    cols = get_feature_columns(full)
    bad = {}
    for t in cut_points:
        part = build_features(df_raw.iloc[: t + 1], **kw)
        ts = df_raw.index[t]
        if ts not in part.index or ts not in full.index:
            continue
        a = full.loc[ts, cols].astype(float)
        b = part.loc[ts, cols].reindex(cols).astype(float)
        diff = ~np.isclose(a.to_numpy(), b.to_numpy(), rtol=1e-6, atol=1e-9, equal_nan=True)
        for c in a.index[diff]:
            bad.setdefault(c, []).append(str(ts))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--cuts", type=int, default=8)
    a = ap.parse_args()

    df = pd.read_csv(f"data_cache/{a.symbol}_{a.interval}.csv",
                     index_col="timestamp", parse_dates=True).tail(a.rows)
    cuts = np.linspace(len(df) // 2, len(df) - 60, a.cuts).astype(int)
    bad = check(df, cuts, include_smart_money=True, include_chart_patterns=True,
                include_scores=True, include_absolute_levels=True)
    if bad:
        print("❌ ستون‌های مشکوک به look-ahead:")
        for c, ts in sorted(bad.items(), key=lambda kv: -len(kv[1])):
            print(f"   {c}: در {len(ts)} از {len(cuts)} نقطه‌ی تست تفاوت داشت")
        sys.exit(1)
    print("✅ هیچ ستونی look-ahead نشان نداد.")


if __name__ == "__main__":
    main()

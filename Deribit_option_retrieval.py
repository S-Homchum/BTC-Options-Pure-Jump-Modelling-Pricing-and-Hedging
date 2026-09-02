import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import time
import warnings
warnings.filterwarnings("ignore")


DERIBIT_BASE = "https://www.deribit.com/api/v2"
DERIBIT_HIST = "https://history.deribit.com/api/v2"

_RESOLUTION_MINUTES = {
    "1": 1, "3": 3, "5": 5, "10": 10, "15": 15, "30": 30,
    "60": 60, "120": 120, "180": 180, "360": 360, "720": 720,
    "1D": 1440,
}

_VALID_EXPIRY_TYPES = {"daily", "weekly", "monthly", "quarterly"}

DAYS_PER_YEAR = 365.0


def last_friday_of_month(year, month):
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day)
    offset = (dt.weekday() - 4) % 7
    return datetime(year, month, last_day - offset, 8, 0, 0)


def generate_expiry_dates(expiry_type, start_date, end_date):
    if expiry_type not in _VALID_EXPIRY_TYPES:
        raise ValueError(
            f"Invalid expiry_type '{expiry_type}'. "
            f"Choose from: {sorted(_VALID_EXPIRY_TYPES)}"
        )

    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=90)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=400)
    expiries = []

    if expiry_type == "daily":
        current = start_dt.replace(hour=8, minute=0, second=0, microsecond=0)
        while current <= end_dt:
            expiries.append(current)
            current += timedelta(days=1)

    elif expiry_type == "weekly":
        current = start_dt
        days_to_friday = (4 - current.weekday()) % 7
        current = (current + timedelta(days=days_to_friday)).replace(
            hour=8, minute=0, second=0, microsecond=0
        )
        while current <= end_dt:
            expiries.append(current)
            current += timedelta(weeks=1)

    elif expiry_type == "monthly":
        y, m = start_dt.year, start_dt.month
        while True:
            exp = last_friday_of_month(y, m)
            if exp > end_dt:
                break
            if exp >= start_dt:
                expiries.append(exp)
            m += 1
            if m > 12:
                m, y = 1, y + 1

    elif expiry_type == "quarterly":
        for y in range(start_dt.year, end_dt.year + 1):
            for qm in (3, 6, 9, 12):
                exp = last_friday_of_month(y, qm)
                if start_dt <= exp <= end_dt:
                    expiries.append(exp)

    return sorted(expiries)


def get_index_price(start_date, end_date, resolution="5", sleep_sec=0.3):
    if resolution not in _RESOLUTION_MINUTES:
        raise ValueError(
            f"Invalid resolution '{resolution}'. "
            f"Choose from: {list(_RESOLUTION_MINUTES.keys())}"
        )

    candle_min = _RESOLUTION_MINUTES[resolution]
    batch_td = timedelta(minutes=candle_min * 5000)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    url = f"{DERIBIT_BASE}/public/get_tradingview_chart_data"
    all_frames = []
    current_start = start_dt
    batch_num = 0

    print(f"[1/4] Fetching BTC index at resolution={resolution} ...")

    while current_start < end_dt:
        current_end = min(current_start + batch_td, end_dt)
        params = {
            "instrument_name": "BTC-PERPETUAL",
            "resolution": resolution,
            "start_timestamp": int(current_start.timestamp() * 1000),
            "end_timestamp": int(current_end.timestamp() * 1000),
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  Error fetching index data: {e}")
            time.sleep(2)
            current_start = current_end
            continue

        result = data.get("result", {})
        ticks = result.get("ticks", [])

        if ticks:
            all_frames.append(pd.DataFrame({
                "timestamp": ticks,
                "open": result.get("open", []),
                "high": result.get("high", []),
                "low": result.get("low", []),
                "close": result.get("close", []),
                "volume": result.get("volume", []),
            }))
            batch_num += 1
            if batch_num % 5 == 0:
                print(f"  Index batch {batch_num}: "
                      f"{current_start:%Y-%m-%d} to {current_end:%Y-%m-%d}")

        current_start = current_end
        time.sleep(sleep_sec)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"close": "perp_close"}, inplace=True)

    print(f"  Total index candles: {len(df)}")
    print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df


def parse_instrument(name):
    parts = str(name).split("-")
    if len(parts) != 4:
        return None
    currency, expiry_str, strike_str, opt_type = parts
    if currency != "BTC" or opt_type not in ("C", "P"):
        return None
    try:
        expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(hour=8)
        return {
            "expiry_dt": expiry_dt,
            "strike": float(strike_str),
            "option_type": opt_type,
        }
    except (ValueError, TypeError):
        return None


def fetch_option_trades_chunk(start_ms, end_ms, count=10000, sleep_sec=0.15,
                              max_retries=4):
    url = f"{DERIBIT_HIST}/public/get_last_trades_by_currency_and_time"
    all_trades = []
    cursor_start = start_ms
    retries = 0

    while cursor_start < end_ms:
        params = {
            "currency": "BTC",
            "kind": "option",
            "start_timestamp": cursor_start,
            "end_timestamp": end_ms,
            "count": count,
            "sorting": "asc",
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            retries = 0
        except Exception as e:
            retries += 1
            print(f"    API error ({retries}/{max_retries}): {e}")
            if retries >= max_retries:
                print("    Giving up on this chunk.")
                break
            time.sleep(2 * retries)
            continue

        result = data.get("result", {})
        trades = result.get("trades", [])
        if not trades:
            break

        all_trades.extend(trades)

        if result.get("has_more", False):
            cursor_start = trades[-1]["timestamp"] + 1
            time.sleep(sleep_sec)
        else:
            break

    return all_trades


def fetch_all_option_trades(start_date, end_date, chunk_hours=6,
                            sleep_sec=0.15, save_csv=True,
                            csv_path="deribit_btc_option_trades_raw.csv"):
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    chunk_td = timedelta(hours=chunk_hours)

    print(f"\n[2/4] Fetching ALL BTC option trades ({start_date} to {end_date}) ...")

    all_trades = []
    current = start_dt
    chunk_num = 0

    while current < end_dt:
        chunk_end = min(current + chunk_td, end_dt)
        trades = fetch_option_trades_chunk(
            int(current.timestamp() * 1000),
            int(chunk_end.timestamp() * 1000),
            sleep_sec=sleep_sec,
        )
        all_trades.extend(trades)
        chunk_num += 1

        if chunk_num % 20 == 0:
            print(f"  Chunk {chunk_num}: {current:%Y-%m-%d %H:%M} "
                  f"| cumulative trades: {len(all_trades)}")

        current = chunk_end
        time.sleep(sleep_sec)

    print(f"\n  Total raw trades fetched: {len(all_trades)}")
    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)

    parsed = df["instrument_name"].apply(parse_instrument)
    valid_mask = parsed.apply(lambda x: x is not None)
    df = df[valid_mask].copy()
    parsed_df = pd.DataFrame(parsed[valid_mask].tolist(), index=df.index)

    df["expiry_dt"] = parsed_df["expiry_dt"]
    df["strike"] = parsed_df["strike"]
    df["option_type"] = parsed_df["option_type"]

    df["trade_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["price_btc"] = pd.to_numeric(df["price"], errors="coerce")
    df["index_price"] = pd.to_numeric(df["index_price"], errors="coerce")
    df["price_usd"] = df["price_btc"] * df["index_price"]

    df["mark_price_btc"] = pd.to_numeric(
        df.get("mark_price", pd.Series(index=df.index, dtype=float)), errors="coerce")
    df["mark_price_usd"] = df["mark_price_btc"] * df["index_price"]

    iv_raw = pd.to_numeric(df.get("iv", pd.Series(index=df.index, dtype=float)),
                           errors="coerce")
    df["iv"] = iv_raw / 100.0

    df["amount"] = pd.to_numeric(df.get("amount", np.nan), errors="coerce")
    df["direction"] = df.get("direction", np.nan)

    df["tte_days"] = (df["expiry_dt"] -
                      df["trade_time"].dt.tz_localize(None)).dt.total_seconds() / 86400.0
    df["moneyness"] = df["strike"] / df["index_price"]

    print(f"  Parsed option trades: {len(df)}")
    print(f"  Date range: {df['trade_time'].min()} to {df['trade_time'].max()}")
    print(f"  Unique instruments: {df['instrument_name'].nunique()}")
    print(f"  Calls / Puts: {(df['option_type'] == 'C').sum()} / "
          f"{(df['option_type'] == 'P').sum()}")

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"  Saved raw trades: {csv_path}")

    return df


def build_option_panel(option_df, resolution="5",
                       expiry_type=None,
                       option_type=None,
                       min_tte_days=0.0, max_tte_days=None,
                       moneyness_range=None,
                       min_price_btc=0.0,
                       start_date=None, end_date=None):
    if resolution not in _RESOLUTION_MINUTES:
        raise ValueError(
            f"Invalid resolution '{resolution}'. "
            f"Choose from: {list(_RESOLUTION_MINUTES.keys())}"
        )

    print(f"\n[3/4] Building full option panel at resolution={resolution} ...")

    df = option_df.copy()
    n0 = len(df)

    if option_type is not None:
        df = df[df["option_type"] == option_type]

    if expiry_type is not None:
        if start_date is None or end_date is None:
            raise ValueError("expiry_type filtering needs start_date and end_date.")
        expiry_set = set(generate_expiry_dates(expiry_type, start_date, end_date))
        df = df[df["expiry_dt"].apply(
            lambda x: x.replace(hour=8, minute=0, second=0, microsecond=0) in expiry_set
        )]

    if min_tte_days is not None:
        df = df[df["tte_days"] >= min_tte_days]
    if max_tte_days is not None:
        df = df[df["tte_days"] <= max_tte_days]

    if moneyness_range is not None:
        lo, hi = moneyness_range
        df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]

    if min_price_btc > 0:
        df = df[df["price_btc"] > min_price_btc]

    print(f"  Trades kept after filters: {len(df)} of {n0}")
    if df.empty:
        print("[!] Nothing left after filtering.")
        return pd.DataFrame()

    freq_str = f"{_RESOLUTION_MINUTES[resolution]}min" if resolution != "1D" else "1D"
    df["bin_time"] = df["trade_time"].dt.floor(freq_str)

    df.sort_values(["bin_time", "instrument_name", "timestamp"], inplace=True)

    grp = df.groupby(["bin_time", "instrument_name"], sort=False)

    panel = grp.agg(
        index_price=("index_price", "last"),
        option_type=("option_type", "last"),
        strike=("strike", "last"),
        expiry_dt=("expiry_dt", "last"),
        price_btc=("price_btc", "last"),
        price_usd=("price_usd", "last"),
        mark_price_btc=("mark_price_btc", "last"),
        mark_price_usd=("mark_price_usd", "last"),
        iv=("iv", "last"),
        vwap_btc=("price_btc", "mean"),
        volume_contracts=("amount", "sum"),
        n_trades_in_bin=("price_btc", "size"),
    ).reset_index()

    panel.rename(columns={"bin_time": "timestamp"}, inplace=True)

    panel["tte_days"] = (
        panel["expiry_dt"] - panel["timestamp"].dt.tz_localize(None)
    ).dt.total_seconds() / 86400.0
    panel["tte_years"] = panel["tte_days"] / DAYS_PER_YEAR

    panel["moneyness"] = panel["strike"] / panel["index_price"]
    panel["log_moneyness"] = np.log(panel["moneyness"])

    panel.sort_values(["timestamp", "expiry_dt", "strike", "option_type"],
                      inplace=True)
    panel.reset_index(drop=True, inplace=True)

    print(f"  Panel rows: {len(panel)}")
    print(f"  Distinct time bins: {panel['timestamp'].nunique()}")
    print(f"  Distinct strikes: {panel['strike'].nunique()}")
    print(f"  Distinct expiries: {panel['expiry_dt'].nunique()}")
    print(f"  Median strikes per bin: "
          f"{panel.groupby('timestamp').size().median():.0f}")

    return panel


def attach_spot(panel, index_df, resolution="5", use_perp_fallback=False):
    print(f"\n[4/4] Attaching spot series ...")

    panel = panel.copy()
    panel["index_source"] = np.where(panel["index_price"].notna(),
                                     "trade", "missing")

    if index_df is None or index_df.empty:
        panel["perp_close"] = np.nan
        return panel

    freq_str = f"{_RESOLUTION_MINUTES[resolution]}min" if resolution != "1D" else "1D"
    idx = index_df[["timestamp", "perp_close"]].copy()
    idx["timestamp"] = idx["timestamp"].dt.floor(freq_str)
    idx = idx.drop_duplicates(subset="timestamp", keep="last")

    merged = pd.merge(panel, idx, on="timestamp", how="left")

    if use_perp_fallback:
        fill_mask = merged["index_price"].isna() & merged["perp_close"].notna()
        merged.loc[fill_mask, "index_price"] = merged.loc[fill_mask, "perp_close"]
        merged.loc[fill_mask, "index_source"] = "perp_proxy"
        if fill_mask.any():
            print(f"  [!] {fill_mask.sum()} rows filled with the PERPETUAL price, "
                  f"not the index. Tagged index_source='perp_proxy'.")

    n_trade = (merged["index_source"] == "trade").sum()
    print(f"  Spot from trade index: {n_trade} of {len(merged)}")
    if (merged["index_source"] == "missing").any():
        print(f"  Missing spot: {(merged['index_source'] == 'missing').sum()}")

    return merged


def index_basis_check(panel):
    if "perp_close" not in panel.columns:
        print("  No perp_close column — nothing to compare.")
        return pd.DataFrame()

    d = panel[(panel.get("index_source", "trade") == "trade") &
              panel["index_price"].notna() & panel["perp_close"].notna()]
    d = d.drop_duplicates(subset="timestamp")

    if d.empty:
        print("  No overlapping bins to compare.")
        return pd.DataFrame()

    basis = (d["perp_close"] - d["index_price"]) / d["index_price"]

    print(f"  Bins compared      : {len(d)}")
    print(f"  Mean basis         : {basis.mean():+.4%}")
    print(f"  Median basis       : {basis.median():+.4%}")
    print(f"  Std dev            : {basis.std():.4%}")
    print(f"  5th / 95th pct     : {basis.quantile(0.05):+.4%} / "
          f"{basis.quantile(0.95):+.4%}")
    print(f"  Worst absolute     : {basis.abs().max():.4%}")
    print("  -> This is the error you take on any row using perp as index.")

    return pd.DataFrame({"timestamp": d["timestamp"].values,
                         "index_price": d["index_price"].values,
                         "perp_close": d["perp_close"].values,
                         "basis": basis.values})


def get_index_series_from_trades(start_date, end_date, resolution="60",
                                 instrument="BTC-PERPETUAL",
                                 sleep_sec=0.12, max_calls=20000,
                                 save_csv=True,
                                 csv_path="deribit_btc_index_true.csv"):
    if resolution not in _RESOLUTION_MINUTES:
        raise ValueError(f"Invalid resolution '{resolution}'.")

    step = timedelta(minutes=_RESOLUTION_MINUTES[resolution])
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    n_bins = int((end_dt - start_dt) / step)
    print(f"\nReconstructing TRUE index from {instrument} trades ...")
    print(f"  Bins to fetch: {n_bins} (one API call each)")
    if n_bins > max_calls:
        raise ValueError(
            f"{n_bins} bins exceeds max_calls={max_calls}. Use a coarser "
            f"resolution, a shorter window, or raise max_calls."
        )

    url = f"{DERIBIT_HIST}/public/get_last_trades_by_instrument_and_time"
    rows = []
    current = start_dt
    i = 0

    while current < end_dt:
        bin_end = min(current + step, end_dt)
        params = {
            "instrument_name": instrument,
            "start_timestamp": int(current.timestamp() * 1000),
            "end_timestamp": int(bin_end.timestamp() * 1000),
            "count": 1,
            "sorting": "desc",     # last trade in the bin
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            trades = r.json().get("result", {}).get("trades", [])
            if trades:
                t = trades[0]
                rows.append({
                    "timestamp": pd.Timestamp(current, tz="UTC"),
                    "index_price_true": t.get("index_price"),
                    "perp_price": t.get("price"),
                    "trade_ts": pd.to_datetime(t.get("timestamp"), unit="ms", utc=True),
                })
        except Exception as e:
            print(f"  Error at {current}: {e}")
            time.sleep(1.5)

        i += 1
        if i % 500 == 0:
            print(f"  {i}/{n_bins} bins | collected {len(rows)}")

        current = bin_end
        time.sleep(sleep_sec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["basis"] = (df["perp_price"] - df["index_price_true"]) / df["index_price_true"]

    print(f"  Bins with an index value: {len(df)} of {n_bins}")
    print(f"  Median perp basis: {df['basis'].median():+.4%}")

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    return df


def attach_true_index(panel, true_index_df, resolution="60"):
    if true_index_df is None or true_index_df.empty:
        return panel

    freq_str = f"{_RESOLUTION_MINUTES[resolution]}min" if resolution != "1D" else "1D"
    idx = true_index_df[["timestamp", "index_price_true"]].copy()
    idx["timestamp"] = idx["timestamp"].dt.floor(freq_str)
    idx = idx.drop_duplicates(subset="timestamp", keep="last")

    out = pd.merge(panel, idx, on="timestamp", how="left")
    mask = out["index_price"].isna() & out["index_price_true"].notna()
    out.loc[mask, "index_price"] = out.loc[mask, "index_price_true"]
    out.loc[mask, "index_source"] = "index_reconstructed"

    out["moneyness"] = out["strike"] / out["index_price"]
    out["log_moneyness"] = np.log(out["moneyness"])

    print(f"  Rows filled from reconstructed index: {int(mask.sum())}")
    return out


def estimate_forwards(panel, min_pairs=3):
    calls = panel[panel["option_type"] == "C"][
        ["timestamp", "expiry_dt", "strike", "price_usd", "tte_years"]]
    puts = panel[panel["option_type"] == "P"][
        ["timestamp", "expiry_dt", "strike", "price_usd"]]

    pairs = pd.merge(calls, puts, on=["timestamp", "expiry_dt", "strike"],
                     suffixes=("_c", "_p"))
    if pairs.empty:
        print("  No matched call/put pairs — cannot estimate forwards.")
        return pd.DataFrame()

    pairs["cp_diff"] = pairs["price_usd_c"] - pairs["price_usd_p"]

    rows = []
    for (ts, exp), g in pairs.groupby(["timestamp", "expiry_dt"]):
        if len(g) < min_pairs or g["strike"].nunique() < 2:
            continue
        slope, intercept = np.polyfit(g["strike"].values, g["cp_diff"].values, 1)
        df_factor = -slope
        if df_factor <= 0:
            continue
        forward = intercept / df_factor
        tte_y = float(g["tte_years"].iloc[0])
        rate = -np.log(df_factor) / tte_y if tte_y > 0 and df_factor > 0 else np.nan
        rows.append({
            "timestamp": ts,
            "expiry_dt": exp,
            "forward": forward,
            "discount_factor": df_factor,
            "implied_rate": rate,
            "n_pairs": len(g),
        })

    out = pd.DataFrame(rows)
    print(f"  Forwards estimated for {len(out)} (timestamp, expiry) pairs.")
    return out


def _btc_price_from_forward(F, K, T, sigma, opt_type):
    from math import log, sqrt, erf

    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    vs = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / vs
    d2 = d1 - vs

    if opt_type == "C":
        return _norm_cdf(d1) - (K / F) * _norm_cdf(d2)
    return (K / F) * _norm_cdf(-d2) - _norm_cdf(-d1)


def _solve_forward(price_btc, K, T, sigma, opt_type,
                   lo_mult=0.01, hi_mult=100.0, tol=1e-10, max_iter=100):
    if not np.isfinite([price_btc, K, T, sigma]).all():
        return np.nan
    if price_btc <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan

    lo, hi = K * lo_mult, K * hi_mult

    try:
        f_lo = _btc_price_from_forward(lo, K, T, sigma, opt_type) - price_btc
        f_hi = _btc_price_from_forward(hi, K, T, sigma, opt_type) - price_btc
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.nan

    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return np.nan   # price not attainable — bad quote or bad iv

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        try:
            f_mid = _btc_price_from_forward(mid, K, T, sigma, opt_type) - price_btc
        except (ValueError, ZeroDivisionError, OverflowError):
            return np.nan
        if abs(f_mid) < tol or (hi - lo) / mid < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)


def estimate_forwards_from_iv(panel, price_col="price_btc", iv_col="iv",
                              moneyness_band=None, min_quotes=1,
                              aggregate="median", keep_per_quote=False):
    df = panel.copy()

    needed = {price_col, iv_col, "strike", "tte_years", "option_type",
              "index_price", "timestamp", "expiry_dt"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")

    if moneyness_band is not None:
        lo, hi = moneyness_band
        df = df[(df["moneyness"] >= lo) & (df["moneyness"] <= hi)]

    df = df[(df[price_col] > 0) & (df[iv_col] > 0) & (df["tte_years"] > 0)]
    df = df[df[[price_col, iv_col, "strike", "tte_years", "index_price"]]
            .notna().all(axis=1)]

    if df.empty:
        print("  No usable quotes for IV inversion.")
        return (pd.DataFrame(), pd.DataFrame()) if keep_per_quote else pd.DataFrame()

    print(f"  Inverting {len(df)} quotes for the forward ...")

    df["forward_quote"] = [
        _solve_forward(p, k, t, s, o)
        for p, k, t, s, o in zip(df[price_col], df["strike"], df["tte_years"],
                                 df[iv_col], df["option_type"])
    ]

    n_failed = df["forward_quote"].isna().sum()
    df = df[df["forward_quote"].notna()]
    print(f"  Inverted: {len(df)} | failed: {n_failed}")

    if df.empty:
        return (pd.DataFrame(), pd.DataFrame()) if keep_per_quote else pd.DataFrame()

    agg = df.groupby(["timestamp", "expiry_dt"]).agg(
        forward=("forward_quote", aggregate),
        forward_std=("forward_quote", "std"),
        n_quotes=("forward_quote", "size"),
        index_price=("index_price", "median"),
        tte_years=("tte_years", "median"),
    ).reset_index()

    agg = agg[agg["n_quotes"] >= min_quotes]

    agg["discount_factor"] = agg["index_price"] / agg["forward"]
    agg["implied_rate"] = np.log(agg["forward"] / agg["index_price"]) / agg["tte_years"]
    agg["forward_cv"] = agg["forward_std"] / agg["forward"]

    print(f"  Forwards from IV: {len(agg)} (timestamp, expiry) pairs.")
    if len(agg):
        print(f"  Median dispersion across strikes: "
              f"{agg['forward_cv'].median():.4%} "
              f"(high values mean stale quotes in that bin)")

    if keep_per_quote:
        return agg, df[["timestamp", "expiry_dt", "instrument_name", "strike",
                        "option_type", "forward_quote"]]
    return agg


def attach_forward_per_row(panel, price_col="price_btc", iv_col="iv",
                           add_slice_stats=True):
    out = panel.copy()

    for c in (price_col, iv_col, "strike", "tte_years", "option_type"):
        if c not in out.columns:
            raise ValueError(f"panel is missing required column: {c}")

    if price_col != "price_btc":
        print(f"  [!] price_col='{price_col}'. Deribit's trade `iv` belongs to "
              f"`price`, so anything else here is internally inconsistent.")

    print(f"\nInverting forward per row from '{price_col}' ({len(out)} rows) ...")
    out["forward"] = [
        _solve_forward(p_, k, t, s_, o)
        for p_, k, t, s_, o in zip(out[price_col], out["strike"],
                                   out["tte_years"], out[iv_col],
                                   out["option_type"])
    ]

    n_ok = out["forward"].notna().sum()
    print(f"  Inverted: {n_ok} | failed (NaN): {len(out) - n_ok}")

    out["discount_factor"] = out["index_price"] / out["forward"]
    out["implied_rate"] = np.log(out["forward"] / out["index_price"]) / out["tte_years"]

    if add_slice_stats:
        grp = out.groupby(["timestamp", "expiry_dt"])["forward"]
        out["forward_slice_med"] = grp.transform("median")
        out["forward_slice_cv"] = grp.transform("std") / out["forward_slice_med"]
        out["forward_dev"] = (out["forward"] - out["forward_slice_med"]) / out["forward_slice_med"]
        cv = out["forward_slice_cv"].dropna()
        if len(cv):
            print(f"  Cross-strike dispersion within a slice: median {cv.median():.4%}")
            print("  -> filter on forward_slice_cv / forward_dev to drop stale quotes")

    return out


def get_chain_snapshot(panel, timestamp):
    ts = pd.Timestamp(timestamp)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")

    available = panel["timestamp"].unique()
    if len(available) == 0:
        return pd.DataFrame()

    nearest = min(available, key=lambda t: abs(pd.Timestamp(t) - ts))
    snap = panel[panel["timestamp"] == nearest].copy()
    snap.sort_values(["expiry_dt", "strike", "option_type"], inplace=True)
    print(f"  Snapshot at {nearest}: {len(snap)} quotes, "
          f"{snap['strike'].nunique()} strikes, {snap['expiry_dt'].nunique()} expiries.")
    return snap.reset_index(drop=True)


def get_live_chain(currency="BTC", save_csv=True,
                   csv_path="deribit_live_chain.csv"):
    print(f"\nFetching live {currency} option chain ...")
    url = f"{DERIBIT_BASE}/public/get_book_summary_by_currency"

    try:
        r = requests.get(url, params={"currency": currency, "kind": "option"},
                         timeout=30)
        r.raise_for_status()
        result = r.json().get("result", [])
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)

    parsed = df["instrument_name"].apply(parse_instrument)
    valid_mask = parsed.apply(lambda x: x is not None)
    df = df[valid_mask].copy()
    parsed_df = pd.DataFrame(parsed[valid_mask].tolist(), index=df.index)

    df["expiry_dt"] = parsed_df["expiry_dt"]
    df["strike"] = parsed_df["strike"]
    df["option_type"] = parsed_df["option_type"]

    now = pd.Timestamp.utcnow()
    df["timestamp"] = now

    for col in ["mark_price", "bid_price", "ask_price", "underlying_price",
                "mark_iv", "interest_rate", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["forward_price"] = pd.to_numeric(df.get("underlying_price", np.nan),
                                        errors="coerce")
    df["underlying_index"] = df.get("underlying_index", np.nan)
    df["index_price"] = df["forward_price"]   # see caveat above
    df["mark_price_btc"] = df.get("mark_price", np.nan)
    df["mark_price_usd"] = df["mark_price_btc"] * df["index_price"]
    df["iv"] = df.get("mark_iv", np.nan) / 100.0

    df["mid_btc"] = (df.get("bid_price", np.nan) + df.get("ask_price", np.nan)) / 2.0
    df["mid_usd"] = df["mid_btc"] * df["index_price"]
    df["spread_btc"] = df.get("ask_price", np.nan) - df.get("bid_price", np.nan)

    df["tte_days"] = (df["expiry_dt"] - now.tz_localize(None)).dt.total_seconds() / 86400.0
    df["tte_years"] = df["tte_days"] / DAYS_PER_YEAR
    df["moneyness"] = df["strike"] / df["index_price"]
    df["log_moneyness"] = np.log(df["moneyness"])

    keep = ["timestamp", "instrument_name", "option_type", "strike", "expiry_dt",
            "tte_days", "tte_years", "index_price", "forward_price",
            "underlying_index", "moneyness", "log_moneyness",
            "bid_price", "ask_price", "mid_btc", "mid_usd", "spread_btc",
            "mark_price_btc", "mark_price_usd", "iv", "volume", "open_interest"]
    df = df[[c for c in keep if c in df.columns]]
    df.sort_values(["expiry_dt", "strike", "option_type"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Instruments: {len(df)} | strikes: {df['strike'].nunique()} "
          f"| expiries: {df['expiry_dt'].nunique()}")

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    return df


def get_deribit_pricing_data(
        start_date="2025-01-01",
        end_date="2025-12-31",
        resolution="5",
        expiry_type=None,
        option_type=None,
        min_tte_days=0.0,
        max_tte_days=None,
        moneyness_range=None,
        min_price_btc=0.0001,
        option_chunk_hours=6,
        compute_forwards=True,
        forward_price_col="price_btc",    # the iv on a trade belongs to `price`
        use_perp_fallback=False,          # perp != index; off by default
        save_intermediate=True,
    ):
    index_df = get_index_price(start_date, end_date, resolution)

    if save_intermediate and not index_df.empty:
        name = f"deribit_btc_index_{resolution}.csv"
        index_df.to_csv(name, index=False)
        print(f"  Saved: {name}")

    option_df = fetch_all_option_trades(
        start_date, end_date,
        chunk_hours=option_chunk_hours,
        save_csv=save_intermediate,
    )

    if option_df.empty:
        print("[!] No option trades fetched.")
        return pd.DataFrame()

    panel = build_option_panel(
        option_df,
        resolution=resolution,
        expiry_type=expiry_type,
        option_type=option_type,
        min_tte_days=min_tte_days,
        max_tte_days=max_tte_days,
        moneyness_range=moneyness_range,
        min_price_btc=min_price_btc,
        start_date=start_date,
        end_date=end_date,
    )

    if panel.empty:
        return pd.DataFrame()

    panel = attach_spot(panel, index_df, resolution,
                        use_perp_fallback=use_perp_fallback)

    print("\nPerpetual basis diagnostic (index vs perp on shared bins):")
    index_basis_check(panel)

    if compute_forwards:
        panel = attach_forward_per_row(panel, price_col=forward_price_col)

    return panel


def resume_from_saved(index_csv, option_csv, resolution="5",
                      expiry_type=None, option_type=None,
                      min_tte_days=0.0, max_tte_days=None,
                      moneyness_range=None, min_price_btc=0.0001,
                      compute_forwards=True, forward_method="iv",
                      forward_price_col="price_btc",
                      start_date="2025-01-01", end_date="2025-12-31"):
    print("Resuming from saved CSVs ...")

    index_df = pd.read_csv(index_csv)
    index_df["timestamp"] = pd.to_datetime(index_df["timestamp"], utc=True)

    option_df = pd.read_csv(option_csv)
    option_df["trade_time"] = pd.to_datetime(option_df["trade_time"], utc=True)
    option_df["expiry_dt"] = pd.to_datetime(option_df["expiry_dt"])
    option_df["timestamp"] = pd.to_numeric(option_df["timestamp"])

    panel = build_option_panel(
        option_df,
        resolution=resolution,
        expiry_type=expiry_type,
        option_type=option_type,
        min_tte_days=min_tte_days,
        max_tte_days=max_tte_days,
        moneyness_range=moneyness_range,
        min_price_btc=min_price_btc,
        start_date=start_date,
        end_date=end_date,
    )

    if panel.empty:
        return pd.DataFrame()

    panel = attach_spot(panel, index_df, resolution,
                        use_perp_fallback=use_perp_fallback)

    if compute_forwards:
        panel = attach_forward_per_row(panel, price_col=forward_price_col)
    return panel


if __name__ == "__main__":

    panel = get_deribit_pricing_data(
        start_date="2025-01-01",
        end_date="2025-12-31",
        resolution="60",          # hourly bins give a fuller chain than 5-min
        expiry_type=None,         # None = every expiry
        option_type=None,         # None = calls and puts
        min_tte_days=1.0,         # drop the last day before expiry
        max_tte_days=None,
        moneyness_range=None,     # None = every strike
        min_price_btc=0.0001,     # drop one-tick dust quotes
        option_chunk_hours=6,
        compute_forwards=True,
        forward_price_col="price_btc",    # invert Deribit's own IV
        use_perp_fallback=False,   # keep the index honest; see index_source
        save_intermediate=True,
    )

    if not panel.empty:
        panel.to_csv("deribit_btc_pricing_panel_2025.csv", index=False)
        print("\nPanel saved: deribit_btc_pricing_panel_2025.csv")



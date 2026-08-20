#!/usr/bin/env python3
"""
find_top_gainers.py (replacement)

功能摘要:
- 抓取上市 (mode=2) 與上櫃/OTC (mode=4) 的股票清單（或讀取使用者提供的 CSV）
- 以批次方式 (batch) 用 yfinance 下載多檔 ticker 的歷史價格（Adj Close 優先）
- 每個 batch 完成就把結果 append 到 results/partial_results.csv（checkpoint）
- 最後合併結果並輸出 results/final_results.csv，並在終端顯示 Top N
- 支援重試、timeout 與簡單的進度顯示

執行範例:
  python3 scripts/find_top_gainers.py --start 2026-07-31 --end 2026-08-14 --top 50 --batch-size 100 --limit 0
  或使用自備清單:
  python3 scripts/find_top_gainers.py --start 2026-07-31 --end 2026-08-14 --tickers-file data/tickers_all.csv --top 50

注意:
- 請安裝依賴: pip install -r requirements.txt
  建議 requirements.txt 包含: requests, beautifulsoup4, yfinance, pandas, tqdm, lxml, html5lib
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import re
import time
from typing import List, Tuple

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm
from io import StringIO
from bs4 import BeautifulSoup

# Constants
ISIN_BASE = "https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
RESULTS_DIR = "results"
CACHE_CSV = os.path.join(RESULTS_DIR, "stock_list.csv")
PARTIAL_CSV = os.path.join(RESULTS_DIR, "partial_results.csv")
FINAL_CSV = os.path.join(RESULTS_DIR, "final_results.csv")


def read_tickers_from_csv(path: str, limit: int = 0) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"tickers file not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            code = str(row[0]).strip()
            # skip header-like lines
            if not code or not re.match(r"^\d{3,4}$", code):
                continue
            name = row[1].strip() if len(row) > 1 else ""
            out.append((code, name))
            if limit and len(out) >= limit:
                break
    return out


def _fetch_isin_html(mode: int, session: requests.Session, timeout: int = 30) -> str:
    url = ISIN_BASE.format(mode=mode)
    resp = session.get(url, timeout=timeout)
    # encoding fallback
    if not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "latin-1"):
        resp.encoding = resp.apparent_encoding or "big5"
    html = resp.text
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        with open(os.path.join(RESULTS_DIR, f"isin_mode{mode}.html"), "w", encoding="utf-8") as fh:
            fh.write(html)
    except Exception:
        pass
    return html



def _parse_isin_table_from_html(html: str) -> pd.DataFrame:
    """
    更健壯的 parse：先用 StringIO + pd.read_html，失敗時 fallback 用 BeautifulSoup 手動解析 <table>。
    """
    # 1) safe pd.read_html via StringIO (解掉 FutureWarning)
    try:
        tables = pd.read_html(StringIO(html), encoding="utf-8", flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(StringIO(html), encoding="utf-8", flavor="html5lib")
        except Exception:
            tables = []

    if tables:
        # 優先找包含關鍵欄位的 table
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("有價" in c or "證券" in c or "代號" in c or "名稱" in c for c in cols):
                return t
        # fallback: 回傳第一個表格
        return tables[0]

    # 2) fallback: 用 BeautifulSoup 手動解析第一個 <table>
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return pd.DataFrame()

    rows = []
    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        rows.append(cols)

    if not rows:
        return pd.DataFrame()

    # pad rows to same length
    maxc = max(len(r) for r in rows)
    rows_padded = [r + [""] * (maxc - len(r)) for r in rows]
    df = pd.DataFrame(rows_padded)

    # 如果第一列看起來像 header（含中文欄名），把它當 header
    first_row = " ".join(map(str, df.iloc[0].tolist()))
    if any(k in first_row for k in ["有價", "證券", "代號", "名稱"]):
        df.columns = df.iloc[0].tolist()
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _extract_code_name(df: pd.DataFrame) -> List[Tuple[str, str]]:
    col_candidates = df.columns.tolist()
    sec_type_col = None
    namecol = None
    for c in col_candidates:
        s = str(c)
        if "有價" in s and "證券" in s:
            sec_type_col = c
        if "代號" in s and "名稱" in s:
            namecol = c
    if namecol is None:
        # try to find a column where first non-null cell contains digits
        for c in col_candidates:
            try:
                sample = str(df[c].dropna().iloc[0])
            except Exception:
                continue
            if re.search(r"\d{3,4}", sample):
                namecol = c
                break
    out: List[Tuple[str, str]] = []
    if namecol is None:
        return out
    # filter for '股票'
    if sec_type_col is not None:
        df = df[df[sec_type_col].astype(str).str.contains("股票", na=False)]
    for v in df[namecol].astype(str).fillna(""):
        m = re.match(r"^\s*(\d{3,4})\s*(.+?)\s*$", v)
        if m:
            code = m.group(1)
            name = m.group(2)
        else:
            parts = v.strip().split(None, 1)
            if parts and re.match(r"^\d{3,4}$", parts[0]):
                code = parts[0]
                name = parts[1] if len(parts) > 1 else ""
            else:
                continue
        out.append((code, name))
    # dedup keep order
    seen = set()
    dedup = []
    for c, n in out:
        if c not in seen:
            seen.add(c)
            dedup.append((c, n))
    return dedup


def get_tw_listed_and_otc(limit: int = 0) -> List[Tuple[str, str]]:
    """抓上市(mode=2) 與上櫃(mode=4)，回傳 list of (code, name)，並快取到 results/stock_list.csv"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; find_top_gainers/1.0; +https://github.com/)",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7"
    })
    all_entries: List[Tuple[str, str]] = []
    for mode in (2, 4):
        try:
            html = _fetch_isin_html(mode, session, timeout=30)
            df_table = _parse_isin_table_from_html(html)
            if df_table is None or df_table.empty:
                continue
            entries = _extract_code_name(df_table)
            if entries:
                all_entries.extend(entries)
            time.sleep(0.5)
        except Exception:
            continue
    # dedup and limit
    seen = set()
    out = []
    for code, name in all_entries:
        if code not in seen:
            seen.add(code)
            out.append((code, name))
            if limit and len(out) >= limit:
                break
    # cache to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        pd.DataFrame(out, columns=["code", "name"]).to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return out


def load_tickers(args) -> List[Tuple[str, str]]:
    # priority: --tickers-file -> cached results/stock_list.csv -> fetch from ISIN
    if args.tickers_file:
        print(f"讀取 ticker 清單: {args.tickers_file}")
        stocks = read_tickers_from_csv(args.tickers_file, limit=args.limit)
        return stocks
    if os.path.exists(CACHE_CSV) and not args.force_refresh:
        try:
            df = pd.read_csv(CACHE_CSV, dtype={"code": str})
            stocks = [(str(r["code"]).zfill(3), r.get("name", "")) for _, r in df.iterrows()]
            if args.limit:
                stocks = stocks[: args.limit]
            if stocks:
                print(f"從快取載入 {len(stocks)} 檔股票 (cache: {CACHE_CSV})")
                return stocks
        except Exception:
            pass
    print("抓取上市與上櫃股票清單...")
    stocks = get_tw_listed_and_otc(limit=args.limit)
    if args.limit:
        stocks = stocks[: args.limit]
    print(f"共取得 {len(stocks)} 檔股票")
    return stocks


def download_batch_with_retry(tickers: List[str], start: str, end: str, max_retries: int = 3, backoff: int = 2):
    attempt = 0
    end_exclusive = pd.to_datetime(end) + pd.Timedelta(days=1)
    while attempt <= max_retries:
        try:
            df = yf.download(tickers=tickers, start=start, end=end_exclusive.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=False, threads=False)
            return df
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep = backoff ** attempt
            time.sleep(min(sleep, 60))


def compute_pct_from_series(s: pd.Series) -> Tuple[float, int]:
    s = s.dropna()
    if s.empty:
        return float("nan"), 0
    first = s.iloc[0]
    last = s.iloc[-1]
    pct = (last / first - 1.0) * 100.0
    return float(pct), int(len(s))


def process_batches(tickers: List[str], start: str, end: str, args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # remove partial if starting fresh and not resuming
    if args.resume and os.path.exists(PARTIAL_CSV):
        print(f"Resume enabled: will append to existing {PARTIAL_CSV}")
    else:
        if os.path.exists(PARTIAL_CSV):
            try:
                os.remove(PARTIAL_CSV)
            except Exception:
                pass

    total = len(tickers)
    n_batches = math.ceil(total / args.batch_size) if total else 0
    all_rows = []
    for i in tqdm(range(n_batches), desc="batches"):
        start_idx = i * args.batch_size
        batch_codes = tickers[start_idx: start_idx + args.batch_size]
        if not batch_codes:
            continue
        batch_tickers = [f"{c}.TW" for c in batch_codes]
        try:
            df = download_batch_with_retry(batch_tickers, start, end, max_retries=args.max_retries, backoff=2)
        except Exception as e:
            # on total failure, mark NaNs for this batch
            rows = [{"ticker": t, "pct": float("nan"), "n_days": 0} for t in batch_tickers]
            df_rows = pd.DataFrame(rows)
            if os.path.exists(PARTIAL_CSV):
                df_rows.to_csv(PARTIAL_CSV, mode="a", index=False, header=False, encoding="utf-8-sig")
            else:
                df_rows.to_csv(PARTIAL_CSV, index=False, header=True, encoding="utf-8-sig")
            all_rows.extend(df_rows.to_dict("records"))
            continue

        # interpret df: could be empty, single-ticker, or multi-ticker multiindex
        rows = []
        if df is None or df.empty:
            for t in batch_tickers:
                rows.append({"ticker": t, "pct": float("nan"), "n_days": 0})
        else:
            if isinstance(df.columns, pd.MultiIndex):
                # look for 'Adj Close' or 'Close' in level 0
                top_level = df.columns.levels[0]
                value_col = None
                if "Adj Close" in top_level:
                    value_col = "Adj Close"
                elif "Close" in top_level:
                    value_col = "Close"
                else:
                    value_col = top_level[0]
                for t in batch_tickers:
                    try:
                        if value_col in df and t in df[value_col].columns:
                            s = df[value_col][t].dropna()
                        else:
                            s = pd.Series(dtype=float)
                    except Exception:
                        s = pd.Series(dtype=float)
                    pct, nd = compute_pct_from_series(s)
                    rows.append({"ticker": t, "pct": pct, "n_days": nd})
            else:
                # single-ticker style or DataFrame with single level columns
                # try to pick 'Adj Close' or 'Close'
                if "Adj Close" in df.columns:
                    s = df["Adj Close"].dropna()
                elif "Close" in df.columns:
                    s = df["Close"].dropna()
                else:
                    # if numeric column exists, pick the last numeric-like column
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    s = df[numeric_cols[-1]].dropna() if numeric_cols else pd.Series(dtype=float)
                for t in batch_tickers:
                    pct, nd = compute_pct_from_series(s)
                    rows.append({"ticker": t, "pct": pct, "n_days": nd})

        df_rows = pd.DataFrame(rows)
        # append to partial CSV
        if os.path.exists(PARTIAL_CSV):
            df_rows.to_csv(PARTIAL_CSV, mode="a", index=False, header=False, encoding="utf-8-sig")
        else:
            df_rows.to_csv(PARTIAL_CSV, index=False, header=True, encoding="utf-8-sig")
        all_rows.extend(df_rows.to_dict("records"))
        # polite short sleep to avoid hammering
        time.sleep(0.2)

    # load final merged results (from partial)
    if os.path.exists(PARTIAL_CSV):
        final_df = pd.read_csv(PARTIAL_CSV, dtype={"ticker": str})
    else:
        final_df = pd.DataFrame(all_rows)
    # cleanup ticker format in final_df if needed
    if "ticker" in final_df.columns:
        # remove possible .TW suffix for matching with names elsewhere
        final_df["ticker"] = final_df["ticker"].astype(str)
    # drop NaN pct
    final_df = final_df.dropna(subset=["pct"])
    if final_df.empty:
        print("找不到任何有效資料。")
        return None
    final_df = final_df.sort_values("pct", ascending=False).reset_index(drop=True)
    # write final CSV
    try:
        final_df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return final_df


def main(argv=None):
    p = argparse.ArgumentParser(description="Find top gainers in TW stocks between two dates (batch mode)")
    p.add_argument("--start", required=True, help="開始日 (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="結束日 (YYYY-MM-DD)")
    p.add_argument("--top", type=int, default=10, help="列出前 N 名")
    p.add_argument("--limit", type=int, default=0, help="限制要處理的股票數 (0=不限制)")
    p.add_argument("--batch-size", type=int, default=100, help="一次 batch 下載的 ticker 數量 (預設 100)")
    p.add_argument("--tickers-file", type=str, default="", help="CSV 檔 (code,name) 作為 ticker 清單，若提供則優先使用")
    p.add_argument("--force-refresh", action="store_true", help="強制重新抓 ISIN 並覆蓋快取")
    p.add_argument("--resume", action="store_true", help="若 partial 結果存在則從該檔續跑 (append)")
    p.add_argument("--max-retries", type=int, default=3, help="batch download 最大重試次數")
    args = p.parse_args(argv)

    try:
        _ = pd.to_datetime(args.start)
        _ = pd.to_datetime(args.end)
    except Exception:
        print("日期格式錯誤，請使用 YYYY-MM-DD")
        return

    # load tickers
    stocks = []
    if args.tickers_file:
        stocks = read_tickers_from_csv(args.tickers_file, limit=args.limit)
    else:
        # if force_refresh, remove existing cache
        if args.force_refresh and os.path.exists(CACHE_CSV):
            try:
                os.remove(CACHE_CSV)
            except Exception:
                pass
        stocks = load_tickers(args)

    if not stocks:
        print("沒有可用的股票清單，終止。請提供 --tickers-file 或確認網頁能連線/快取。")
        return

    if args.limit and args.limit > 0:
        stocks = stocks[: args.limit]

    codes = [c for c, _ in stocks]
    print(f"開始計算 {len(codes)} 檔股票的漲幅，期間 {args.start} ~ {args.end}，batch_size={args.batch_size}")

    final_df = process_batches(codes, args.start, args.end, args)
    if final_df is None:
        return

    # 映回公司名稱 (如果有 cache)
    try:
        name_map = {c: n for c, n in stocks}
        # ensure ticker in final_df uses numeric code (strip .TW if present)
        def strip_tw(t):
            return t.replace(".TW", "") if isinstance(t, str) else t
        final_df["code"] = final_df["ticker"].astype(str).apply(strip_tw)
        final_df["name"] = final_df["code"].map(name_map)
    except Exception:
        pass

    topn = args.top
    print(f"前 {topn} 名漲幅：")
    display_df = final_df.head(topn).copy()
    # format pct
    if "pct" in display_df.columns:
        display_df["pct"] = display_df["pct"].map(lambda x: f"{x:,.2f}%")
    print(display_df[["code", "name", "pct", "n_days"]].to_string(index=False))

    # also write top N to results/topN.csv
    try:
        display_df.to_csv(os.path.join(RESULTS_DIR, f"top_{topn}.csv"), index=False, encoding="utf-8-sig")
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""find_top_gainers.py

Usage:
  python3 scripts/find_top_gainers.py --start 2026-01-01 --end 2026-06-30 --top 10

What it does:
  - 下載目前上市上櫃（從證交所 ISIN 列表抓取）的股票代碼
  - 使用 yfinance 抓取每檔在指定期間的調整後收盤價（Adj Close）
  - 計算從 start 日的第一個可得收盤價到 end 日的最後一個可得收盤價的漲幅
  - 列出漲幅前 N 名（預設 10）

Notes:
  - 需要能連網以抓取 ISIN 列表與 yfinance 數據。
  - yfinance 的 ticker 格式為 {code}.TW，例如 2330.TW
  - 執行可能較慢（會對很多 ticker 發出 HTTP 請求），建議在有穩定網路下執行。
"""

import argparse
import concurrent.futures
import datetime
import sys
from typing import List, Tuple

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tqdm import tqdm

import re
import os

ISIN_LIST_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
RESULTS_DIR = "results"

def get_tw_stock_list() -> List[Tuple[str, str]]:
    \"\"\"更健壯的從台灣 ISIN 頁面抓取上市/上櫃代碼與名稱。
    回傳 list of (code, name)，code 為純數字字串（例如 '2330'）。
    同時會把原始 HTML 存到 results/isin_page.html 以便除錯。
    \"\"\"
    try:
        resp = requests.get(ISIN_LIST_URL, timeout=20)
    except Exception as e:
        raise RuntimeError(f\"無法連到 ISIN 網頁: {e}\")
    if resp.status_code != 200:
        raise RuntimeError(f\"無法抓取 ISIN 列表, status={resp.status_code}\")
    # 若網站宣告的 encoding 不對，使用 apparent_encoding 作為備援
    if not resp.encoding or resp.encoding.lower() in ('iso-8859-1', 'latin-1'):
        resp.encoding = resp.apparent_encoding or 'big5'
    # 最終以 big5/utf-8 互換嘗試也可以
    html = resp.text

    # 儲存原始 HTML 以便後續檢查
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        with open(os.path.join(RESULTS_DIR, "isin_page.html"), "w", encoding="utf-8") as fh:
            fh.write(html)
    except Exception:
        # 忽略寫檔錯誤，但仍繼續解析
        pass

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    stocks = []

    # 先嘗試由 table 找出代碼/名稱
    for table in tables:
        for row in table.find_all("tr"):
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cols) >= 2:
                code = cols[0].strip().strip(\"'\\u200b\\u00a0\")
                name = cols[1].strip()
                # 寬鬆驗證：3 或 4 位數字 (有些公司代碼可能是 3 位或 4 位)
                if re.match(r'^\\d{3,4}$', code):
                    stocks.append((code, name))

    # 若 table 解析不到，再用 regex 備援抓取頁面內的 3-4 位數字序列
    if not stocks:
        codes = sorted(set(re.findall(r'\\b(\\d{3,4})\\b', html)))
        for code in codes:
            # 嘗試在鄰近文字抓公司名稱（簡單 approach）
            name = ""
            # 找到 code 在 html 出現的位置，取後面一小段當作名稱候選
            m = re.search(r'(' + re.escape(code) + r')[^<\\n\\r]{0,80}', html)
            if m:
                snippet = m.group(0)
                # 從 snippet 去掉數字本身，取剩下文字當 name candidate
                cand = re.sub(re.escape(code), '', snippet).strip(' -:：,，\\t\\n\\r')
                name = re.sub(r'\\s+', ' ', cand).strip()
            stocks.append((code, name))

    # 去重並保持順序
    seen = set()
    out = []
    for code, name in stocks:
        if code not in seen:
            seen.add(code)
            out.append((code, name))
    return out

def pct_change_for_ticker(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[str, float, int]:
    """回傳 (ticker, pct_change, valid_days)

    pct_change = (last_adj / first_adj - 1) if both exists, 否則回傳 None
    """
    try:
        # yfinance 的 end 是 exclusive，所以加 1 天
        end_exclusive = end + pd.Timedelta(days=1)
        hist = yf.download(tickers=ticker, start=start.strftime("%Y-%m-%d"), end=end_exclusive.strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
        if hist is None or hist.empty:
            return ticker, float('nan'), 0
        # 使用 Adj Close if available
        if 'Adj Close' in hist.columns:
            s = hist['Adj Close'].dropna()
        elif 'Close' in hist.columns:
            s = hist['Close'].dropna()
        else:
            return ticker, float('nan'), 0
        if s.empty:
            return ticker, float('nan'), 0
        first = s.iloc[0]
        last = s.iloc[-1]
        pct = (last / first - 1) * 100.0
        return ticker, float(pct), len(s)
    except Exception:
        return ticker, float('nan'), 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find top gainers in TW stocks between two dates")
    parser.add_argument('--start', required=True, help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='結束日 (YYYY-MM-DD)')
    parser.add_argument('--top', type=int, default=10, help='列出前 N 名 (default 10)')
    parser.add_argument('--max-workers', type=int, default=10, help='同時執行的 worker 數量')
    parser.add_argument('--limit', type=int, default=0, help='限制要處理的股票數 (0=不限制，方便測試)')
    args = parser.parse_args(argv)

    try:
        start = pd.to_datetime(args.start)
        end = pd.to_datetime(args.end)
    except Exception as e:
        print("日期格式錯誤，請使用 YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    if end < start:
        print("結束日必須在開始日之後", file=sys.stderr)
        sys.exit(1)

    print("抓取台股代碼列表...")
    stocks = get_tw_stock_list()
    if args.limit > 0:
        stocks = stocks[:args.limit]
    print(f"共取得 {len(stocks)} 檔股票，開始計算漲幅 (可能需要一些時間)...")

    tickers = [f"{code}.TW" for code, _ in stocks]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as exc:
        futures = {exc.submit(pct_change_for_ticker, t, start, end): t for t in tickers}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            ticker = futures[f]
            try:
                t, pct, ndays = f.result()
                results.append((t, pct, ndays))
            except Exception:
                results.append((ticker, float('nan'), 0))

    df = pd.DataFrame(results, columns=['ticker', 'pct', 'n_days']).dropna(subset=['pct'])
    if df.empty:
        print('找不到任何有效資料。')
        sys.exit(0)

    df = df.sort_values('pct', ascending=False)
    topn = args.top
    print(f"前 {topn} 名漲幅：")
    print(df.head(topn).to_string(index=False, formatters={'pct': '{:,.2f}%'.format}))


if __name__ == '__main__':
    main()

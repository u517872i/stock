# 台股找出漲幅最大的股票工具

新增檔案:
- scripts/find_top_gainers.py  : 主程式
- requirements.txt            : 執行所需套件

使用方式（在有 Python3 與網路的環境下）:

1. 建議建立虛擬環境並安裝套件：

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. 執行：

   python3 scripts/find_top_gainers.py --start 2026-01-01 --end 2026-06-30 --top 10

參數說明：
- --start: 開始日 (含)
- --end: 結束日 (含)
- --top: 顯示前 N 名 (預設 10)
- --limit: 若只是想測試，可限制只處理前 N 檔股票 (0 表示不限制)
- --max-workers: 同時執行的 thread 數量，預設 10

注意事項：
- 程式會從證交所的 ISIN 列表抓取股票代碼，並使用 yfinance 下載歷史價格，因此需要能連網。
- yfinance 取得的資料品質與完整性依賴 Yahoo Finance；若遇到缺資料可改用其他資料源或自行提供 ticker 清單。
- 如果你想要針對特定股票清單執行，我可以幫你加上選項來讀入 CSV 檔。

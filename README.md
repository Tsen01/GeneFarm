# GeneFarm
GeneFarm 是一個結合基因資料分析與體重成長預測模型的智慧畜牧平台，系統採用 Bi-LSTM + Attention 與 Transformer 深度學習模型，並搭配 MongoDB 資料庫與網頁介面，提供雙角色模式：

● 基因研究員：專注於基因數據分析與預測。<br>
● 牧場使用者：聚焦於羊隻管理與成長預測。

---

## 目錄  
- [資料來源](#資料來源)
- [系統目的](#系統目的) 
- [系統特色](#系統特色)   
- [快速開始](#快速開始)  
- [團隊](#團隊)  
- [貢獻指南](#貢獻指南)  
- [授權](#授權)  

---

## 資料來源
- 羊隻資料來自農業部畜產試驗所南區分所屏東廠區，由陳水財研究員提供。
- 基因資料來自 [NCBI 的公開基因資料集](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275445)

---

## 系統目的  
隨著全球糧食需求上升與永續發展的壓力，傳統畜牧方式已難以應付，因此現代畜牧需要更精緻化與高效率的解決方案。以台灣羊隻為例，單次生育大多只有一到二胎，我們希望透過基因精準育種，提升多胎的可能，並結合成長預測，減少飼料浪費，朝向精準畜牧。

---

## 系統特色  
- **基因篩選模型 (Transformer)**：篩選出羊隻的潛在多胎繁殖基因，提升多胎的可能。  
- **成長預測模型 (Bi-LSTM + Attention)**：預測羊隻體重成長趨勢，輔助飼料規劃與資源管理。  
- **大型資料處理 (MongoDB)**：彈性管理羊隻三代祖譜、成長紀錄與基因資料等多元結構，支援高效資料查詢。
- **視覺化圖形設計**：以圖表呈現成長趨勢與基因篩選結果，提供決策參考。
- **多欄位模糊搜尋**：快速搜尋羊隻資訊與歷史資料，提升操作便利性。

---

## 快速開始
1. 建立虛擬環境並安裝相關套件
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS  
# .venv\Scripts\activate  # Windows  
pip install -r requirements.txt
```
2. 設定 .env
```bash
MONGO_HOST=YOUR_MONGO_HOST
MONGO_PORT=27017(預設)
MONGO_USER=YOUR_MONGO_USER
MONGO_PASSWORD=YOUR_MONGO_PASSWORD
AUTH_DB=YOUR_AUTH_DB
DB_NAME=YOUR_DATABASE_NAME

ALLOWED_ORIGINS=http://localhost,http://127.0.0.1,http://127.0.0.1:8000
```
3. 啟動後端服務有兩種方式：
  - 第一種輸入指令
  ```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  ```
  打開網頁輸入網址 http://127.0.0.1:8000 or http://localhost:8000

  
  - 第二種開啟 goat_notebook.ipynb，並執行以下程式
  ```bash
  import uvicorn
  import nest_asyncio
  from threading import Thread
  
  # 允許 Jupyter 重複使用事件 loop
  nest_asyncio.apply()
  
  def run_app():
      # 移除 reload=True，避免 signal 問題
      uvicorn.run("main:app", host="127.0.0.1", port=8000)
      # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
  
  # 使用 Thread 非同步啟動 FastAPI
  server_thread = Thread(target=run_app, daemon=True)
  server_thread.start()
  ```

  點擊執行結果的網址或在網頁輸入網址 http://127.0.0.1:8000
  ```bash
  INFO:     Uvicorn running on http://127.0.0.1:8000
  ```

---

## 團隊
- 國立屏東科技大學資訊管理系
- 指導教授：賴佳瑜
- 專案成員：李育岑、何承諺、王子齊

---

## 貢獻指南
1. Fork 專案並建立新分支
2. 進行修改、更新測試
3. 提交 Pull Request，說明變更內容與動機

---

## 授權
本專案以 MIT License 授權。歡迎自由使用與擴充。

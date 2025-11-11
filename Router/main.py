import os
from fastapi import FastAPI, Request, HTTPException, Body, Query, File, UploadFile, Depends
from pydantic import BaseModel, EmailStr
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient
from typing import Dict, Optional
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import pandas as pd
import threading
import joblib, torch
from predict import (
    _load_lstm, _load_bilstm,
    multi_step_recursive_predict, multi_step_lr_predict,
    predict_with_lstm, predict_with_bilstm, predict_with_lr
)
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import io, base64
from scipy.optimize import curve_fit
from urllib.parse import quote_plus
from flask import request, jsonify
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from io import BytesIO
from model_def import TabTransformerCLS, EncoderLayerWithAttn
from zoneinfo import ZoneInfo

def logistic_growth(x, K, r, x0):
    return K / (1 + np.exp(-r * (x - x0)))

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None

load_dotenv()

# 允許的來源
origins = ["http://localhost:8000", "http://127.0.0.1:8000"]

print("SSH_HOST:", os.getenv("SSH_HOST"))
print("MONGO_HOST:", os.getenv("MONGO_HOST"))
print("ALLOWED_ORIGINS:", os.getenv("ALLOWED_ORIGINS"))


# SSH and MongoDB Settings from Environment Variables
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", 22))
SSH_USER = os.getenv("SSH_USER")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")

MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
AUTH_DB = os.getenv("AUTH_DB", "admin")
DB_NAME = os.getenv("DB_NAME", "goat_project")

GOAT_COLLECTIONS = ["0007A1", "0007A2", "0007A3", "0007A5", "0007S2Breed"]
SHEEP_COLLECTIONS = [
    "0009-0013A11_MilkAnalysis", "0009-0013A9_Milk", "0009-0013_A4_Kidding",
    "0009-0013_Yean", "S2_Breed", "S7_Sex", "basic", "pubmat"
]

# 全域的 MongoDB 客戶端和 SSH Tunnel
mongo_client = None
ssh_tunnel = None
client_lock = threading.Lock()

# 載入 CSV
df = pd.read_csv("../../weight_prediction_data.csv")

# --------- 通用轉換工具 ---------
def as_float(x):
    # 轉為 Python float，空值/無法轉換 -> None
    try:
        if x is None:
            return None
        # pandas 的 NaN
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, (np.floating, np.integer, np.number)):
            return float(x)
        # 字串也嘗試轉
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None

def as_date_str(x):
    # 轉為 'YYYY-MM-DD'；無法轉 -> None
    if x is None:
        return None
    try:
        # 直接處理 pandas.Timestamp / numpy.datetime64 / str / int
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        # 只要日期（和你 safe_date 對齊）
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def as_str(x):
    # 轉為 Python str，包含 numpy 泛型 -> .item()
    if x is None:
        return None
    try:
        if isinstance(x, np.generic):
            x = x.item()
        return str(x)
    except Exception:
        return None

# --------- 路由 ---------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongo_client, ssh_tunnel
    try:
        # 建立 SSH Tunnel 和 MongoDB 連線
        ssh_tunnel = SSHTunnelForwarder(
            (SSH_HOST, SSH_PORT),
            ssh_username=SSH_USER,
            ssh_password=SSH_PASSWORD,
            remote_bind_address=(MONGO_HOST, MONGO_PORT)
        )
        ssh_tunnel.start()

        mongo_client = MongoClient(
            f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@127.0.0.1:{ssh_tunnel.local_bind_port}/?authSource={AUTH_DB}"
        )

        app.state.mongo_client = mongo_client
        app.state.ssh_tunnel = ssh_tunnel
        print("🔗 MongoDB 連線已建立")
        yield

    finally:
        # 關閉 MongoDB 和 SSH Tunnel 連線
        if mongo_client is not None:
            mongo_client.close()
            print("🛑 MongoDB 連線已關閉")
        if ssh_tunnel is not None:
            ssh_tunnel.stop()
            print("🛑 SSH Tunnel 已關閉")

app = FastAPI(lifespan=lifespan)

# 啟動時載入模型
model_data = joblib.load("gene_model.pkl")
svd = model_data["svd"]
scaler = model_data["scaler"]
model = model_data["model"]
gene_names = model_data["gene_names"]

model.eval()  # 設成推論模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 配置模板
templates = Jinja2Templates(directory="templates")  # 指定 HTML 模板的目錄

# 設定靜態文件資料夾
base_dir = os.path.dirname(__file__)  # 取得 main.py 所在目錄
static_dir = os.path.join(base_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Enable CORS (more secure than allowing all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

from auth import auth_router
app.include_router(auth_router, prefix="/auth", tags=["auth"])

from farm_upload import farm_router
app.include_router(farm_router, prefix="/farm", tags=['farm_upload'])

from gene_upload import gene_router
# 註冊 gene_upload router
app.include_router(gene_router, prefix="/gene", tags=["gene_upload"])

# 路由：首頁
@app.get("/", response_class=HTMLResponse)
async def redirect_to_animal_manager():
    return RedirectResponse("/AnimalManager", status_code=302)

@app.get("/AnimalManager", response_class=HTMLResponse)
async def animal_manager_home(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

# 路由：我的牧場
@app.get("/AnimalManager/myfarm", response_class=HTMLResponse)
async def animal_manager_myfarm(request: Request):
    return templates.TemplateResponse("myfarm.html", {"request": request})

@app.get("/AnimalManager/Edit/{collectionName}", response_class=HTMLResponse)
async def animal_manager_myfarm(request: Request):
    return templates.TemplateResponse("EditData.html", {"request": request})

@app.get("/AnimalManager/EditGene/{collectionName}", response_class=HTMLResponse)
async def animal_manager_myfarm(request: Request):
    return templates.TemplateResponse("EditGene.html", {"request": request})

@app.get("/AnimalManager/predict", response_class=HTMLResponse)
async def animal_manager_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/AnimalManager/search", response_class=HTMLResponse)
async def animal_manager_search(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/AnimalManager/gene", response_class=HTMLResponse)
async def animal_manager_gene(request: Request):
    return templates.TemplateResponse("gene.html", {"request": request})

@app.get("/AnimalManager/genePredict", response_class=HTMLResponse)
async def animal_manager_gene(request: Request):
    return templates.TemplateResponse("genePredict.html", {"request": request})

@app.get("/AnimalManager/history", response_class=HTMLResponse)
async def animal_manager_gene(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


# 測試是否有成功連到 MongoDB
@app.get("/test_connection")
async def test_connection():
    try:
        db = mongo_client[DB_NAME]
        db_list = db.list_collection_names()
        return {"status": "success", "collections": db_list}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 載入資料集
client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@localhost:27017/{DB_NAME}?authSource=admin")
db = client[DB_NAME]

from auth import get_current_user

@app.get("/get_collections")
async def get_collections():
    try:
        collections = db.list_collection_names()
        return collections  # 會自動轉成 JSON
    except Exception as e:
        print(f"發生錯誤：{e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict")
def run_prediction(payload: Dict = Body(...), current_user=Depends(get_current_user)):
    db = mongo_client['user_accounts']
    history_collection = db['growth_prediction']
    model_type = payload.get("model_type", "lstm")
    input_data = payload.get("input_data")
    user_id = current_user.get("sub")
    user_name = current_user.get("username")
    user_role = current_user.get("role")

    if not input_data or len(input_data) < 12:
        raise HTTPException(status_code=400, detail="input_data must have at least 12 items")

    try:
        # 出生體重與日期
        self_birweight = safe_float(input_data[1])
        self_birmeadate = safe_date(input_data[2])

        # 收集所有有效測量
        weight_date_pairs = []
        for i in range(3, 13, 2):
            weight = safe_float(input_data[i])
            meadate = safe_date(input_data[i+1])
            if weight is not None and meadate is not None:
                weight_date_pairs.append((meadate, weight))

        if self_birweight is None or self_birmeadate is None:
            raise HTTPException(status_code=400, detail="出生體重與出生測量日期為必填")
        if len(weight_date_pairs) == 0:
            raise HTTPException(status_code=400, detail="至少要有一筆有效的測量資料")

        weight_date_pairs.sort(key=lambda x: x[0])
        dates = [d for d, _ in weight_date_pairs]
        days_since_birth = [(d - self_birmeadate).days for d in dates]
        weights = [w for _, w in weight_date_pairs]

        # 預測天數
        try:
            predict_days = int(input_data[18].strip())
        except (IndexError, ValueError, AttributeError):
            predict_days = 30

        if predict_days <= 0:
            raise HTTPException(status_code=400, detail="預測天數必須大於 0")

        # === 準備輸入給 LSTM ===
        seq_data = []
        for i in range(3, 13, 2):
            weight = safe_float(input_data[i])
            mea_date = safe_date(input_data[i+1])
            if weight is not None and mea_date is not None:
                days = (mea_date - self_birmeadate).days
                if days >= 0:
                    seq_data.append([weight, days])
        seq_data = seq_data[-4:]
        while len(seq_data) < 4:
            seq_data.insert(0, [0.0, 0.0])
        X_seq = np.array([seq_data], dtype=np.float32)

        # 讀取 breed mapping & lifespan
        breed_info = joblib.load("../breed_info_BiLSTM_Mapping.pkl")
        breed_mapping = breed_info['mapping']
        breed_lifespan = breed_info['lifespan']

        # 先記下所有原始欄位名，避免在迭代時動態改字典
        original_cols = list(breed_mapping.keys())

        # 先建立反向 mapping
        for col in original_cols:
            if not col.endswith("_reverse"):
                breed_mapping[f"{col}_reverse"] = {v: k for k, v in breed_mapping[col].items()}

        # 現在再檢查 mapping（正向 & 反向都會有）
        for col in breed_mapping:
            print(f"{col} 的 mapping：")
            print(breed_mapping[col])
        print("-" * 40)

        # 編碼函式
        def encode_breed(breed, field_name):
            return breed_mapping.get(f"{field_name}_reverse", {}).get(breed, 0)


        self_breed = encode_breed(input_data[13], 'self_Breed')
        dam_breed = encode_breed(input_data[15], 'dam_Breed')
        sire_breed = encode_breed(input_data[17], 'sire_Breed')

        # 如果要找名稱，直接用原 mapping (數字 -> 名稱)
        self_breed_name = breed_mapping['self_Breed_reverse'].get(self_breed, 'Unknown')

        # 取最後一筆測量日期，如果沒有就用出生日期
        last_mea_date = dates[-1] if len(dates) > 0 else self_birmeadate

        # 預測目標日期 = 最後測量日期 + 預測天數
        target_date = last_mea_date + timedelta(days=predict_days)

        # 計算距離出生的總天數（用於 log 與 age_ratio）
        total_days_from_birth = (target_date - self_birmeadate).days
        log_predict_days = math.log(total_days_from_birth + 1)

        # 計算年齡比例
        max_life_days = breed_lifespan.get(self_breed_name, 365 * 10)
        age_ratio = total_days_from_birth / max_life_days
        max_weight = 90.0

        # 只用 weight，不用 days
        flat_seq = [w for w, d in seq_data]  # 3 維
        Regression_static_feat = [log_predict_days, age_ratio]  # 2 維
        X_lr = np.array([flat_seq + Regression_static_feat], dtype=np.float32)  # 總共 5 維

        # 靜態特徵
        static_feat = [self_breed, dam_breed, sire_breed, log_predict_days, age_ratio, max_weight]

        # 這裡只需要呼叫，不需要再定義函數
        att_model = _load_lstm()
        bilstm_model = _load_bilstm()
        # 把最後真實點接上去
        last_day = last_mea_date
        last_weight = weights[-1]

        # 有Attention的Bi-LSTM
        lstm_preds, lstm_dates = multi_step_recursive_predict(att_model, X_seq[0].tolist(), static_feat,
                                                            self_birmeadate, self_breed_name, breed_lifespan, last_real_weight=last_weight, last_mea_date = last_day)
        # 沒有Attention的Bi-LSTM
        bilstm_preds, bilstm_dates = multi_step_recursive_predict(bilstm_model, X_seq[0].tolist(), static_feat,
                                                                self_birmeadate, self_breed_name, breed_lifespan, last_real_weight=last_weight, last_mea_date = last_day)
        lr_preds, lr_dates = multi_step_lr_predict(list(flat_seq), static_feat,
                                                self_birmeadate, self_breed_name, breed_lifespan, last_real_weight=last_weight, last_mea_date = last_day)

        # 把日期轉成距出生的天數
        lstm_days = [(d - self_birmeadate).days for d in lstm_dates]
        bilstm_days = [(d - self_birmeadate).days for d in bilstm_dates]
        lr_days = [(d - self_birmeadate).days for d in lr_dates]

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    def to_native(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (pd.Timestamp, datetime, date)):
            return x.isoformat()
        return x
    
    # --- 如果前端/資料有「實際的未來點」，試著對齊計算 metrics ---
    # 例如 weight_date_pairs 中若包含已來到未來時間的實測點（dates > last_mea_date），就拿來比對
    actuals_by_date = {d.date(): w for d, w in weight_date_pairs}  # 原先收集到的所有點（通常只有過去）
    actual_series = [{"date": d.isoformat(), "weight": float(w)} for d, w in weight_date_pairs]
    print("所有實際點：", actuals_by_date)
    # 如果你前端會傳更多未來真實值到 input_data，我們要把它們也塞進 weight_date_pairs 才會被使用
    
        # === 評估指標 (改為單步預測) ===
    def compute_single_step_metrics(weight_date_pairs, inputs):
        y_true, y_pred = [], []

        for (d, w) in weight_date_pairs:
            lstm_p = predict_with_lstm(inputs["X_seq"], inputs["static_feat"])
            bilstm_p = predict_with_bilstm(inputs["X_seq"], inputs["static_feat"])
            lr_p = predict_with_lr(inputs["flat_seq"], inputs["log_predict_days"], inputs["age_ratio"])

            y_true.append(float(w))
            y_pred.append({
                "Bilstm_attention": lstm_p,
                "Bilstm_no_attention": bilstm_p,
                "linear_regression": lr_p
            })

        # 逐模型計算誤差
        metrics = {}
        for model_name in ["Bilstm_attention", "Bilstm_no_attention", "linear_regression"]:
            preds = [p[model_name] for p in y_pred]
            mse_v = float(mean_squared_error(y_true, preds))
            mae_v = float(mean_absolute_error(y_true, preds))
            rmse_v = float(np.sqrt(mse_v))
            metrics[model_name] = {
                "MSE": mse_v,
                "MAE": mae_v,
                "RMSE": rmse_v,
                "n_points_used": len(y_true)
            }

        return metrics

    model_inputs = {
        "X_seq": X_seq[0].tolist(), # shape (4, 2) 最近4天的序列特徵
        "static_feat": static_feat, # shape (6,) 靜態特徵
        "flat_seq": flat_seq,       # 攤平成一維的序列資料 (給LR用)
        "log_predict_days": log_predict_days, # 每筆對應的 log(days)
        "age_ratio": age_ratio      # 每筆對應的 age_ratio
    }
    # --- 評估指標 ---
    metrics = compute_single_step_metrics(weight_date_pairs, model_inputs) # weight_date_pairs = [(datetime, weight), ...]
    
    # ---- 更新 record：把 preds 轉成純 Python list (float)，並另外存 last_value ----
    def to_float_list(arr):
        # 將可能為 numpy array / list / tuple 的 preds 轉為 list of native floats
        return [float(x) for x in (np.asarray(arr).ravel().tolist())]

    lstm_preds_list = to_float_list(lstm_preds)
    bilstm_preds_list = to_float_list(bilstm_preds)
    lr_preds_list = to_float_list(lr_preds)

    record = {
        "model": model_type,
        "earnum": to_native(input_data[0]),
        # 儲存整段預測（list）
        "lstm_prediction_series": lstm_preds_list,
        "bilstm_prediction_series": bilstm_preds_list,
        "linear_regression_series": lr_preds_list,
        "metrics": metrics,
        # 儲存 summary scalar（例如最後一個預測值）
        "lstm_prediction_last": lstm_preds_list[-1] if len(lstm_preds_list) > 0 else None,
        "bilstm_prediction_last": bilstm_preds_list[-1] if len(bilstm_preds_list) > 0 else None,
        "linear_regression_last": lr_preds_list[-1] if len(lr_preds_list) > 0 else None,
        "user_id": user_id,
        "user_name": user_name,
        "user_role": user_role,
        "timestamp": datetime.now()
    }
    history_collection.insert_one(record)

    # --- 回傳 JSON ---
    result_payload = {
        "model": model_type,
        "earnum": to_native(input_data[0]),
        "predictions": {
            "Bi-LSTM_Attention": [{"date": d.isoformat(), "days": (d - self_birmeadate).days, "weight": float(w)} for d,w in zip(lstm_dates, lstm_preds)],
            "Bi-LSTM_no_Attention": [{"date": d.isoformat(), "days": (d - self_birmeadate).days, "weight": float(w)} for d,w in zip(bilstm_dates, bilstm_preds)],
            "Linear_Regression": [{"date": d.isoformat(), "days": (d - self_birmeadate).days, "weight": float(w)} for d,w in zip(lr_dates, lr_preds)]
        },
        "metrics": metrics,
        "actual": actual_series
        # "growth_curve_base64": growth_curve_base64
    }

    return result_payload

@app.get("/get_sheep_list")
def get_sheep_list():
    # 全部轉成字串，避免 numpy 型別混入
    ids = df["self_EarNum"].dropna().astype(str).unique().tolist()
    return JSONResponse(content=ids)

@app.get("/get_sheep_data")
def get_sheep_data(earnum: str = Query(...), current_user = Depends(get_current_user)):
    # 以字串比對，避免 df 裡是數字型別時比對失敗
    rows = df
    rows = rows[rows["self_EarNum"].astype(str) == str(earnum)].sort_values(by="self_MeaDate")
    if rows.empty:
        raise HTTPException(status_code=404, detail="Ear number not found")

    # 取第一列當其他欄位來源
    row = rows.iloc[0]

    # 取前 5 筆 (weight, date)，不足補 None
    # 注意順序：你的 /predict 讀的是 weight 在前、date 在後
    pairs = rows[["self_Weight", "self_MeaDate"]].head(5).values.tolist()
    while len(pairs) < 5:
        pairs.append([None, None])

    # 攤平成 [w1, d1, w2, d2, ...]
    flat_pairs = []
    for w, d in pairs:
        flat_pairs.append(as_float(w))
        flat_pairs.append(as_date_str(d))

    # 建構 input_data（固定長度與索引位置）
    input_data = [
        as_str(row.get("self_EarNum")),          # 0
        as_float(row.get("self_BirWeight")),     # 1
        as_date_str(row.get("self_MeaDate")),    # 2 這裡若你有出生日期欄，改成那個欄位
        *flat_pairs,                             # 3..12 (5 組)
        as_str(row.get("self_Breed")),           # 13
        as_str(row.get("dam_EarNum")),           # 14
        as_str(row.get("dam_Breed")),            # 15
        as_str(row.get("sire_EarNum")),          # 16
        as_str(row.get("sire_Breed")),           # 17
        # 不提供 18（predict_days），/predict 會 fallback = 30
    ]

    # 用 JSONResponse，確保是純 JSON 序列化
    return JSONResponse(content=input_data)

@app.post("/genePredict")
async def upload_csv(file: UploadFile = File(...), current_user = Depends(get_current_user)):
    db = mongo_client['user_accounts']
    history_collection = db['gene_prediction']
    user_id = current_user.get("sub")
    user_name = current_user.get("username")
    user_role = current_user.get("role")
    # 用 FastAPI 提供的 UploadFile 讀檔案
    df = pd.read_csv(file.file)

    # ========= 你的分析流程 =========
    df = df[['ID', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']]
    df['ID'] = df['ID'].astype(str).str.strip()
    df.set_index('ID', inplace=True)

    group1 = ['C1', 'C2', 'C3']
    group0 = ['C4', 'C5', 'C6']
    
    expr = df[group1 + group0]
    keep = (expr >= 1).sum(axis=1) >= 2
    df_f = df.loc[keep]

    log2fc = np.log2(df_f[group1].mean(axis=1) + 1) - np.log2(df_f[group0].mean(axis=1) + 1)
    t_stat, pval = ttest_ind(df_f[group1].T, df_f[group0].T, axis=0, equal_var=False)
    rej, qval, _, _ = multipletests(pval, method='fdr_bh')

    sig = (np.abs(log2fc) > 1) & (qval < 0.05)
    sig_genes = df_f.index[sig]

    # 火山圖
    # plt.figure(figsize=(6, 5))
    # plt.scatter(log2fc, -np.log10(pval), c='gray', alpha=0.5, s=10)
    # plt.scatter(log2fc[sig], -np.log10(pval[sig]), c='red', s=10)
    # plt.axvline(x=1, color='purple', linestyle='--', label = "2X higher in high fertility")
    # plt.axvline(x=-1, color='blue', linestyle='--', label = "2X lower in low fertility")
    # plt.axhline(y=-np.log10(0.05), color='green', linestyle='--', label = "p_value threshold")
    # plt.xlabel("繁殖率組基因表達(high fertility v.s low fertility)")
    # plt.ylabel("顯著性(p_value)")
    # plt.title("Volcano Plot")
    # plt.legend(loc="upper left")   # <<< 加這個才會顯示 label

    # buf = BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)
    # img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    # plt.close()

    # 建立結果 DataFrame（給火山圖）
    results_df = pd.DataFrame({
        "gene": df_f.index,
        "log2FC": log2fc,
        "pval": pval,
        "qval": qval,
        "negLog10P": -np.log10(pval),
        "significant": sig
    })

    # 準備 Highcharts 用的資料
    volcano_data = [
        {
            "gene": row['gene'],
            "x": float(row['log2FC']),
            "y": float(row['negLog10P']),
            "significant": bool(row['significant'])
        }
        for _, row in results_df.iterrows()
    ]

    # --- 如果沒有顯著基因，直接回傳 ---
    if len(sig_genes) == 0:
        return {
            "sig_gene_count": 0,
            "top_genes": [],
            "volcano_data": volcano_data
        }

    # 保留顯著基因表達值
    X_filtered = expr.loc[sig_genes].values.T  # (samples, genes)
    X_reduced = svd.transform(X_filtered)      # SVD 降維
    X_scaled = scaler.transform(X_reduced)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        _ = model(X_tensor, capture_attn=True)
        attn = model.last_attn  # (B, H, L, L)

    # CLS -> 特徵注意力
    cls2feat = attn[:, :, 0, 1:]  # (B, H, F)
    comp_importance = cls2feat.mean(dim=(0, 1)).cpu().numpy()
    comp_importance /= comp_importance.sum() + 1e-12

    # SVD 分量映射回基因
    Vt = svd.components_  # (n_components, n_genes)
    gene_importance = np.abs(comp_importance[:, None] * Vt).sum(axis=0)

    # 1) 建立 mapping 並取 log2fc_filtered
    log2fc_map = dict(zip(sig_genes, log2fc))
    log2fc_filtered = np.array([log2fc_map.get(g, np.nan) for g in sig_genes])

    # 2) 移除 missing
    valid_mask = ~np.isnan(log2fc_filtered)
    gene_names = np.array(sig_genes)[valid_mask]
    gene_importance = np.array(gene_importance)[valid_mask]
    log2fc_filtered = log2fc_filtered[valid_mask]

    # 3) 分成高、低繁殖率
    mask_high = log2fc_filtered > 0
    mask_low  = log2fc_filtered < 0

    # 高繁殖率 top10
    idx_high = np.argsort(-gene_importance[mask_high])[:10]
    top_high = [
        {"gene": g, "score": float(s), "log2FC": float(fc)}
        for g, s, fc in zip(
            gene_names[mask_high][idx_high],
            gene_importance[mask_high][idx_high],
            log2fc_filtered[mask_high][idx_high]
        )
    ]

    # 低繁殖率 top10
    idx_low = np.argsort(-gene_importance[mask_low])[:10]
    top_low = [
        {"gene": g, "score": float(s), "log2FC": float(fc)}
        for g, s, fc in zip(
            gene_names[mask_low][idx_low],
            gene_importance[mask_low][idx_low],
            log2fc_filtered[mask_low][idx_low]
        )
    ]

    record = {
        "timestamp": datetime.now(),
        "sig_gene_count": len(sig_genes),
        "top_high_genes": top_high,
        "top_low_genes": top_low,
        "user_id": user_id,
        "user_name": user_name,
        "user_role": user_role
    }

    history_collection.insert_one(record)

    # 存入資料庫
    # history_collection.insert_one({
    #     "timestamp": datetime.now(),
    #     **record
    # })

    # 回傳 API
    return {
        "sig_gene_count": len(sig_genes),
        "top_high_genes": top_high,
        "top_low_genes": top_low,
        "volcano_data": volcano_data
    }

@app.post("/get_personal_data")
async def get_personal_data(
    request: Request,
    current_user=Depends(get_current_user)
):
    # current_user 從 token 解出來
    user_id = current_user.get("sub")
    role = current_user.get("role")

    if not user_id or not role:
        raise HTTPException(status_code=400, detail="使用者資料不完整")

    # 根據角色選擇資料庫
    db = mongo_client['user_accounts']
    if role == "Farmer":
        collection = db['growth_prediction']
    elif role == "GeneticResearcher":
        collection = db['gene_prediction']
    else:
        raise HTTPException(status_code=400, detail="角色錯誤")

    # 從資料庫過濾 user_id
    data_cursor = collection.find({"user_id": user_id}, {"_id": 0})
    data = list(data_cursor)

    if not data:
        raise HTTPException(status_code=404, detail="找不到歷史資料")

    return {"data": data}
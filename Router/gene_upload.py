from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends, Body
from fastapi.responses import JSONResponse
from auth import get_current_user
from bson import ObjectId
from pymongo import MongoClient
from main import mongo_client
import pandas as pd
import io, os, math, re

gene_router = APIRouter()

def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

# 統一排除 user_id 的 function
def exclude_user_id(data: list[dict]) -> list[dict]:
    """
    輸入 MongoDB 查詢結果 (list of dict)，自動移除 user_id 欄位
    """
    return [{k: v for k, v in d.items() if k != "user_id"} for d in data]

@gene_router.post("/upload")
async def upload_gene_data(request: Request, file: UploadFile = File(...), current_user=Depends(get_current_user)):
    try:
        # 取得 mongo client
        mongo_client = request.app.state.mongo_client
        db = mongo_client["DNAdata"]

        # 檢查檔案格式
        filename = file.filename
        name_without_ext = os.path.splitext(filename)[0]
        # 將非法字元替換成下劃線
        collection_name = "".join(c if c.isalnum() or c=="_" else "_" for c in name_without_ext)

        # 讀檔
        contents = await file.read()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="只支援 CSV 或 Excel 格式")

        records = df.to_dict(orient="records")
        if not records:
            raise HTTPException(status_code=400, detail="檔案內沒有資料")
        
        # 每筆加上 user_id 欄位
        user_id = current_user["sub"]
        for r in records:
            r["user_id"] = user_id

        # 使用檔名作為 collection
        gene_collection = db[collection_name]
        result = gene_collection.insert_many(records)

        return {"message": "匯入成功", "collection": collection_name, "insertedCount": len(result.inserted_ids)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@gene_router.get("/collections")
async def list_collections(request: Request, current_user=Depends(get_current_user)):
    db = request.app.state.mongo_client["DNAdata"]
    user_id = current_user["sub"]

    # 只回傳該 user 匯入過的 collection
    collections = []
    for name in db.list_collection_names():
        col = db[name]
        if col.find_one({"user_id": user_id}):
            collections.append(name)
    return collections

@gene_router.get("/data/{collection_name}")
async def get_collection_data(request: Request, collection_name: str, current_user=Depends(get_current_user)):
    try:
        db = request.app.state.mongo_client["DNAdata"]
        user_id = current_user["sub"]

        # 嘗試抓資料
        raw_data = list(db[collection_name].find({"user_id": user_id}))
        data = []

        for d in raw_data:
            d["_id"] = str(d["_id"])      # _id 轉成字串
            d.pop("user_id", None)        # 不回傳 user_id
            data.append(d)
        
        data = clean_json(data)  # 清理 NaN / inf

        # 欄位名稱
        fields = [f for f in data[0].keys() if f not in ["_id"]] if data else []

        return {"fields": fields, "data": data}

    except Exception as e:
        # 錯誤回傳 JSON
        import traceback
        print("載入資料集失敗:", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"無法載入資料集 {collection_name}: {str(e)}"}
        )


# 更新單筆資料
@gene_router.put("/update/{collection_name}/{doc_id}")
async def update_record(request: Request, collection_name: str, doc_id: str, payload: dict = Body(...), current_user=Depends(get_current_user)):
    db = request.app.state.mongo_client["DNAdata"]
    user_id = current_user["sub"]

    result = db[collection_name].update_one(
        {"_id": ObjectId(doc_id), "user_id": user_id},
        {"$set": payload}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="資料未找到或無法更新")

    return {"message": "更新成功"}

# 新增單筆資料
@gene_router.post("/manual_add/{currentCollection}")
async def add_record(
    request: Request,
    currentCollection: str,       # 與 URL path 對應
    payload: dict = Body(...),
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["DNAdata"]
    user_id = current_user["sub"]

    payload["user_id"] = user_id
    result = db[currentCollection].insert_one(payload)
    return {"message": "新增成功", "_id": str(result.inserted_id)}


# 刪除資料集
@gene_router.delete("/delete_collection/{collection_name}")
async def delete_collection(collection_name: str, request: Request, current_user=Depends(get_current_user)):
    db = request.app.state.mongo_client["DNAdata"]
    user_id = current_user["sub"]

    # 確認 collection 有資料屬於該 user
    if db[collection_name].find_one({"user_id": user_id}) is None:
        raise HTTPException(status_code=404, detail="資料集不存在或無權限刪除")

    # 刪除整個 collection
    db.drop_collection(collection_name)
    return {"message": f"資料集 {collection_name} 已刪除"}

# 刪除一筆資料
@gene_router.delete("/delete/{collection_name}/{doc_id}")
async def delete_record(
    request: Request,
    collection_name: str,
    doc_id: str,
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["DNAdata"]
    user_id = current_user["sub"]

    result = db[collection_name].delete_one({"_id": ObjectId(doc_id), "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="資料未找到或無法刪除")

    return {"message": "刪除成功"}

# 刪除欄位
@gene_router.delete("/delete_field/{collection_name}/{field}")
async def delete_field(
    request: Request,
    collection_name: str,
    field: str,
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["DNAdata"]
    user_id = current_user["sub"]

    # 移除所有該欄位
    db[collection_name].update_many(
        {"user_id": user_id},
        {"$unset": {field: ""}}
    )
    return {"message": f"欄位 {field} 已刪除"}

# 更新欄位順序
@gene_router.post("/update_field_order/{collection_name}")
async def update_field_order(
    request: Request,
    collection_name: str,
    body: dict = Body(...),
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["DNAdata"]
    config = db["FieldConfig"]
    user_id = current_user["sub"]

    new_order = body.get("fields", [])
    if not new_order:
        raise HTTPException(status_code=400, detail="欄位順序不可為空")

    config.update_one(
        {"collection": collection_name, "user_id": user_id},
        {"$set": {"fields": new_order}},
        upsert=True
    )
    return {"message": "欄位順序已更新"}

# 搜尋
@gene_router.get("/search_collections")
async def get_user_collections(request: Request, current_user=Depends(get_current_user)):
    username = current_user.get("username")
    user_id = current_user.get("sub")  # JWT sub
    role = current_user.get("role", "")

    collections = []

    # FarmData 中使用者匯入的資料集
    farm_db = request.app.state.mongo_client["DNAdata"]
    for name in farm_db.list_collection_names():
        col = farm_db[name]
        if col.find_one({"user_id": user_id}):
            collections.append({"name": name, "label": name})

    ua_db = request.app.state.mongo_client["user_accounts"]
    if "gene_prediction" in ua_db.list_collection_names() and username:
        col = ua_db["gene_prediction"]
        if col.find_one({"user_name": username}):
            collections.append({"name": "gene_prediction", "label": "基因預測的歷史資料"})
            
    print("USERNAME", username)
    return {"collections": collections, "role": role}


# 多欄位模糊搜尋
@gene_router.post("/search/{collection_name}")
async def search_collection(request: Request, collection_name: str, body: dict, current_user=Depends(get_current_user)):
    username = current_user.get("username") or current_user.get("user_name")
    user_id = current_user.get("sub")
    user_role = current_user.get("role", "")

    keyword = body.get("keyword", "").strip()
    fields = body.get("fields", [])

    gene_db = request.app.state.mongo_client["DNAdata"]
    user_collections = [name for name in gene_db.list_collection_names() if gene_db[name].find_one({"user_id": user_id})]
    
    # 權限控制
    role_allowed_collections = {
        "Farmer": ["growth_prediction"] + user_collections,
        "GeneticResearcher": ["DNAdata", "gene_prediction"]
    }
    allowed_collections = role_allowed_collections.get(user_role, [])
    if collection_name not in allowed_collections and collection_name != "gene_prediction":
        raise HTTPException(status_code=403, detail="沒有權限查詢此資料集")

    # 判斷資料庫
    if collection_name == "gene_prediction":
        db = request.app.state.mongo_client["user_accounts"]
        query = {"user_name": username}
    else:
        db = request.app.state.mongo_client["DNAdata"]
        query = {"user_id": user_id}

    # 加入搜尋條件
    if keyword and fields:
        query["$or"] = [{f: {"$regex": re.escape(keyword), "$options": "i"}} for f in fields]

    collection = db[collection_name]
    raw_data = list(collection.find(query, {"_id": 0}))
    data = exclude_user_id(raw_data)
    data = clean_json(data or [])

    return data
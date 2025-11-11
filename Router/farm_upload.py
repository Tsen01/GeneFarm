from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends, Body
from fastapi.responses import JSONResponse
from auth import get_current_user
from bson import ObjectId
from pymongo import MongoClient
from main import mongo_client
from typing import Dict, Any
import pandas as pd
import io, os, math, re

farm_router = APIRouter()

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

@farm_router.post("/upload")
async def upload_farm_data(request: Request, file: UploadFile = File(...), current_user=Depends(get_current_user)):
    try:
        # 改為 FarmData
        db = request.app.state.mongo_client["FarmData"]

        filename = file.filename
        name_without_ext = os.path.splitext(filename)[0]
        collection_name = "".join(c if c.isalnum() or c=="_" else "_" for c in name_without_ext)

        contents = await file.read()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="只支援 CSV 或 Excel 格式")

        if df.empty:
            raise HTTPException(status_code=400, detail="檔案內沒有資料")

        records = df.to_dict(orient="records")

        # 加上 user_id
        user_id = current_user["sub"]
        for record in records:
            record["user_id"] = user_id

        collection = db[collection_name]
        result = collection.insert_many(records)

        return {
            "message": "匯入成功",
            "collection": collection_name,
            "insertedCount": len(result.inserted_ids)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 手動新增資料
@farm_router.post("/manual_add/{collection_name}")
async def manual_add_data(
    request: Request,
    collection_name: str,
    body: dict = Body(...),
    current_user=Depends(get_current_user)
):
    try:
        db = request.app.state.mongo_client["FarmData"]
        user_id = current_user["sub"]

        # 強制加上 user_id
        body["user_id"] = user_id  

        collection = db[collection_name]
        collection.insert_one(body)

        return {"message": "新增成功", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 手動新增資料(現有資料集): 自動顯示該資料集的所有欄位
@farm_router.get("/fields/{collection_name}")
async def get_collection_fields(request: Request, collection_name: str, current_user=Depends(get_current_user)):
    db = request.app.state.mongo_client["FarmData"]
    user_id = current_user["sub"]

    sample = db[collection_name].find_one({"user_id": user_id}, {"_id": 0, "user_id": 0})
    if not sample:
        return []
    
    # 自動排除 user_id
    return [f for f in sample.keys() if f != "user_id"]

# 新增單筆資料
@farm_router.post("/manual_add/{currentCollection}")
async def add_record(
    request: Request,
    currentCollection: str,       # 與 URL path 對應
    payload: dict = Body(...),
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["FarmData"]
    user_id = current_user["sub"]

    payload["user_id"] = user_id
    result = db[currentCollection].insert_one(payload)
    return {"message": "新增成功", "_id": str(result.inserted_id)}

@farm_router.get("/collections")
async def list_collections(request: Request, current_user=Depends(get_current_user)):
    db = request.app.state.mongo_client["FarmData"]
    user_id = current_user["sub"]

    # 只回傳該 user 匯入過的 collection
    collections = []
    for name in db.list_collection_names():
        col = db[name]
        if col.find_one({"user_id": user_id}):
            collections.append(name)
    return collections

# 選擇資料集
@farm_router.get("/data/{collection_name}")
async def get_collection_data(request: Request, collection_name: str, current_user=Depends(get_current_user)):
    try:
        db = request.app.state.mongo_client["FarmData"]
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


# 修改一筆資料
@farm_router.put("/update/{collection_name}/{doc_id}")
async def update_record(
    request: Request,
    collection_name: str,
    doc_id: str,
    body: dict = Body(...),
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["FarmData"]
    user_id = current_user["sub"]

    result = db[collection_name].update_one(
        {"_id": ObjectId(doc_id), "user_id": user_id},
        {"$set": body}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="資料未找到或無法更新")

    return {"message": "更新成功"}


# 刪除資料集
@farm_router.delete("/delete_collection/{collection_name}")
async def delete_collection(collection_name: str, request: Request, current_user=Depends(get_current_user)):
    db = request.app.state.mongo_client["FarmData"]
    user_id = current_user["sub"]

    # 確認 collection 有資料屬於該 user
    if db[collection_name].find_one({"user_id": user_id}) is None:
        raise HTTPException(status_code=404, detail="資料集不存在或無權限刪除")

    # 刪除整個 collection
    db.drop_collection(collection_name)
    return {"message": f"資料集 {collection_name} 已刪除"}

# 刪除一筆資料
@farm_router.delete("/delete/{collection_name}/{doc_id}")
async def delete_record(
    request: Request,
    collection_name: str,
    doc_id: str,
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["FarmData"]
    user_id = current_user["sub"]

    result = db[collection_name].delete_one({"_id": ObjectId(doc_id), "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="資料未找到或無法刪除")

    return {"message": "刪除成功"}

# 刪除欄位
@farm_router.delete("/delete_field/{collection_name}/{field}")
async def delete_field(
    request: Request,
    collection_name: str,
    field: str,
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["FarmData"]
    user_id = current_user["sub"]

    # 移除所有該欄位
    db[collection_name].update_many(
        {"user_id": user_id},
        {"$unset": {field: ""}}
    )
    return {"message": f"欄位 {field} 已刪除"}

# 更新欄位順序
@farm_router.post("/update_field_order/{collection_name}")
async def update_field_order(
    request: Request,
    collection_name: str,
    body: dict = Body(...),
    current_user=Depends(get_current_user)
):
    db = request.app.state.mongo_client["FarmData"]
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


@farm_router.get("/search_collections")
async def get_user_collections(request: Request, current_user=Depends(get_current_user)):
    username = current_user.get("username")
    user_id = current_user.get("sub")  # JWT sub
    role = current_user.get("role", "")

    collections = []

    # FarmData 中使用者匯入的資料集
    farm_db = request.app.state.mongo_client["FarmData"]
    for name in farm_db.list_collection_names():
        col = farm_db[name]
        if col.find_one({"user_id": user_id}):
            collections.append({"name": name, "label": name})

    ua_db = request.app.state.mongo_client["user_accounts"]
    if "growth_prediction" in ua_db.list_collection_names() and username:
        col = ua_db["growth_prediction"]
        if col.find_one({"user_name": username}):
            collections.append({"name": "growth_prediction", "label": "成長預測的歷史資料"})
            
    print("USERNAME", username)
    return {"collections": collections, "role": role}


# 多欄位模糊搜尋
@farm_router.post("/search/{collection_name}")
async def search_collection(request: Request, collection_name: str, body: dict, current_user=Depends(get_current_user)):
    username = current_user.get("username") or current_user.get("user_name")
    user_id = current_user.get("sub")
    user_role = current_user.get("role", "")

    keyword = body.get("keyword", "").strip()
    fields = body.get("fields", [])

    farm_db = request.app.state.mongo_client["FarmData"]
    user_collections = [name for name in farm_db.list_collection_names() if farm_db[name].find_one({"user_id": user_id})]
    
    # 權限控制
    role_allowed_collections = {
        "Farmer": ["growth_prediction"] + user_collections,
        "GeneticResearcher": ["DNAdata", "gene_prediction"]
    }
    allowed_collections = role_allowed_collections.get(user_role, [])
    if collection_name not in allowed_collections and collection_name != "growth_prediction":
        raise HTTPException(status_code=403, detail="沒有權限查詢此資料集")

    # 判斷資料庫
    if collection_name == "growth_prediction":
        db = request.app.state.mongo_client["user_accounts"]
        query = {"user_name": username}
    else:
        db = request.app.state.mongo_client["FarmData"]
        query = {"user_id": user_id}

    # 加入搜尋條件
    if keyword and fields:
        query["$or"] = [{f: {"$regex": re.escape(keyword), "$options": "i"}} for f in fields]

    collection = db[collection_name]
    raw_data = list(collection.find(query, {"_id": 0}))
    data = exclude_user_id(raw_data)
    data = clean_json(data or [])

    return data
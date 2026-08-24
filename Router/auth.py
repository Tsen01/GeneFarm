from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, EmailStr
from uuid import uuid4
from database import create_user_collections
from utils import hash_password, verify_password, create_token, decode_token
# from main import mongo_client

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = Field(..., pattern="^(Farmer|GeneticResearcher)$")

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    token: str
    role: str
    user_id: str
    email: EmailStr
    username: str = None
    farmname: str = None

auth_router = APIRouter()

security = HTTPBearer()

def get_current_user(token: str = Depends(security)):
    try:
        payload = decode_token(token.credentials)
        return payload  # 內含 sub(user_id), role, email, username
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# 註冊
@auth_router.post("/register", response_model=TokenResponse)
def register(request: Request, body: RegisterRequest):

    # 取得 FastAPI 啟動時建立的 MongoDB Client
    mongo_client = request.app.state.mongo_client

    # 確認 MongoDB Client 是否已正確初始化
    if not mongo_client:
        raise RuntimeError("MongoDB client register 未正確初始化")

    # 使用 user_accounts 資料庫
    db = mongo_client['user_accounts']

    # 確保 Email 欄位具有唯一索引，避免相同 Email 重複註冊
    db.users.create_index("email", unique=True)

    # 檢查 Email 是否已經註冊
    if db.users.find_one({"email": body.email}):
        raise HTTPException(status_code=400, detail="此 Email 已被註冊，請登入或使用其他 Email。")

    # 建立唯一的使用者 ID
    user_id = str(uuid4())
    
    # 將使用者密碼進行雜湊後再儲存
    hashed_pw = hash_password(body.password)

    # 將使用者基本資料寫入 users collection
    db.users.insert_one({
        "username": body.username,
        "email":body.email,
        "password": hashed_pw,
        "role": body.role,
        "user_id": user_id
    })

    # 建立該使用者專屬的 MongoDB collections
    create_user_collections(mongo_client, user_id, body.role)

    # 建立 JWT Token，將使用者基本資訊寫入 Token
    token = create_token({
        "sub": user_id, 
        "username": body.username,
        "role": body.role, 
        "email": body.email
    })

    return TokenResponse(
        token=token,
        role=body.role,
        user_id=user_id,
        email=body.email,
        username=body.username
    )

# 登入
@auth_router.post("/login", response_model=TokenResponse)
async def login(request: Request, body: LoginRequest):
    try:
        # 取得 FastAPI 啟動時建立的 MongoDB Client
        mongo_client = request.app.state.mongo_client

        # 確認 MongoDB Client 是否已正確初始化
        if not mongo_client:
            raise RuntimeError("MongoDB client login 未正確初始化")

        # 使用 user_accounts 資料庫
        db = mongo_client['user_accounts']

        # 根據 Email 查詢使用者
        user = db.users.find_one({"email": body.email})
        if not user:
            raise HTTPException(status_code=401, detail="無此帳號，請先註冊！")
        
        if not verify_password(body.password, user['password']):
            raise HTTPException(status_code=401, detail="密碼錯誤，請重新輸入！")

        token = create_token({
            "sub": user['user_id'],
            "username": user['username'],
            "role": user['role'],
            "email": user['email']
        })
        
        return TokenResponse(
            token=token,
            role=user['role'],
            user_id=user['user_id'],
            email=user['email'],
            username=user['username']
        )
    except HTTPException:
        raise # 已經處理過的 HTTPException，直接拋出
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@auth_router.get("/me", response_model=TokenResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return TokenResponse(
        token="",  # /me 不需要重新產生 Token，因此回傳空字串
        role=current_user["role"],
        user_id=current_user["sub"],
        email=current_user["email"],
        username=current_user.get("username"),
    )

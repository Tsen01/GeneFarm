from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, EmailStr
from uuid import uuid4
from database import create_user_collections
from utils import hash_password, verify_password, create_token, decode_token
from main import mongo_client

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
        print("Token payload: ", payload)
        return payload  # 內含 sub(user_id), role, email, username
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@auth_router.post("/register", response_model=TokenResponse)
def register(request: Request, body: RegisterRequest):
    print("原始請求內容:", body.json())
    mongo_client = request.app.state.mongo_client
    db = mongo_client['user_accounts']
    if not mongo_client:
        raise RuntimeError("MongoDB client register 未正確初始化")

    # 確保 email 欄位具有唯一索引（只會建立一次）
    db.users.create_index("email", unique=True)

    if db.users.find_one({"email": body.email}):
        raise HTTPException(status_code=400, detail="此 Email 已被註冊，請登入或使用其他 Email。")

    user_id = str(uuid4())
    hashed_pw = hash_password(body.password)

    db.users.insert_one({
        "username": body.username,
        "email":body.email,
        "password": hashed_pw,
        "role": body.role,
        "user_id": user_id
    })

    create_user_collections(mongo_client, user_id, body.role)

    print("register 接收到：", body)
    print("username:", body.username)
    print("email:", body.email)
    print("password:", body.password)
    print("role:", body.role)

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


@auth_router.post("/login", response_model=TokenResponse)
async def login(request: Request, body: LoginRequest):
    try:
        mongo_client = request.app.state.mongo_client
        db = mongo_client['user_accounts']
        if not mongo_client:
            raise RuntimeError("MongoDB client login 未正確初始化")
        print("成功連線到 goat_project")
        print("解析登入資料:", body)
        
        user = db.users.find_one({"email": body.email})
        if not user:
            print(f"❌ 查無此使用者：{body.email}")
            raise HTTPException(status_code=401, detail="無此帳號")
        
        if not verify_password(body.password, user['password']):
            raise HTTPException(status_code=401, detail="密碼錯誤，請重新輸入！")

        token = create_token({
            "sub": user['user_id'],
            "username": user['username'],
            "role": user['role'],
            "email": user['email']
        })
        print("user =", user)
        return TokenResponse(
            token=token,
            role=user['role'],
            user_id=user['user_id'],
            email=user['email'],
            username=user['username']
        )
    except Exception as e:
        print("登入錯誤：", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@auth_router.get("/me", response_model=TokenResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return TokenResponse(
        token="",  # /me 不需要再發 token，回空字串就好
        role=current_user["role"],
        user_id=current_user["sub"],
        email=current_user["email"],
        username=current_user.get("username"),
    )
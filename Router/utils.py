from passlib.context import CryptContext
from jose import jwt
import os
from datetime import datetime, timedelta, timezone

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60

if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY 未設定")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed: str):
    return pwd_context.verify(plain_password, hashed)

def create_token(data: dict):
    to_encode = data.copy()
    
    # 設定 JWT Token 的過期時間
    # expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=JWT_EXPIRE_MINUTES
    )
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token: str):
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

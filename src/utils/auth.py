import os
import secrets
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import Request, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Configuration
SECRET_KEY = os.getenv("AUTH_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def hash_password(password: str) -> str:
    """Hash a password using bcrypt directly (avoids passlib version conflicts)."""
    # Bcrypt has a 72-character limit.
    pwd_bytes = password[:72].encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash using bcrypt directly."""
    try:
        return bcrypt.checkpw(
            plain_password[:72].encode('utf-8'), 
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get the current user from the cookie session."""
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    if token.startswith("Bearer "):
        token = token[7:]
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            return None
        return {"username": username, "role": role}
    except JWTError:
        return None

def login_required(user: Optional[Dict[str, Any]]):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required to access this resource",
        )
    return user

def admin_required(user: Optional[Dict[str, Any]]):
    # Allow both generic admin and super_admin roles
    if not user or user.get("role") not in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user

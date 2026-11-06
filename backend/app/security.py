import os
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, Request, status
import jwt
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session
from .db import get_db
from .models import SessionToken, User

hasher = PasswordHash.recommended()
ROLES = {"admin", "researcher", "analyst", "viewer"}
SECRET = os.getenv("JWT_SECRET", "")

def ensure_secret():
    if len(SECRET) < 32:
        raise RuntimeError("JWT_SECRET debe contener al menos 32 caracteres")

def create_token(user: User) -> tuple[str, str, datetime]:
    ensure_secret()
    jti = secrets.token_hex(24)
    expiry = datetime.now(timezone.utc) + timedelta(hours=8)
    payload = {"sub": str(user.id), "jti": jti, "exp": expiry}
    return jwt.encode(payload, SECRET, algorithm="HS256"), jti, expiry

def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get("session")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sesión requerida")
    try:
        claims = jwt.decode(token, SECRET, algorithms=["HS256"])
        user_id = int(claims["sub"])
        jti = str(claims["jti"])
    except (jwt.PyJWTError, ValueError, KeyError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sesión inválida")
    user = db.get(User, user_id)
    if not user or not user.active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Usuario inactivo")
    session = db.scalar(select(SessionToken).where(SessionToken.jti == jti, SessionToken.user_id == user_id))
    if not session or session.revoked:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sesión cerrada")
    request.state.jti = jti
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        csrf_cookie = request.cookies.get("csrf_token")
        if not csrf_cookie or request.headers.get("X-CSRF-Token") != csrf_cookie:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Token CSRF requerido")
    return user

def require_roles(*roles: str):
    def dependency(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Permiso insuficiente")
        return user
    return dependency

def user_json(user: User):
    return {"id": user.id, "email": user.email, "name": user.name, "role": user.role, "active": user.active}

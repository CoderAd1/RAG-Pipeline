"""Authentication endpoints using Supabase Auth."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.database import get_supabase
from loguru import logger

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthRequest(BaseModel):
    email: str
    password: str


@router.post("/signup")
async def signup(req: AuthRequest):
    supabase = get_supabase()
    try:
        res = supabase.auth.sign_up({"email": req.email, "password": req.password})
        if res.user:
            return {
                "user": {"id": str(res.user.id), "email": res.user.email},
                "access_token": res.session.access_token if res.session else None,
                "message": "Account created. Check your email to confirm." if not res.session else "Signed up successfully.",
            }
        raise HTTPException(status_code=400, detail="Signup failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(req: AuthRequest):
    supabase = get_supabase()
    try:
        res = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
        if res.user and res.session:
            return {
                "user": {"id": str(res.user.id), "email": res.user.email},
                "access_token": res.session.access_token,
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Invalid email or password")


@router.post("/logout")
async def logout():
    supabase = get_supabase()
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    return {"message": "Logged out"}


@router.get("/me")
async def me(authorization: str = ""):
    """Validate a token and return user info."""
    supabase = get_supabase()
    try:
        user = supabase.auth.get_user(authorization)
        if user and user.user:
            return {"user": {"id": str(user.user.id), "email": user.user.email}}
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

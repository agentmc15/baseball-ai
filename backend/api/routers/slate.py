"""Router placeholder"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_all():
    return {"message": "Not implemented yet"}

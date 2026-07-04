"""Kamera yönetimi (CRUD) endpoint'leri."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", response_model=list[schemas.CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    return db.query(models.Camera).order_by(models.Camera.id).all()


@router.post("", response_model=schemas.CameraOut, status_code=201)
def create_camera(payload: schemas.CameraCreate, db: Session = Depends(get_db)):
    cam = models.Camera(**payload.model_dump())
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


@router.get("/{camera_id}", response_model=schemas.CameraOut)
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = db.get(models.Camera, camera_id)
    if not cam:
        raise HTTPException(404, "Kamera bulunamadı")
    return cam


@router.patch("/{camera_id}", response_model=schemas.CameraOut)
def update_camera(
    camera_id: int, payload: schemas.CameraUpdate, db: Session = Depends(get_db)
):
    cam = db.get(models.Camera, camera_id)
    if not cam:
        raise HTTPException(404, "Kamera bulunamadı")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(cam, k, v)
    db.commit()
    db.refresh(cam)
    return cam


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    cam = db.get(models.Camera, camera_id)
    if not cam:
        raise HTTPException(404, "Kamera bulunamadı")
    db.delete(cam)
    db.commit()

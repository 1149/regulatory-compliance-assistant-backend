"""
Cleanup Management API Routes

Provides endpoints for managing the automatic file cleanup system.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from database import SessionLocal
from cleanup_service import cleanup_service, get_cleanup_stats

router = APIRouter(prefix="/api/cleanup", tags=["cleanup"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/stats")
async def get_cleanup_statistics():
    """
    Get statistics about the cleanup service and documents.
    """
    try:
        stats = await get_cleanup_stats()
        return {
            "status": "success",
            "data": stats,
            "service_running": cleanup_service.is_running,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cleanup statistics: {e}"
        )

@router.post("/manual")
async def trigger_manual_cleanup():
    """
    Manually trigger a cleanup operation.
    This will immediately delete all documents older than 2 hours.
    """
    try:
        cleanup_stats = await cleanup_service.manual_cleanup()
        return {
            "status": "success",
            "message": "Manual cleanup completed",
            "data": cleanup_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Manual cleanup failed: {e}"
        )

@router.get("/status")
async def get_cleanup_service_status():
    """
    Get the current status of the background cleanup service.
    """
    return {
        "service_running": cleanup_service.is_running,
        "cleanup_interval_hours": cleanup_service.cleanup_interval_hours,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/start")
async def start_cleanup_service():
    """
    Start the background cleanup service if it's not already running.
    """
    if cleanup_service.is_running:
        return {
            "status": "already_running",
            "message": "Cleanup service is already running",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # Note: In a production environment, you'd want to use a proper task queue
        # like Celery or run this in a separate process/container
        import asyncio
        asyncio.create_task(cleanup_service.start_background_cleanup())
        
        return {
            "status": "started",
            "message": "Background cleanup service started",
            "cleanup_interval_hours": cleanup_service.cleanup_interval_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start cleanup service: {e}"
        )

@router.post("/stop")
async def stop_cleanup_service():
    """
    Stop the background cleanup service.
    """
    cleanup_service.stop_background_cleanup()
    return {
        "status": "stopped",
        "message": "Background cleanup service stopped",
        "timestamp": datetime.utcnow().isoformat()
    }

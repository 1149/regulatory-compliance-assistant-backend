"""
Automated Cleanup Service for Regulatory Compliance Assistant

This service handles automatic deletion of uploaded files and their database records
after a specified time period (2 hours by default) to manage storage and maintain privacy.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import and_

from database import SessionLocal, Document, Entity
from file_utils import delete_file_safely

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cleanup_service")

class CleanupService:
    """
    Service responsible for automatically cleaning up old documents and files.
    """
    
    def __init__(self, cleanup_interval_hours: float = 2.0):
        """
        Initialize the cleanup service.
        
        Args:
            cleanup_interval_hours: Time in hours after which documents should be deleted
        """
        self.cleanup_interval_hours = cleanup_interval_hours
        self.cleanup_interval_seconds = cleanup_interval_hours * 3600
        self.is_running = False
        
    async def cleanup_old_documents(self, db: Session) -> dict:
        """
        Clean up documents and files older than the specified interval.
        
        Args:
            db: Database session
            
        Returns:
            dict: Cleanup statistics
        """
        try:
            # Calculate cutoff time
            cutoff_time = datetime.utcnow() - timedelta(hours=self.cleanup_interval_hours)
            
            # Find documents older than cutoff time
            old_documents = db.query(Document).filter(
                Document.upload_date < cutoff_time
            ).all()
            
            if not old_documents:
                logger.info(f"No documents older than {self.cleanup_interval_hours} hours found")
                return {
                    "status": "success",
                    "documents_deleted": 0,
                    "files_deleted": 0,
                    "entities_deleted": 0,
                    "errors": []
                }
            
            cleanup_stats = {
                "status": "success",
                "documents_deleted": 0,
                "files_deleted": 0,
                "entities_deleted": 0,
                "errors": []
            }
            
            for document in old_documents:
                try:
                    # Delete associated files
                    files_deleted = await self._delete_document_files(document)
                    cleanup_stats["files_deleted"] += files_deleted
                    
                    # Delete associated entities
                    entities_count = db.query(Entity).filter(
                        Entity.document_id == document.id
                    ).count()
                    
                    db.query(Entity).filter(Entity.document_id == document.id).delete()
                    cleanup_stats["entities_deleted"] += entities_count
                    
                    # Delete document record
                    db.delete(document)
                    cleanup_stats["documents_deleted"] += 1
                    
                    logger.info(
                        f"Cleaned up document {document.id} ({document.filename}) "
                        f"uploaded on {document.upload_date}"
                    )
                    
                except Exception as e:
                    error_msg = f"Error cleaning up document {document.id}: {str(e)}"
                    logger.error(error_msg)
                    cleanup_stats["errors"].append(error_msg)
                    # Continue with other documents even if one fails
                    continue
            
            # Commit all deletions
            db.commit()
            
            logger.info(
                f"Cleanup completed: {cleanup_stats['documents_deleted']} documents, "
                f"{cleanup_stats['files_deleted']} files, "
                f"{cleanup_stats['entities_deleted']} entities deleted"
            )
            
            return cleanup_stats
            
        except Exception as e:
            db.rollback()
            error_msg = f"Critical error during cleanup: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "documents_deleted": 0,
                "files_deleted": 0,
                "entities_deleted": 0
            }
    
    async def _delete_document_files(self, document: Document) -> int:
        """
        Delete all files associated with a document.
        
        Args:
            document: Document database record
            
        Returns:
            int: Number of files successfully deleted
        """
        files_deleted = 0
        
        # List of potential file paths to delete
        file_paths_to_delete = []
        
        # Main text file
        if document.path_to_text:
            file_paths_to_delete.append(Path(document.path_to_text))
            
            # Derive other file paths based on the text path
            text_path = Path(document.path_to_text)
            base_path = text_path.parent / text_path.stem
            
            # Original uploaded file (PDF, DOCX, etc.)
            for ext in ['.pdf', '.docx', '.doc', '.txt']:
                original_file = text_path.parent / f"{text_path.stem.replace('_raw', '')}{ext}"
                if original_file != text_path:  # Don't add the same file twice
                    file_paths_to_delete.append(original_file)
            
            # Raw text file
            raw_text_file = text_path.parent / f"{text_path.stem.replace('.txt', '')}_raw.txt"
            if raw_text_file != text_path:
                file_paths_to_delete.append(raw_text_file)
        
        # Delete each file
        for file_path in file_paths_to_delete:
            try:
                if file_path.exists():
                    delete_file_safely(str(file_path))
                    files_deleted += 1
                    logger.debug(f"Deleted file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
        
        return files_deleted
    
    async def start_background_cleanup(self):
        """
        Start the background cleanup task that runs periodically.
        """
        if self.is_running:
            logger.warning("Cleanup service is already running")
            return
        
        self.is_running = True
        logger.info(
            f"Starting background cleanup service - files will be deleted after "
            f"{self.cleanup_interval_hours} hours"
        )
        
        while self.is_running:
            try:
                # Create database session
                db = SessionLocal()
                try:
                    cleanup_stats = await self.cleanup_old_documents(db)
                    
                    # Log summary if any cleanup occurred
                    if cleanup_stats["documents_deleted"] > 0:
                        logger.info(f"Cleanup cycle completed: {cleanup_stats}")
                    
                finally:
                    db.close()
                
                # Wait before next cleanup cycle (check every 30 minutes)
                await asyncio.sleep(1800)  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in background cleanup cycle: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes
    
    def stop_background_cleanup(self):
        """
        Stop the background cleanup service.
        """
        self.is_running = False
        logger.info("Background cleanup service stopped")
    
    async def manual_cleanup(self) -> dict:
        """
        Perform a manual cleanup operation.
        
        Returns:
            dict: Cleanup statistics
        """
        logger.info("Performing manual cleanup...")
        db = SessionLocal()
        try:
            return await self.cleanup_old_documents(db)
        finally:
            db.close()

# Global cleanup service instance
cleanup_service = CleanupService(cleanup_interval_hours=2.0)

async def get_cleanup_stats() -> dict:
    """
    Get statistics about documents eligible for cleanup.
    
    Returns:
        dict: Statistics about current documents
    """
    db = SessionLocal()
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=cleanup_service.cleanup_interval_hours)
        
        total_documents = db.query(Document).count()
        old_documents = db.query(Document).filter(Document.upload_date < cutoff_time).count()
        recent_documents = total_documents - old_documents
        
        # Get oldest and newest document dates
        oldest_doc = db.query(Document).order_by(Document.upload_date.asc()).first()
        newest_doc = db.query(Document).order_by(Document.upload_date.desc()).first()
        
        return {
            "total_documents": total_documents,
            "documents_pending_cleanup": old_documents,
            "recent_documents": recent_documents,
            "cleanup_interval_hours": cleanup_service.cleanup_interval_hours,
            "oldest_document_date": oldest_doc.upload_date.isoformat() if oldest_doc else None,
            "newest_document_date": newest_doc.upload_date.isoformat() if newest_doc else None,
            "next_cleanup_cutoff": (datetime.utcnow() - timedelta(hours=cleanup_service.cleanup_interval_hours)).isoformat()
        }
    finally:
        db.close()

#!/usr/bin/env python3
"""
Test script for the cleanup service functionality.
Creates test documents and verifies they are cleaned up correctly.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from database import SessionLocal, Document, Entity, Base, engine
from cleanup_service import CleanupService

async def test_cleanup_service():
    """Test the cleanup service functionality."""
    print("🧪 Testing Cleanup Service Functionality")
    print("=" * 50)
    
    # Create test cleanup service with very short interval (1 minute for testing)
    test_cleanup_service = CleanupService(cleanup_interval_hours=1/60)  # 1 minute
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Create a test document with old timestamp
        old_time = datetime.utcnow() - timedelta(minutes=2)  # 2 minutes ago
        
        test_document = Document(
            filename="test_cleanup_document.pdf",
            upload_date=old_time,
            status="processed_text",
            path_to_text="/fake/path/test.txt",
            subject="Test Document for Cleanup"
        )
        
        db.add(test_document)
        db.commit()
        db.refresh(test_document)
        
        print(f"✅ Created test document with ID: {test_document.id}")
        print(f"   Upload date: {test_document.upload_date}")
        print(f"   File: {test_document.filename}")
        
        # Add a test entity for the document
        test_entity = Entity(
            document_id=test_document.id,
            text="Test Entity",
            label="TEST",
            start_char=0,
            end_char=11
        )
        
        db.add(test_entity)
        db.commit()
        
        print(f"✅ Created test entity for document")
        
        # Check documents before cleanup
        document_count_before = db.query(Document).count()
        entity_count_before = db.query(Entity).count()
        
        print(f"\n📊 Before cleanup:")
        print(f"   Documents: {document_count_before}")
        print(f"   Entities: {entity_count_before}")
        
        # Run cleanup
        print(f"\n🧹 Running cleanup (cutoff: {test_cleanup_service.cleanup_interval_hours} hours)...")
        cleanup_stats = await test_cleanup_service.cleanup_old_documents(db)
        
        print(f"✅ Cleanup completed:")
        print(f"   Status: {cleanup_stats['status']}")
        print(f"   Documents deleted: {cleanup_stats['documents_deleted']}")
        print(f"   Files deleted: {cleanup_stats['files_deleted']}")
        print(f"   Entities deleted: {cleanup_stats['entities_deleted']}")
        
        if cleanup_stats['errors']:
            print(f"   Errors: {cleanup_stats['errors']}")
        
        # Check documents after cleanup
        document_count_after = db.query(Document).count()
        entity_count_after = db.query(Entity).count()
        
        print(f"\n📊 After cleanup:")
        print(f"   Documents: {document_count_after}")
        print(f"   Entities: {entity_count_after}")
        
        # Verify cleanup worked
        if document_count_after < document_count_before:
            print("✅ Test PASSED: Documents were cleaned up successfully!")
        else:
            print("❌ Test FAILED: No documents were cleaned up")
            
        if entity_count_after < entity_count_before:
            print("✅ Test PASSED: Entities were cleaned up successfully!")
        else:
            print("❌ Test FAILED: No entities were cleaned up")
        
        # Create a recent document to verify it's NOT deleted
        recent_document = Document(
            filename="recent_document.pdf",
            upload_date=datetime.utcnow(),  # Now
            status="processed_text",
            path_to_text="/fake/path/recent.txt",
            subject="Recent Document"
        )
        
        db.add(recent_document)
        db.commit()
        
        print(f"\n✅ Created recent document (should NOT be deleted)")
        
        # Run cleanup again
        cleanup_stats_2 = await test_cleanup_service.cleanup_old_documents(db)
        
        print(f"\n🧹 Second cleanup run:")
        print(f"   Documents deleted: {cleanup_stats_2['documents_deleted']}")
        
        if cleanup_stats_2['documents_deleted'] == 0:
            print("✅ Test PASSED: Recent documents were preserved!")
        else:
            print("❌ Test FAILED: Recent documents were incorrectly deleted")
        
        # Clean up our test data
        db.query(Document).filter(Document.filename.in_(["recent_document.pdf"])).delete()
        db.commit()
        
        print("\n🧪 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_cleanup_service())

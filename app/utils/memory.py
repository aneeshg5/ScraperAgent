"""
Memory utilities for session continuity and research history
Handles storage and retrieval of research sessions and user interactions
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
from contextlib import asynccontextmanager
import aiosqlite
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agent_memory.db")
DB_PATH = DATABASE_URL.replace("sqlite:///", "")


class MemorySaver:
    """
    Memory management for the multimodal research agent
    Handles session storage, research history, and user preferences
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._initialized = False
    
    async def initialize(self):
        """Initialize the database and create tables"""
        if self._initialized:
            return
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create research sessions table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS research_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE,
                        query TEXT NOT NULL,
                        urls TEXT,  -- JSON array of URLs
                        analysis TEXT,
                        vision_insights TEXT,  -- JSON array
                        sources_analyzed TEXT,  -- JSON array
                        processing_time REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT  -- JSON for additional data
                    )
                """)
                
                # Create user preferences table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        preference_key TEXT,
                        preference_value TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create search history table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS search_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        query_type TEXT,  -- 'research', 'vision', 'general'
                        frequency INTEGER DEFAULT 1,
                        last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                        results_quality REAL  -- User feedback score
                    )
                """)
                
                # Create indexes for better performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON research_sessions(timestamp)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_query ON research_sessions(query)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query)")
                
                await db.commit()
                
            self._initialized = True
            logger.info("✅ Memory database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory database: {str(e)}")
            raise
    
    async def save_research_session(
        self,
        query: str,
        urls: List[str],
        analysis: str,
        vision_insights: Optional[List[str]] = None,
        sources_analyzed: Optional[List[str]] = None,
        processing_time: Optional[float] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a research session to memory
        
        Returns:
            session_id of the saved session
        """
        await self.initialize()
        
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query) % 10000}"
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO research_sessions 
                    (session_id, query, urls, analysis, vision_insights, sources_analyzed, 
                     processing_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    query,
                    json.dumps(urls),
                    analysis,
                    json.dumps(vision_insights) if vision_insights else None,
                    json.dumps(sources_analyzed) if sources_analyzed else None,
                    processing_time,
                    json.dumps(metadata) if metadata else None
                ))
                
                await db.commit()
            
            # Update search history
            await self._update_search_history(query, 'research')
            
            logger.info(f"✅ Research session saved: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save research session: {str(e)}")
            raise
    
    async def get_research_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific research session
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT session_id, query, urls, analysis, vision_insights, 
                           sources_analyzed, processing_time, timestamp, metadata
                    FROM research_sessions 
                    WHERE session_id = ?
                """, (session_id,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        return {
                            'session_id': row[0],
                            'query': row[1],
                            'urls': json.loads(row[2]) if row[2] else [],
                            'analysis': row[3],
                            'vision_insights': json.loads(row[4]) if row[4] else None,
                            'sources_analyzed': json.loads(row[5]) if row[5] else None,
                            'processing_time': row[6],
                            'timestamp': row[7],
                            'metadata': json.loads(row[8]) if row[8] else None
                        }
                    
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Failed to retrieve session {session_id}: {str(e)}")
            return None
    
    async def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent research sessions
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT session_id, query, urls, analysis, vision_insights,
                           sources_analyzed, processing_time, timestamp, metadata
                    FROM research_sessions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    
                    sessions = []
                    for row in rows:
                        sessions.append({
                            'session_id': row[0],
                            'query': row[1],
                            'urls': json.loads(row[2]) if row[2] else [],
                            'analysis': row[3],
                            'vision_insights': json.loads(row[4]) if row[4] else None,
                            'sources_analyzed': json.loads(row[5]) if row[5] else None,
                            'processing_time': row[6],
                            'timestamp': row[7],
                            'metadata': json.loads(row[8]) if row[8] else None
                        })
                    
                    return sessions
                    
        except Exception as e:
            logger.error(f"❌ Failed to retrieve recent sessions: {str(e)}")
            return []
    
    async def search_sessions(
        self, 
        query_pattern: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search research sessions by query pattern
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT session_id, query, urls, analysis, timestamp
                    FROM research_sessions 
                    WHERE query LIKE ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (f"%{query_pattern}%", limit)) as cursor:
                    rows = await cursor.fetchall()
                    
                    sessions = []
                    for row in rows:
                        sessions.append({
                            'session_id': row[0],
                            'query': row[1],
                            'urls': json.loads(row[2]) if row[2] else [],
                            'analysis': row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                            'timestamp': row[4]
                        })
                    
                    return sessions
                    
        except Exception as e:
            logger.error(f"❌ Failed to search sessions: {str(e)}")
            return []
    
    async def _update_search_history(self, query: str, query_type: str):
        """
        Update search history for query suggestions
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Check if query exists
                async with db.execute("""
                    SELECT frequency FROM search_history WHERE query = ?
                """, (query,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        # Update existing
                        await db.execute("""
                            UPDATE search_history 
                            SET frequency = frequency + 1, last_used = CURRENT_TIMESTAMP
                            WHERE query = ?
                        """, (query,))
                    else:
                        # Insert new
                        await db.execute("""
                            INSERT INTO search_history (query, query_type)
                            VALUES (?, ?)
                        """, (query, query_type))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to update search history: {str(e)}")
    
    async def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get popular/frequent queries for suggestions
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT query, query_type, frequency, last_used
                    FROM search_history 
                    ORDER BY frequency DESC, last_used DESC
                    LIMIT ?
                """, (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    
                    return [
                        {
                            'query': row[0],
                            'query_type': row[1],
                            'frequency': row[2],
                            'last_used': row[3]
                        }
                        for row in rows
                    ]
                    
        except Exception as e:
            logger.error(f"❌ Failed to get popular queries: {str(e)}")
            return []
    
    async def save_user_preference(
        self, 
        user_id: str, 
        preference_key: str, 
        preference_value: Any
    ):
        """
        Save user preference
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO user_preferences 
                    (user_id, preference_key, preference_value, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, preference_key, json.dumps(preference_value)))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to save user preference: {str(e)}")
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get all user preferences
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT preference_key, preference_value
                    FROM user_preferences 
                    WHERE user_id = ?
                """, (user_id,)) as cursor:
                    rows = await cursor.fetchall()
                    
                    preferences = {}
                    for row in rows:
                        try:
                            preferences[row[0]] = json.loads(row[1])
                        except json.JSONDecodeError:
                            preferences[row[0]] = row[1]
                    
                    return preferences
                    
        except Exception as e:
            logger.error(f"❌ Failed to get user preferences: {str(e)}")
            return {}
    
    async def clear_old_sessions(self, days_old: int = 30):
        """
        Clear research sessions older than specified days
        """
        await self.initialize()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    DELETE FROM research_sessions 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                await db.commit()
                
                logger.info(f"✅ Cleared {deleted_count} old research sessions")
                return deleted_count
                
        except Exception as e:
            logger.error(f"❌ Failed to clear old sessions: {str(e)}")
            return 0
    
    async def clear_all_sessions(self):
        """
        Clear all research sessions (use with caution)
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM research_sessions")
                await db.execute("DELETE FROM search_history")
                await db.commit()
                
                logger.info("✅ All sessions and search history cleared")
                
        except Exception as e:
            logger.error(f"❌ Failed to clear all sessions: {str(e)}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Count sessions
                async with db.execute("SELECT COUNT(*) FROM research_sessions") as cursor:
                    session_count = (await cursor.fetchone())[0]
                
                # Count search history
                async with db.execute("SELECT COUNT(*) FROM search_history") as cursor:
                    search_count = (await cursor.fetchone())[0]
                
                # Count user preferences
                async with db.execute("SELECT COUNT(*) FROM user_preferences") as cursor:
                    pref_count = (await cursor.fetchone())[0]
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'total_sessions': session_count,
                    'search_history_entries': search_count,
                    'user_preferences': pref_count,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {str(e)}")
            return {}


# Global memory saver instance
memory_saver = MemorySaver()


# Context manager for database connections
@asynccontextmanager
async def get_db_connection():
    """Context manager for database connections"""
    await memory_saver.initialize()
    async with aiosqlite.connect(memory_saver.db_path) as db:
        yield db 
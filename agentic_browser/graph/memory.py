"""
Memory and persistence for LangGraph sessions.

Provides SQLite-based checkpointing for session resume.
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

from ..config import get_base_dir


class SessionStore:
    """SQLite-based session storage for agent state.
    
    Enables:
    - Session persistence across restarts
    - Resume interrupted tasks
    - Session history for debugging
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize session store.
        
        Args:
            db_path: Path to SQLite database (default: ~/.agentic_browser/sessions.db)
        """
        if db_path is None:
            db_path = get_base_dir() / "sessions.db"
        
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed BOOLEAN DEFAULT FALSE,
                status TEXT DEFAULT 'active',
                final_answer TEXT,
                error TEXT,
                state_json TEXT NOT NULL,
                embedding BLOB
            )
        """)
        
        # Migration: Add embedding column if it doesn't exist
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass  # Column likely exists
            
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                agent TEXT NOT NULL,
                action TEXT,
                args_json TEXT,
                result TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # --- STRATEGY BANK TABLES ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding BLOB,
                usage_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_strategies (
                session_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (strategy_id) REFERENCES strategies(id),
                PRIMARY KEY (session_id, strategy_id)
            )
        """)
        
        conn.commit()
    
    # ... (Keep existing methods until search_sessions) ...
    
    def get_embedding_model(self):
        """Lazy load sentence transformer model via public accessor."""
        return self._get_embedding_model()

    def _get_embedding_model(self):
        """Lazy load sentence transformer model."""
        if not hasattr(self, '_embedding_model'):
            try:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("⚠️ sentence-transformers not installed. Semantic search disabled.")
                self._embedding_model = None
            except Exception as e:
                print(f"⚠️ Error loading embedding model: {e}")
                self._embedding_model = None
        return self._embedding_model

    # ... (Keep existing methods) ...

    def save_strategy(self, name: str, description: str, session_id: str) -> str:
        """Save a strategy, deduplicating if a similar one exists.
        
        Args:
            name: Short name of strategy
            description: Detailed description
            session_id: Session that used this strategy
            
        Returns:
            strategy_id: The ID of the strategy (new or existing)
        """
        import uuid
        import numpy as np
        
        now = datetime.utcnow().isoformat()
        embedding_blob = self._compute_embedding(description)
        
        conn = self._get_conn()
        
        # 1. Check for existing similar strategy
        existing_id = None
        
        if embedding_blob:
            # Get all strategies
            rows = conn.execute("SELECT id, embedding FROM strategies WHERE embedding IS NOT NULL").fetchall()
            
            query_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            best_score = -1.0
            best_id = None
            
            for row in rows:
                if row['embedding']:
                    emb = np.frombuffer(row['embedding'], dtype=np.float32)
                    score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                    if score > best_score:
                        best_score = score
                        best_id = row['id']
            
            # Threshold for "same strategy"
            if best_score > 0.85:
                print(f"[STRATEGY] Found existing strategy match: {best_id} (score={best_score:.2f})")
                existing_id = best_id
        
        if existing_id:
            # Update existing
            conn.execute("""
                UPDATE strategies 
                SET usage_count = usage_count + 1, last_used_at = ?
                WHERE id = ?
            """, (now, existing_id))
            strategy_id = existing_id
        else:
            # Insert new
            strategy_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO strategies (id, name, description, embedding, usage_count, created_at, last_used_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)
            """, (strategy_id, name, description, embedding_blob, now, now))
            print(f"[STRATEGY] Created new strategy: {name} ({strategy_id})")
            
        # Link to session
        try:
            conn.execute("""
                INSERT INTO session_strategies (session_id, strategy_id)
                VALUES (?, ?)
            """, (session_id, strategy_id))
        except sqlite3.IntegrityError:
            pass # Already linked
            
        conn.commit()
        return strategy_id

    def search_strategies(self, query: str, limit: int = 3) -> list[dict]:
        """Search for strategies conceptually."""
        conn = self._get_conn()
        
        # 1. Try Semantic Search
        try:
            model = self._get_embedding_model()
            if model:
                import numpy as np
                query_embedding = model.encode(query)
                
                rows = conn.execute("SELECT id, name, description, usage_count, embedding FROM strategies WHERE embedding IS NOT NULL").fetchall()
                
                scored_results = []
                for row in rows:
                    if row['embedding']:
                        emb = np.frombuffer(row['embedding'], dtype=np.float32)
                        score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                        scored_results.append((score, dict(row)))
                
                scored_results.sort(key=lambda x: x[0], reverse=True)
                
                results = []
                for score, row in scored_results[:limit]:
                    del row['embedding']
                    row['score'] = float(score)
                    # Boost score slightly by usage count to prefer proven strategies
                    # Sigmoid-like boost: +0.05 for frequently used
                    row['score'] += min(0.1, row['usage_count'] * 0.01)
                    results.append(row)
                    
                if results and results[0]['score'] > 0.3:
                    return results
        except Exception as e:
            print(f"Strategy semantic search failed: {e}")
            
        # 2. Fallback Keyword
        sql = """
            SELECT id, name, description, usage_count
            FROM strategies
            WHERE (name LIKE ? OR description LIKE ?)
            ORDER BY usage_count DESC
            LIMIT ?
        """
        wildcard = f"%{query}%"
        rows = conn.execute(sql, (wildcard, wildcard, limit)).fetchall()
        return [dict(row) for row in rows]

    # ... (Rest of existing methods) ...

    def _compute_embedding(self, text: str) -> Optional[bytes]:
        """Compute embedding for text."""
        model = self._get_embedding_model()
        if model and text:
            try:
                embedding = model.encode(text)
                return embedding.tobytes()
            except Exception as e:
                print(f"Embedding error: {e}")
        return None

    def create_session(self, session_id: str, goal: str, state: dict) -> None:
        """Create a new session."""
        now = datetime.utcnow().isoformat()
        
        # Compute embedding for goal
        embedding_blob = self._compute_embedding(goal)
        
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO sessions (id, goal, created_at, updated_at, state_json, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, goal, now, now, json.dumps(state), embedding_blob))
        conn.commit()
    
    def update_session(self, session_id: str, state: dict) -> None:
        """Update session state."""
        now = datetime.utcnow().isoformat()
        completed = state.get("task_complete", False)
        final_answer = state.get("final_answer")
        error = state.get("error")
        # Also re-fetch goal if needed? Assuming goal doesn't change much.
        
        status = "active"
        if completed:
            status = "success" if not error else "failure"
        elif error:
            status = "failure"
            
        conn = self._get_conn()
        
        # If we have a final answer, update embedding to include it
        if final_answer:
            # We need the goal first to make a rich embedding
            row = conn.execute("SELECT goal FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row:
                rich_text = f"{row['goal']} \n\nOutcome: {final_answer}"
                embedding_blob = self._compute_embedding(rich_text)
                
                conn.execute("""
                    UPDATE sessions 
                    SET state_json = ?, updated_at = ?, completed = ?, final_answer = ?, error = ?, status = ?, embedding = ?
                    WHERE id = ?
                """, (json.dumps(state), now, completed, final_answer, error, status, embedding_blob, session_id))
            else:
                # Fallback standard update
                 conn.execute("""
                    UPDATE sessions 
                    SET state_json = ?, updated_at = ?, completed = ?, final_answer = ?, error = ?, status = ?
                    WHERE id = ?
                """, (json.dumps(state), now, completed, final_answer, error, status, session_id))
        else:
            conn.execute("""
                UPDATE sessions 
                SET state_json = ?, updated_at = ?, completed = ?, final_answer = ?, error = ?, status = ?
                WHERE id = ?
            """, (json.dumps(state), now, completed, final_answer, error, status, session_id))
            
        conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session state."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        
        if row:
            return {
                "id": row["id"],
                "goal": row["goal"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "completed": bool(row["completed"]),
                "final_answer": row["final_answer"],
                "state": json.loads(row["state_json"]),
                "status": row["status"] if "status" in row.keys() else "unknown"
            }
        return None

    def add_step(
        self, 
        session_id: str, 
        step_number: int,
        agent: str,
        action: Optional[str] = None,
        args: Optional[dict] = None,
        result: Optional[str] = None,
    ) -> None:
        """Log a step in the session."""
        now = datetime.utcnow().isoformat()
        args_json = json.dumps(args) if args else None
        
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO session_steps (session_id, step_number, agent, action, args_json, result, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, step_number, agent, action, args_json, result, now))
        conn.commit()

    def search_sessions(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid semantic + keyword search for sessions."""
        conn = self._get_conn()
        
        # 1. Try Semantic Search first
        try:
            model = self._get_embedding_model()
            if model:
                import numpy as np
                query_embedding = model.encode(query)
                
                # Fetch all sessions with embeddings
                rows = conn.execute("SELECT id, goal, status, created_at, final_answer, embedding FROM sessions WHERE embedding IS NOT NULL").fetchall()
                
                scored_results = []
                for row in rows:
                    if row['embedding']:
                        emb = np.frombuffer(row['embedding'], dtype=np.float32)
                        # Cosine similarity
                        score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                        scored_results.append((score, dict(row)))
                
                # Sort by score descending
                scored_results.sort(key=lambda x: x[0], reverse=True)
                
                # Filter/Clean results
                results = []
                for score, row in scored_results[:limit]:
                    del row['embedding'] # Remove blob
                    row['score'] = float(score)
                    results.append(row)
                
                if results and results[0]['score'] > 0.3: # Threshold
                    return results
        except Exception as e:
            print(f"Semantic search failed, falling back to keyword: {e}")
            
        # 2. Fallback to Keyword Search
        sql = """
            SELECT id, goal, status, created_at, final_answer
            FROM sessions
            WHERE (goal LIKE ? OR final_answer LIKE ?)
            ORDER BY created_at DESC
            LIMIT ?
        """
        wildcard = f"%{query}%"
        rows = conn.execute(sql, (wildcard, wildcard, limit)).fetchall()
        return [dict(row) for row in rows]

    def get_session_steps(self, session_id: str) -> list[dict]:
        """Get all steps for a session."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT step_number, agent, action, args_json, result, timestamp
            FROM session_steps
            WHERE session_id = ?
            ORDER BY step_number ASC
        """, (session_id,)).fetchall()
        
        steps = []
        for row in rows:
            step = dict(row)
            if step["args_json"]:
                try:
                    step["args"] = json.loads(step["args_json"])
                except:
                    step["args"] = {}
            steps.append(step)
        return steps
    
    # ... keep existing methods ...

    
    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        """Get recent sessions.
        
        Args:
            limit: Maximum sessions to return
            
        Returns:
            List of session dicts (without full state)
        """
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT id, goal, created_at, updated_at, completed, final_answer
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session and its steps.
        
        Args:
            session_id: Session identifier
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM session_steps WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

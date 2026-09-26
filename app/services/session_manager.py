import os
import shutil
import time
import uuid
import duckdb
from typing import Dict, Any, Optional, Tuple

UPLOAD_DIR = "/tmp/analytics_sessions"
SESSION_TIMEOUT_SECONDS = 3600  # 1 hour auto-purge

class SessionManager:
    def __init__(self):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        # Store in-memory session metadata: {session_id: {"db_path": str, "last_accessed": float, "schema": dict}}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, file_path: str, filename: str) -> Dict[str, Any]:
        self.cleanup_expired_sessions()
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        db_path = os.path.join(session_dir, "workspace.duckdb")
        conn = duckdb.connect(db_path)

        # Ingestion layer based on file format
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".csv":
            conn.execute(f"CREATE TABLE active_dataset AS SELECT * FROM read_csv_auto('{file_path}')")
        elif ext in [".xlsx", ".xls"]:
            # Uses DuckDB spatial/excel extension or pandas fallback
            conn.execute("INSTALL spatial; LOAD spatial;")
            conn.execute(f"CREATE TABLE active_dataset AS SELECT * FROM st_read('{file_path}')")
        else:
            conn.close()
            raise ValueError(f"Unsupported file extension: {ext}")

        # Extract dataset metadata
        total_rows = conn.execute("SELECT COUNT(*) FROM active_dataset").fetchone()[0]
        schema_info = conn.execute("DESCRIBE active_dataset").fetchall()
        
        columns = [row[0] for row in schema_info]
        schema_types = {row[0]: row[1] for row in schema_info}
        
        preview_df = conn.execute("SELECT * FROM active_dataset LIMIT 5").df()
        preview_data = preview_df.to_dict(orient="records")
        conn.close()

        summary = {
            "session_id": session_id,
            "filename": filename,
            "total_rows": total_rows,
            "total_columns": len(columns),
            "columns": columns,
            "schema_types": schema_types,
            "preview_data": preview_data
        }

        self._sessions[session_id] = {
            "db_path": db_path,
            "last_accessed": time.time(),
            "summary": summary
        }

        return summary

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self._sessions:
            return None
        self._sessions[session_id]["last_accessed"] = time.time()
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            session_dir = os.path.dirname(self._sessions[session_id]["db_path"])
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
            del self._sessions[session_id]
            return True
        return False

    def cleanup_expired_sessions(self):
        now = time.time()
        expired = [
            sid for sid, meta in self._sessions.items()
            if now - meta["last_accessed"] > SESSION_TIMEOUT_SECONDS
        ]
        for sid in expired:
            self.delete_session(sid)

session_manager = SessionManager()
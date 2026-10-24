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

    def create_multi_table_session(self, file_data_list: list[tuple[str, str]]) -> Dict[str, Any]:
        """Ingests multiple files into separate named tables in the DuckDB session database."""
        self.cleanup_expired_sessions()
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        db_path = os.path.join(session_dir, "workspace.duckdb")
        conn = duckdb.connect(db_path)

        tables_metadata = {}
        primary_table = None

        for idx, (file_path, filename) in enumerate(file_data_list):
            table_name = os.path.splitext(filename)[0].lower().replace(" ", "_").replace("-", "_")
            if idx == 0:
                primary_table = table_name

            ext = os.path.splitext(filename)[1].lower()
            if ext == ".csv":
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')")
            elif ext in [".xlsx", ".xls"]:
                conn.execute("INSTALL spatial; LOAD spatial;")
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM st_read('{file_path}')")
            else:
                conn.close()
                raise ValueError(f"Unsupported file extension: {ext}")

            # Alias first table as active_dataset for single-table query compatibility
            if idx == 0 and table_name != "active_dataset":
                conn.execute(f"CREATE TABLE active_dataset AS SELECT * FROM {table_name}")

            total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            schema_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
            columns = [row[0] for row in schema_info]
            schema_types = {row[0]: row[1] for row in schema_info}
            preview_df = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").df()
            
            tables_metadata[table_name] = {
                "total_rows": total_rows,
                "columns": columns,
                "schema_types": schema_types,
                "preview_data": preview_df.to_dict(orient="records")
            }

        conn.close()

        summary = {
            "session_id": session_id,
            "mode": "multi_table",
            "primary_table": primary_table,
            "tables": tables_metadata,
            "filename": ", ".join([f[1] for f in file_data_list]),
            "total_rows": sum(t["total_rows"] for t in tables_metadata.values()),
            "total_columns": sum(len(t["columns"]) for t in tables_metadata.values()),
            "columns": list(tables_metadata[primary_table]["columns"]) if primary_table else [],
            "schema_types": tables_metadata[primary_table]["schema_types"] if primary_table else {},
            "preview_data": tables_metadata[primary_table]["preview_data"] if primary_table else []
        }

        self._sessions[session_id] = {
            "db_path": db_path,
            "last_accessed": time.time(),
            "summary": summary
        }

        return summary

session_manager = SessionManager()
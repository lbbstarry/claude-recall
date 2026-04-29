-- claude-recall storage schema (Week 1: BM25 only; vectors added in Week 2)

CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    mtime       REAL NOT NULL,
    sha256      TEXT NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,         -- sha256(session_id || turn_index)
    session_id    TEXT NOT NULL,
    project       TEXT NOT NULL,
    turn_index    INTEGER NOT NULL,
    start_uuid    TEXT NOT NULL,
    end_uuid      TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    ended_at      TEXT NOT NULL,
    role_mix      TEXT NOT NULL,            -- "user", "assistant", or "user+assistant"
    tool_names    TEXT NOT NULL,            -- comma-joined
    has_tool_use  INTEGER NOT NULL,
    text          TEXT NOT NULL,
    token_count   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project);
CREATE INDEX IF NOT EXISTS idx_chunks_started ON chunks(started_at);

-- FTS5 virtual table mirroring chunks.text. Use external content for size.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

//! Event store (SPEC §4.2) — rusqlite in WAL mode. Holds the `events` table
//! exactly as specified, plus tables for enrollment examples (SPEC §5 Phase
//! B-lite) and key/value settings. A `schema_migrations` table tracks applied
//! migrations from day one. Killing/relaunching the app never loses events.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};

use crate::classify::proto::Enrollment;
use crate::error::Result;
use crate::types::{Event, EventType, Source};

/// Encode an f32 vector as little-endian bytes for BLOB storage.
fn f32_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode a little-endian BLOB back into an f32 vector.
fn blob_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A stored enrollment example with its row id.
#[derive(Debug, Clone)]
pub struct StoredEnrollment {
    pub id: i64,
    pub enrollment: Enrollment,
}

/// The SQLite-backed store.
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open (creating if needed) a store at `path` and run migrations.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// Open an in-memory store (tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let mut store = Store { conn };
        store.migrate()?;
        Ok(store)
    }

    /// Idempotent, ordered migrations tracked in `schema_migrations`.
    fn migrate(&mut self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );",
        )?;

        let migrations: &[(i64, &str)] = &[
            (
                1,
                "CREATE TABLE events (
                    uuid          TEXT PRIMARY KEY,
                    event_type    TEXT NOT NULL,
                    occurred_at   TEXT NOT NULL,
                    tz_offset_min INTEGER NOT NULL,
                    duration_ms   INTEGER NOT NULL,
                    confidence    REAL NOT NULL,
                    burst_count   INTEGER NOT NULL DEFAULT 1,
                    model_version TEXT NOT NULL,
                    source        TEXT NOT NULL,
                    device_id     TEXT NOT NULL,
                    uploaded_at   TEXT NULL,
                    deleted       INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX idx_events_pending ON events (uploaded_at) WHERE uploaded_at IS NULL;
                CREATE INDEX idx_events_day ON events (occurred_at);",
            ),
            (
                2,
                "CREATE TABLE enrollment_examples (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    class       TEXT NOT NULL,
                    embedding   BLOB NOT NULL,
                    is_negative INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT NOT NULL
                );
                CREATE INDEX idx_enroll_class ON enrollment_examples (class);",
            ),
            (
                3,
                "CREATE TABLE settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );",
            ),
        ];

        let current: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
            [],
            |r| r.get(0),
        )?;

        for (version, sql) in migrations {
            if *version > current {
                let tx = self.conn.transaction()?;
                tx.execute_batch(sql)?;
                tx.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?1, ?2)",
                    params![version, Utc::now().to_rfc3339()],
                )?;
                tx.commit()?;
            }
        }
        Ok(())
    }

    /// Current schema version (highest applied migration).
    pub fn schema_version(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
            [],
            |r| r.get(0),
        )?)
    }

    // ----- events -------------------------------------------------------------

    /// Insert an event. Ignores a duplicate uuid (idempotent local write).
    pub fn insert_event(&self, e: &Event) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO events
                (uuid, event_type, occurred_at, tz_offset_min, duration_ms, confidence,
                 burst_count, model_version, source, device_id, uploaded_at, deleted)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
            params![
                e.uuid,
                e.event_type.as_str(),
                e.occurred_at.to_rfc3339(),
                e.tz_offset_min,
                e.duration_ms,
                e.confidence,
                e.burst_count,
                e.model_version,
                e.source.as_str(),
                e.device_id,
                e.uploaded_at.map(|t| t.to_rfc3339()),
                e.deleted as i64,
            ],
        )?;
        Ok(())
    }

    fn row_to_event(row: &rusqlite::Row) -> rusqlite::Result<Event> {
        let occurred: String = row.get("occurred_at")?;
        let uploaded: Option<String> = row.get("uploaded_at")?;
        let etype: String = row.get("event_type")?;
        let src: String = row.get("source")?;
        Ok(Event {
            uuid: row.get("uuid")?,
            event_type: EventType::parse(&etype).unwrap_or(EventType::Cough),
            occurred_at: DateTime::parse_from_rfc3339(&occurred)
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            tz_offset_min: row.get("tz_offset_min")?,
            duration_ms: row.get("duration_ms")?,
            confidence: row.get("confidence")?,
            burst_count: row.get("burst_count")?,
            model_version: row.get("model_version")?,
            source: parse_source(&src),
            device_id: row.get("device_id")?,
            uploaded_at: uploaded
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|d| d.with_timezone(&Utc)),
            deleted: row.get::<_, i64>("deleted")? != 0,
        })
    }

    /// Fetch one event by uuid.
    pub fn get_event(&self, uuid: &str) -> Result<Option<Event>> {
        Ok(self
            .conn
            .query_row(
                "SELECT * FROM events WHERE uuid = ?1",
                params![uuid],
                Self::row_to_event,
            )
            .optional()?)
    }

    /// Events awaiting upload: not yet uploaded and not deleted, oldest first.
    pub fn pending_events(&self, limit: usize) -> Result<Vec<Event>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM events
             WHERE uploaded_at IS NULL AND deleted = 0
             ORDER BY occurred_at ASC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], Self::row_to_event)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Uuids of already-uploaded events the user has since deleted — these need a
    /// server-side DELETE (tombstone sync, SPEC §4.3).
    pub fn pending_tombstones(&self, limit: usize) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT uuid FROM events
             WHERE deleted = 1 AND uploaded_at IS NOT NULL
             ORDER BY occurred_at ASC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Mark events uploaded at `at`.
    pub fn mark_uploaded(&self, uuids: &[String], at: DateTime<Utc>) -> Result<()> {
        let ts = at.to_rfc3339();
        for uuid in uuids {
            self.conn.execute(
                "UPDATE events SET uploaded_at = ?2 WHERE uuid = ?1",
                params![uuid, ts],
            )?;
        }
        Ok(())
    }

    /// Tombstone an event locally (user removed a false positive).
    pub fn mark_deleted(&self, uuid: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE events SET deleted = 1 WHERE uuid = ?1",
            params![uuid],
        )?;
        Ok(())
    }

    /// Physically remove rows (used after a tombstone DELETE is confirmed, or for
    /// never-uploaded local deletes).
    pub fn purge(&self, uuids: &[String]) -> Result<()> {
        for uuid in uuids {
            self.conn
                .execute("DELETE FROM events WHERE uuid = ?1", params![uuid])?;
        }
        Ok(())
    }

    /// Events with `occurred_at` in `[from, to)`, newest first — history/export.
    pub fn events_in_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<Event>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM events
             WHERE occurred_at >= ?1 AND occurred_at < ?2 AND deleted = 0
             ORDER BY occurred_at DESC",
        )?;
        let rows = stmt.query_map(
            params![from.to_rfc3339(), to.to_rfc3339()],
            Self::row_to_event,
        )?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Count of not-deleted events.
    pub fn event_count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM events WHERE deleted = 0", [], |r| {
                r.get(0)
            })?)
    }

    // ----- settings -----------------------------------------------------------

    pub fn setting_get(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT value FROM settings WHERE key = ?1",
                params![key],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub fn setting_set(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // ----- enrollment ---------------------------------------------------------

    /// Add an enrollment example; returns its row id.
    pub fn add_enrollment(
        &self,
        class: EventType,
        embedding: &[f32],
        is_negative: bool,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO enrollment_examples (class, embedding, is_negative, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                class.as_str(),
                f32_to_blob(embedding),
                is_negative as i64,
                Utc::now().to_rfc3339()
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// All enrollments (for rebuilding the prototype matcher).
    pub fn enrollments(&self) -> Result<Vec<StoredEnrollment>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, class, embedding, is_negative FROM enrollment_examples ORDER BY id",
        )?;
        let rows = stmt.query_map([], |row| {
            let class: String = row.get("class")?;
            let blob: Vec<u8> = row.get("embedding")?;
            Ok(StoredEnrollment {
                id: row.get("id")?,
                enrollment: Enrollment {
                    class: EventType::parse(&class).unwrap_or(EventType::Cough),
                    embedding: blob_to_f32(&blob),
                    is_negative: row.get::<_, i64>("is_negative")? != 0,
                },
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Delete one enrollment example.
    pub fn delete_enrollment(&self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM enrollment_examples WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Count of positive (non-negative) examples per class.
    pub fn enrollment_counts(&self) -> Result<std::collections::HashMap<EventType, i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT class, COUNT(*) FROM enrollment_examples WHERE is_negative = 0 GROUP BY class",
        )?;
        let rows = stmt.query_map([], |row| {
            let class: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((class, count))
        })?;
        let mut out = std::collections::HashMap::new();
        for r in rows {
            let (class, count) = r?;
            if let Some(et) = EventType::parse(&class) {
                out.insert(et, count);
            }
        }
        Ok(out)
    }
}

fn parse_source(s: &str) -> Source {
    match s {
        "desktop-win" => Source::DesktopWin,
        "mobile-ios" => Source::MobileIos,
        "mobile-android" => Source::MobileAndroid,
        _ => Source::DesktopMac,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_event(uuid: &str) -> Event {
        Event {
            uuid: uuid.to_string(),
            event_type: EventType::Sniffle,
            occurred_at: Utc::now(),
            tz_offset_min: -420,
            duration_ms: 900,
            confidence: 0.7,
            burst_count: 2,
            model_version: "band-heuristic@0".to_string(),
            source: Source::DesktopMac,
            device_id: "dev-1".to_string(),
            uploaded_at: None,
            deleted: false,
        }
    }

    #[test]
    fn migrations_apply_and_are_idempotent() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.schema_version().unwrap(), 3);
        // Re-running migrate on the same connection is a no-op.
        let mut store = store;
        store.migrate().unwrap();
        assert_eq!(store.schema_version().unwrap(), 3);
    }

    #[test]
    fn event_roundtrips() {
        let store = Store::open_in_memory().unwrap();
        let e = sample_event("u1");
        store.insert_event(&e).unwrap();
        let got = store.get_event("u1").unwrap().unwrap();
        assert_eq!(got.event_type, EventType::Sniffle);
        assert_eq!(got.burst_count, 2);
        assert_eq!(got.tz_offset_min, -420);
        assert!(got.uploaded_at.is_none());
    }

    #[test]
    fn duplicate_uuid_is_ignored() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("dup")).unwrap();
        store.insert_event(&sample_event("dup")).unwrap();
        assert_eq!(store.event_count().unwrap(), 1);
    }

    #[test]
    fn pending_query_excludes_uploaded_and_deleted() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("a")).unwrap();
        store.insert_event(&sample_event("b")).unwrap();
        store.insert_event(&sample_event("c")).unwrap();
        store.mark_uploaded(&["a".to_string()], Utc::now()).unwrap();
        store.mark_deleted("b").unwrap();
        let pending = store.pending_events(100).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].uuid, "c");
    }

    #[test]
    fn tombstones_are_uploaded_then_deleted() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("x")).unwrap();
        // Never uploaded, then deleted → not a tombstone (just drop locally).
        store.insert_event(&sample_event("y")).unwrap();
        store.mark_deleted("y").unwrap();
        assert!(store.pending_tombstones(100).unwrap().is_empty());
        // Uploaded then deleted → tombstone.
        store.mark_uploaded(&["x".to_string()], Utc::now()).unwrap();
        store.mark_deleted("x").unwrap();
        let tomb = store.pending_tombstones(100).unwrap();
        assert_eq!(tomb, vec!["x".to_string()]);
        store.purge(&tomb).unwrap();
        assert!(store.get_event("x").unwrap().is_none());
    }

    #[test]
    fn settings_roundtrip() {
        let store = Store::open_in_memory().unwrap();
        assert!(store.setting_get("mode").unwrap().is_none());
        store.setting_set("mode", "auto-batch").unwrap();
        store.setting_set("mode", "offline-strict").unwrap();
        assert_eq!(
            store.setting_get("mode").unwrap().unwrap(),
            "offline-strict"
        );
    }

    #[test]
    fn enrollment_crud() {
        let store = Store::open_in_memory().unwrap();
        let id = store
            .add_enrollment(EventType::Hawk, &[0.1, 0.2, 0.3], false)
            .unwrap();
        store
            .add_enrollment(EventType::Hawk, &[0.4, 0.5, 0.6], false)
            .unwrap();
        store
            .add_enrollment(EventType::Hawk, &[0.0, 0.0, 1.0], true)
            .unwrap();
        let all = store.enrollments().unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].enrollment.embedding, vec![0.1, 0.2, 0.3]);
        let counts = store.enrollment_counts().unwrap();
        assert_eq!(counts[&EventType::Hawk], 2); // negatives excluded
        store.delete_enrollment(id).unwrap();
        assert_eq!(store.enrollments().unwrap().len(), 2);
    }

    #[test]
    fn events_survive_reopen_on_disk() {
        let dir = std::env::temp_dir().join(format!("sinus-store-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.db");
        {
            let store = Store::open(&path).unwrap();
            store.insert_event(&sample_event("persist")).unwrap();
        }
        {
            let store = Store::open(&path).unwrap();
            assert!(store.get_event("persist").unwrap().is_some());
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

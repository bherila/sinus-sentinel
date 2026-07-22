//! Event store (SPEC §4.2) — rusqlite in WAL mode. Holds the `events` table
//! exactly as specified, plus tables for enrollment examples (SPEC §5 Phase
//! B-lite) and key/value settings. A `schema_migrations` table tracks applied
//! migrations from day one. Killing/relaunching the app never loses events.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};

use crate::classify::proto::Enrollment;
use crate::error::Result;
use crate::types::{Event, EventType, Source};

/// "Now", truncated to whole seconds.
///
/// The PHR stores timestamps as MySQL `DATETIME` (second precision) and echoes
/// them back truncated. Stamping sub-second precision locally would make a
/// device's own `updated_at` compare strictly newer than the copy the server
/// just confirmed, so it would re-push the identical document on every flush,
/// forever. Truncating here makes the comparison converge.
fn now_stamp() -> String {
    truncate_to_second(Utc::now()).to_rfc3339()
}

/// Full-precision "now", for timestamps that never round-trip through the
/// server.
///
/// Flag timestamps are only ever compared against themselves (`flag_synced_at`
/// vs `flag_updated_at`), and two flag changes can easily land inside the same
/// second — reporting a false positive and immediately undoing it. Truncating
/// those would make the undo look already-synced and strand it.
fn flag_stamp() -> String {
    Utc::now().to_rfc3339()
}

/// Drop sub-second precision so local and server timestamps are comparable.
pub fn truncate_to_second(t: DateTime<Utc>) -> DateTime<Utc> {
    t - chrono::Duration::nanoseconds(t.timestamp_subsec_nanos() as i64)
}

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
    /// Raw 16-byte uuid keying this example against the PHR.
    pub uuid: Vec<u8>,
    pub enrollment: Enrollment,
    pub created_at: String,
    /// Similarity to a previous take of the same class when this example was
    /// recorded. `None` for the first take and pre-quality-metadata rows.
    pub similarity: Option<f32>,
    /// Same-class similarity minus the closest other enrolled class.
    pub separation: Option<f32>,
    /// How loud the take was, dBFS — helps explain a take that matches poorly.
    pub peak_dbfs: Option<f32>,
    pub model_version: Option<String>,
    /// For a negative: the event whose misdetection produced it.
    pub source_event_uuid: Option<String>,
    /// Whether the server has this example.
    pub synced: bool,
}

/// Fields for a new enrollment example.
#[derive(Debug, Clone)]
pub struct EnrollmentInsert<'a> {
    pub class: EventType,
    pub embedding: &'a [f32],
    pub is_negative: bool,
    pub similarity: Option<f32>,
    pub separation: Option<f32>,
    pub peak_dbfs: Option<f32>,
    pub model_version: Option<&'a str>,
    pub source_event_uuid: Option<&'a str>,
    /// See [`Enrollment::negative_scoped`] — false (unscoped) for a plain
    /// false-positive report, true for the negative half of a correction.
    pub negative_scoped: bool,
}

/// An enrollment example pulled down from the PHR.
#[derive(Debug, Clone)]
pub struct RemoteEnrollment {
    pub uuid: Vec<u8>,
    pub class: EventType,
    pub embedding: Vec<f32>,
    pub is_negative: bool,
    pub created_at: String,
    pub similarity: Option<f32>,
    pub separation: Option<f32>,
    pub peak_dbfs: Option<f32>,
    pub model_version: Option<String>,
    pub source_event_uuid: Option<String>,
    pub negative_scoped: bool,
}

/// Settings that follow the user between machines.
///
/// Everything else is deliberately device-local: `server_url`, `patient_id`,
/// `device_id` and `model_path` describe *this* install, and `mode` is a
/// per-machine network policy — a metered laptop and a desktop legitimately
/// differ, and pulling `offline-strict` onto a second machine would silently
/// disable its sync.
pub const SYNCED_SETTING_KEYS: &[&str] = &["sensitivity", "quiet_start", "quiet_end"];

/// The synced settings, plus the newest edit time among them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SettingsDoc {
    pub values: std::collections::BTreeMap<String, String>,
    /// `None` when nothing has ever been written locally.
    pub updated_at: Option<DateTime<Utc>>,
}

impl SettingsDoc {
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// A false-positive / correction flag whose current state has not reached the
/// server yet.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingFlag {
    pub uuid: String,
    pub false_positive: bool,
    pub corrected_to: Option<EventType>,
    /// The `flag_updated_at` value observed when this was queued; used to detect
    /// a concurrent re-flag when the result comes back.
    pub updated_at: String,
}

/// After this many server rejections an event stops being re-sent and is left for
/// the history UI to surface (SPEC §4.3). Rejections are permanent verdicts, not
/// transient failures, so re-sending forever is pointless.
pub const MAX_REJECTIONS: i64 = 3;

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
        // Three threads (capture, sync, UI) each hold their own Connection to
        // this file. WAL allows a single writer at a time, and the default
        // busy_timeout of 0 turns any collision into an immediate SQLITE_BUSY —
        // which, at call sites that discard write errors, silently drops an
        // event or a setting. Wait instead.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
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
            (
                4,
                "ALTER TABLE events ADD COLUMN reject_count INTEGER NOT NULL DEFAULT 0;
                 ALTER TABLE events ADD COLUMN rejected_at TEXT NULL;",
            ),
            (
                5,
                "ALTER TABLE enrollment_examples ADD COLUMN similarity REAL NULL;
                 ALTER TABLE enrollment_examples ADD COLUMN separation REAL NULL;",
            ),
            (
                // Local-only backbone embeddings of detected events, so a false
                // positive can be enrolled as a negative example after the fact.
                // Never uploaded; pruned once the event ages out of the UI.
                6,
                "CREATE TABLE event_embeddings (
                    uuid      TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                );",
            ),
            (
                // Loudness, plus false-positive / recharacterisation state.
                //
                // `flag_updated_at` is what makes Undo syncable. A queue keyed
                // on `false_positive_at IS NOT NULL` cannot represent a
                // *cleared* flag — clearing nulls that very column, so the row
                // would drop silently out of the queue and leave the PHR marked
                // forever while the local row reverted. Keying on
                // `flag_synced_at IS NULL` alone is worse: every never-flagged
                // row matches. So every flag mutation stamps `flag_updated_at`,
                // and pending means "changed since it was last synced".
                7,
                "ALTER TABLE events ADD COLUMN peak_dbfs REAL NULL;
                 ALTER TABLE events ADD COLUMN mean_dbfs REAL NULL;
                 ALTER TABLE events ADD COLUMN noise_floor_dbfs REAL NULL;
                 ALTER TABLE events ADD COLUMN false_positive_at TEXT NULL;
                 ALTER TABLE events ADD COLUMN corrected_to TEXT NULL;
                 ALTER TABLE events ADD COLUMN corrected_at TEXT NULL;
                 ALTER TABLE events ADD COLUMN flag_updated_at TEXT NULL;
                 ALTER TABLE events ADD COLUMN flag_synced_at TEXT NULL;
                 CREATE INDEX idx_events_flag_pending ON events (flag_updated_at)
                     WHERE flag_updated_at IS NOT NULL;",
            ),
            (
                // Enrollments become syncable: a stable id, provenance, and a
                // soft-delete so a removal can be propagated rather than just
                // vanishing locally.
                //
                // `uuid` is a raw 16-byte BLOB, matching the PHR's BINARY(16)
                // exactly — the same bytes on both sides, base64 only on the
                // wire. `randomblob` backfills existing rows without needing a
                // Rust-side migration hook.
                8,
                "ALTER TABLE enrollment_examples ADD COLUMN uuid BLOB NULL;
                 ALTER TABLE enrollment_examples ADD COLUMN model_version TEXT NULL;
                 ALTER TABLE enrollment_examples ADD COLUMN source_event_uuid TEXT NULL;
                 ALTER TABLE enrollment_examples ADD COLUMN peak_dbfs REAL NULL;
                 ALTER TABLE enrollment_examples ADD COLUMN synced_at TEXT NULL;
                 ALTER TABLE enrollment_examples ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0;
                 UPDATE enrollment_examples SET uuid = randomblob(16) WHERE uuid IS NULL;
                 CREATE UNIQUE INDEX idx_enroll_uuid ON enrollment_examples (uuid);",
            ),
            (
                // Settings get a clock so they can be reconciled last-write-wins
                // against the PHR. `synced_at` records the server timestamp we
                // last agreed on, so adopting a server value does not look like
                // a fresh local edit and bounce back up.
                9,
                "ALTER TABLE settings ADD COLUMN updated_at TEXT NULL;
                 ALTER TABLE settings ADD COLUMN synced_at TEXT NULL;",
            ),
            (
                // How far a negative reaches. Default 0 = unscoped, which is
                // both the old behaviour and the right one for a plain
                // false-positive report: a borderline sound that YAMNet wobbles
                // between cough/sneeze/throat-clearing must not simply re-fire
                // under a sibling label. Only the negative half of a
                // *correction* is scoped, because a positive for the corrected
                // class carries the same embedding.
                10,
                "ALTER TABLE enrollment_examples
                     ADD COLUMN negative_scoped INTEGER NOT NULL DEFAULT 0;",
            ),
        ];

        // The three worker threads each open their own Store at startup, so on
        // the first launch after an upgrade they race here. Read the applied
        // version *inside* an IMMEDIATE transaction — a DEFERRED one (rusqlite's
        // default) takes its write lock too late, letting two threads both see
        // the old version and the loser fail with "duplicate column name",
        // killing capture or sync for the whole session.
        for (version, sql) in migrations {
            let tx = self
                .conn
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

            let current: i64 = tx.query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
                [],
                |r| r.get(0),
            )?;
            if *version <= current {
                continue;
            }

            tx.execute_batch(sql)?;
            tx.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?1, ?2)",
                params![version, Utc::now().to_rfc3339()],
            )?;
            tx.commit()?;
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
                 burst_count, model_version, source, device_id, uploaded_at, deleted,
                 peak_dbfs, mean_dbfs, noise_floor_dbfs)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
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
                e.peak_dbfs,
                e.mean_dbfs,
                e.noise_floor_dbfs,
            ],
        )?;
        Ok(())
    }

    /// Decode a row into an [`Event`], or `Ok(None)` if it is unparseable. A health
    /// diary must never *misreport* a corrupt row (unknown `event_type`, malformed
    /// `occurred_at`), so instead of silently coercing to a default we skip the row
    /// and log it (SPEC §4 — fail loud, not silently wrong). Genuine column-type
    /// errors still propagate as `Err`.
    fn row_to_event(row: &rusqlite::Row) -> rusqlite::Result<Option<Event>> {
        let uuid: String = row.get("uuid")?;
        let occurred: String = row.get("occurred_at")?;
        let uploaded: Option<String> = row.get("uploaded_at")?;
        let etype: String = row.get("event_type")?;
        let src: String = row.get("source")?;
        let rejected: Option<String> = row.get("rejected_at")?;
        let false_positive: Option<String> = row.get("false_positive_at")?;
        let corrected_to: Option<String> = row.get("corrected_to")?;
        let corrected: Option<String> = row.get("corrected_at")?;

        let Some(event_type) = EventType::parse(&etype) else {
            eprintln!("store: skipping event {uuid}: unknown event_type {etype:?}");
            return Ok(None);
        };
        let Ok(occurred_at) = DateTime::parse_from_rfc3339(&occurred) else {
            eprintln!("store: skipping event {uuid}: unparseable occurred_at {occurred:?}");
            return Ok(None);
        };

        Ok(Some(Event {
            uuid,
            event_type,
            occurred_at: occurred_at.with_timezone(&Utc),
            tz_offset_min: row.get("tz_offset_min")?,
            duration_ms: row.get("duration_ms")?,
            confidence: row.get("confidence")?,
            burst_count: row.get("burst_count")?,
            peak_dbfs: row.get("peak_dbfs")?,
            mean_dbfs: row.get("mean_dbfs")?,
            noise_floor_dbfs: row.get("noise_floor_dbfs")?,
            model_version: row.get("model_version")?,
            source: parse_source(&src),
            device_id: row.get("device_id")?,
            uploaded_at: uploaded
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|d| d.with_timezone(&Utc)),
            deleted: row.get::<_, i64>("deleted")? != 0,
            false_positive_at: false_positive
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|d| d.with_timezone(&Utc)),
            corrected_to: corrected_to.as_deref().and_then(EventType::parse),
            corrected_at: corrected
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|d| d.with_timezone(&Utc)),
            reject_count: row.get("reject_count")?,
            rejected_at: rejected
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|d| d.with_timezone(&Utc)),
        }))
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
            .optional()?
            .flatten())
    }

    /// Events awaiting upload: not yet uploaded, not deleted, and not yet rejected
    /// past [`MAX_REJECTIONS`], oldest first. Corrupt rows are skipped (logged).
    pub fn pending_events(&self, limit: usize) -> Result<Vec<Event>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM events
             WHERE uploaded_at IS NULL AND deleted = 0 AND reject_count < ?2
             ORDER BY occurred_at ASC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64, MAX_REJECTIONS], Self::row_to_event)?;
        Ok(rows
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect())
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

    /// Record a server rejection: bumps `reject_count` and stamps `rejected_at`.
    /// Once `reject_count` reaches [`MAX_REJECTIONS`] the event drops out of
    /// [`Store::pending_events`] and is left for the history UI (SPEC §4.3).
    pub fn mark_rejected(&self, uuids: &[String], at: DateTime<Utc>) -> Result<()> {
        let ts = at.to_rfc3339();
        for uuid in uuids {
            self.conn.execute(
                "UPDATE events SET reject_count = reject_count + 1, rejected_at = ?2
                 WHERE uuid = ?1",
                params![uuid, ts],
            )?;
        }
        Ok(())
    }

    // ----- false positives and corrections ------------------------------------

    /// Report an event as a misdetection. The row is kept — a health record
    /// should retain the fact that the classifier got it wrong — but it stops
    /// counting everywhere, and the flag is queued for the PHR.
    pub fn mark_false_positive(&self, uuid: &str) -> Result<()> {
        let now = flag_stamp();
        self.conn.execute(
            "UPDATE events
             SET false_positive_at = ?2, corrected_to = NULL, corrected_at = NULL,
                 flag_updated_at = ?2
             WHERE uuid = ?1",
            params![uuid, now],
        )?;
        Ok(())
    }

    /// Record what a misdetected sound actually was. Unlike a false positive the
    /// event still happened, so it keeps counting — under `class`.
    pub fn recharacterize(&self, uuid: &str, class: EventType) -> Result<()> {
        let now = flag_stamp();
        self.conn.execute(
            "UPDATE events
             SET false_positive_at = NULL, corrected_to = ?3, corrected_at = ?2,
                 flag_updated_at = ?2
             WHERE uuid = ?1",
            params![uuid, now, class.as_str()],
        )?;
        Ok(())
    }

    /// Undo a false-positive report or a correction. Stamping `flag_updated_at`
    /// is what keeps the cleared state in the sync queue — without it the row
    /// would drop out and the PHR would stay flagged forever.
    pub fn clear_flag(&self, uuid: &str) -> Result<()> {
        let now = flag_stamp();
        self.conn.execute(
            "UPDATE events
             SET false_positive_at = NULL, corrected_to = NULL, corrected_at = NULL,
                 flag_updated_at = ?2
             WHERE uuid = ?1",
            params![uuid, now],
        )?;
        Ok(())
    }

    /// Flags whose current state has not reached the server yet.
    pub fn pending_flags(&self, limit: usize) -> Result<Vec<PendingFlag>> {
        let mut stmt = self.conn.prepare(
            "SELECT uuid, false_positive_at, corrected_to, flag_updated_at FROM events
             WHERE flag_updated_at IS NOT NULL
               AND (flag_synced_at IS NULL OR flag_synced_at < flag_updated_at)
             ORDER BY flag_updated_at ASC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            let corrected: Option<String> = row.get("corrected_to")?;
            let false_positive: Option<String> = row.get("false_positive_at")?;
            Ok(PendingFlag {
                uuid: row.get("uuid")?,
                false_positive: false_positive.is_some(),
                corrected_to: corrected.as_deref().and_then(EventType::parse),
                updated_at: row.get("flag_updated_at")?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Record that the server now agrees with the flag state we sent.
    ///
    /// Stamps the `flag_updated_at` value read when the batch was built, not
    /// `now` and not the column's current value: if the user changed the flag
    /// again while the request was in flight, the row's `flag_updated_at` has
    /// moved past what we sent and it must stay pending. `WHERE flag_updated_at
    /// = ?2` also skips the row entirely in that case.
    pub fn mark_flags_synced(&self, flags: &[PendingFlag]) -> Result<()> {
        for flag in flags {
            self.conn.execute(
                "UPDATE events SET flag_synced_at = ?2
                 WHERE uuid = ?1 AND flag_updated_at = ?2",
                params![flag.uuid, flag.updated_at],
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

    /// Events with `occurred_at` in `[from, to)`, newest first, **excluding
    /// reported misdetections** — this is what counts, charts, and the
    /// congestion score are built from. Use [`Store::recent_events`] for the
    /// history list, which must still show flagged rows so they can be undone.
    pub fn events_in_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<Event>> {
        self.events_in_range_inner(from, to, false)
    }

    /// Events in `[from, to)`, newest first, **including** reported
    /// misdetections so the UI can render them struck through with an undo.
    pub fn recent_events(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<Vec<Event>> {
        self.events_in_range_inner(from, to, true)
    }

    fn events_in_range_inner(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        include_false_positives: bool,
    ) -> Result<Vec<Event>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM events
             WHERE occurred_at >= ?1 AND occurred_at < ?2 AND deleted = 0
               AND (?3 OR false_positive_at IS NULL)
             ORDER BY occurred_at DESC",
        )?;
        let rows = stmt.query_map(
            params![from.to_rfc3339(), to.to_rfc3339(), include_false_positives],
            Self::row_to_event,
        )?;
        Ok(rows
            .collect::<rusqlite::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect())
    }

    /// Everything waiting to reach the server, not just events.
    ///
    /// The flush scheduler gates on this: a flag, a teach take, an enrollment
    /// deletion or a settings edit made while offline would otherwise never
    /// retry, because a count of events alone is zero and the scheduler would
    /// never wake the engine again.
    pub fn pending_work_count(&self) -> Result<i64> {
        let events = self.pending_count()?;

        let tombstones: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM events WHERE deleted = 1 AND uploaded_at IS NOT NULL",
            [],
            |r| r.get(0),
        )?;

        let flags: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM events
             WHERE flag_updated_at IS NOT NULL
               AND (flag_synced_at IS NULL OR flag_synced_at < flag_updated_at)",
            [],
            |r| r.get(0),
        )?;

        let enrollments: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM enrollment_examples
             WHERE (deleted = 0 AND synced_at IS NULL)
                OR (deleted = 1 AND synced_at IS NOT NULL)",
            [],
            |r| r.get(0),
        )?;

        let settings: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM settings
             WHERE updated_at IS NOT NULL
               AND (synced_at IS NULL OR synced_at < updated_at)",
            [],
            |r| r.get(0),
        )?;

        Ok(events + tombstones + flags + enrollments + settings)
    }

    /// Count of events that count: not deleted, not a reported misdetection.
    pub fn event_count(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM events WHERE deleted = 0 AND false_positive_at IS NULL",
            [],
            |r| r.get(0),
        )?)
    }

    /// Count of events awaiting upload (same predicate as [`Store::pending_events`]).
    /// Cheap for the tray's pending badge (SPEC §6) without materializing rows.
    pub fn pending_count(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM events
             WHERE uploaded_at IS NULL AND deleted = 0 AND reject_count < ?1",
            params![MAX_REJECTIONS],
            |r| r.get(0),
        )?)
    }

    // ----- event embeddings ---------------------------------------------------

    /// Retain the backbone embedding that produced an event (local-only; enables
    /// enrolling a reported false positive as a negative example). Skips empty
    /// embeddings — the heuristic backbone can produce none worth keeping.
    pub fn put_event_embedding(&self, uuid: &str, embedding: &[f32]) -> Result<()> {
        if embedding.is_empty() {
            return Ok(());
        }
        self.conn.execute(
            "INSERT OR REPLACE INTO event_embeddings (uuid, embedding) VALUES (?1, ?2)",
            params![uuid, f32_to_blob(embedding)],
        )?;
        Ok(())
    }

    /// The stored embedding for an event, if one was retained.
    pub fn get_event_embedding(&self, uuid: &str) -> Result<Option<Vec<f32>>> {
        Ok(self
            .conn
            .query_row(
                "SELECT embedding FROM event_embeddings WHERE uuid = ?1",
                params![uuid],
                |r| r.get::<_, Vec<u8>>(0),
            )
            .optional()?
            .map(|blob| blob_to_f32(&blob)))
    }

    /// Drop one event's stored embedding (after it is consumed by a report).
    pub fn delete_event_embedding(&self, uuid: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM event_embeddings WHERE uuid = ?1",
            params![uuid],
        )?;
        Ok(())
    }

    /// Drop embeddings whose event is gone or occurred before `cutoff` — the
    /// event is no longer reportable in the UI, so its embedding is dead weight.
    pub fn prune_event_embeddings(&self, cutoff: DateTime<Utc>) -> Result<usize> {
        Ok(self.conn.execute(
            "DELETE FROM event_embeddings WHERE uuid NOT IN (
                SELECT uuid FROM events WHERE occurred_at >= ?1 AND deleted = 0
             )",
            params![cutoff.to_rfc3339()],
        )?)
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

    /// Write a setting as a *local* edit: stamps `updated_at = now`, which is
    /// what makes it win the next last-write-wins reconciliation with the PHR.
    pub fn setting_set(&self, key: &str, value: &str) -> Result<()> {
        let now = now_stamp();
        self.conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                            updated_at = excluded.updated_at",
            params![key, value, now],
        )?;
        Ok(())
    }

    /// Adopt a value that came *from* the server.
    ///
    /// Records the server's timestamp as `updated_at`, not `now`. Stamping local
    /// time here would make the value we just accepted look like a fresh local
    /// edit, so the loser of a race would immediately re-push its adopted copy
    /// with a newer timestamp and the two machines would ping-pong forever.
    pub fn setting_set_synced(&self, key: &str, value: &str, server_time: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings (key, value, updated_at, synced_at) VALUES (?1, ?2, ?3, ?3)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                            updated_at = excluded.updated_at,
                                            synced_at = excluded.synced_at",
            params![key, value, server_time],
        )?;
        Ok(())
    }

    /// When a setting was last written, for last-write-wins comparison.
    pub fn setting_updated_at(&self, key: &str) -> Result<Option<DateTime<Utc>>> {
        let raw: Option<Option<String>> = self
            .conn
            .query_row(
                "SELECT updated_at FROM settings WHERE key = ?1",
                params![key],
                |r| r.get(0),
            )
            .optional()?;
        Ok(raw
            .flatten()
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|d| d.with_timezone(&Utc)))
    }

    /// The settings that sync to the PHR, with the newest local edit time among
    /// them. Deliberately excludes device-local keys — `server_url`,
    /// `patient_id`, `device_id`, `model_path` — and `mode`, which is per-machine
    /// network policy: pulling `offline-strict` onto a second machine would
    /// silently disable its sync.
    pub fn settings_doc(&self) -> Result<SettingsDoc> {
        let mut values = std::collections::BTreeMap::new();
        let mut updated_at: Option<DateTime<Utc>> = None;

        for key in SYNCED_SETTING_KEYS {
            if let Some(value) = self.setting_get(key)? {
                values.insert(key.to_string(), value);
            }
            if let Some(stamp) = self.setting_updated_at(key)? {
                updated_at = Some(updated_at.map_or(stamp, |current| current.max(stamp)));
            }
        }

        Ok(SettingsDoc { values, updated_at })
    }

    /// Adopt a server settings document, per key, skipping any key edited more
    /// recently here.
    ///
    /// The per-key check matters because the UI thread writes settings on its
    /// own connection: between the sync thread's GET and this apply, the user
    /// can move the sensitivity slider, and an unconditional write would clobber
    /// an edit made a second ago. Returns the keys actually changed.
    pub fn apply_settings_doc(&mut self, doc: &SettingsDoc) -> Result<Vec<String>> {
        let Some(server_time) = doc.updated_at else {
            return Ok(Vec::new());
        };
        let server_stamp = server_time.to_rfc3339();
        let mut changed = Vec::new();

        let tx = self.conn.transaction()?;
        for (key, value) in &doc.values {
            if !SYNCED_SETTING_KEYS.contains(&key.as_str()) {
                continue;
            }

            let local: Option<(String, Option<String>)> = tx
                .query_row(
                    "SELECT value, updated_at FROM settings WHERE key = ?1",
                    params![key],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;

            if let Some((local_value, local_stamp)) = &local {
                let local_time = local_stamp
                    .as_deref()
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|d| d.with_timezone(&Utc));
                if local_time.is_some_and(|t| t > server_time) {
                    continue;
                }
                if local_value == value {
                    continue;
                }
            }

            tx.execute(
                "INSERT INTO settings (key, value, updated_at, synced_at) VALUES (?1, ?2, ?3, ?3)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                updated_at = excluded.updated_at,
                                                synced_at = excluded.synced_at",
                params![key, value, server_stamp],
            )?;
            changed.push(key.clone());
        }
        tx.commit()?;

        Ok(changed)
    }

    // ----- enrollment ---------------------------------------------------------

    /// Add an enrollment example; returns its row id.
    pub fn add_enrollment(
        &self,
        class: EventType,
        embedding: &[f32],
        is_negative: bool,
    ) -> Result<i64> {
        self.add_enrollment_with_quality(class, embedding, is_negative, None, None)
    }

    /// Add an enrollment and retain the quality observed against earlier takes.
    pub fn add_enrollment_with_quality(
        &self,
        class: EventType,
        embedding: &[f32],
        is_negative: bool,
        similarity: Option<f32>,
        separation: Option<f32>,
    ) -> Result<i64> {
        self.add_enrollment_full(EnrollmentInsert {
            class,
            embedding,
            is_negative,
            similarity,
            separation,
            peak_dbfs: None,
            model_version: None,
            source_event_uuid: None,
            negative_scoped: false,
        })
    }

    /// Add an enrollment with full provenance. Generates the uuid that keys it
    /// against the PHR.
    pub fn add_enrollment_full(&self, insert: EnrollmentInsert<'_>) -> Result<i64> {
        let uuid = uuid::Uuid::new_v4();
        self.conn.execute(
            "INSERT INTO enrollment_examples
                (uuid, class, embedding, is_negative, created_at, similarity, separation,
                 peak_dbfs, model_version, source_event_uuid, negative_scoped)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                uuid.as_bytes().as_slice(),
                insert.class.as_str(),
                f32_to_blob(insert.embedding),
                insert.is_negative as i64,
                now_stamp(),
                insert.similarity,
                insert.separation,
                insert.peak_dbfs,
                insert.model_version,
                insert.source_event_uuid,
                insert.negative_scoped as i64,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// All live enrollments (for rebuilding the prototype matcher).
    pub fn enrollments(&self) -> Result<Vec<StoredEnrollment>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, uuid, class, embedding, is_negative, created_at, similarity, separation,
                    peak_dbfs, model_version, source_event_uuid, synced_at, negative_scoped
             FROM enrollment_examples WHERE deleted = 0 ORDER BY id",
        )?;
        let rows = stmt.query_map([], Self::row_to_enrollment)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn row_to_enrollment(row: &rusqlite::Row) -> rusqlite::Result<StoredEnrollment> {
        let class: String = row.get("class")?;
        let blob: Vec<u8> = row.get("embedding")?;
        let synced_at: Option<String> = row.get("synced_at")?;
        Ok(StoredEnrollment {
            id: row.get("id")?,
            uuid: row.get("uuid")?,
            enrollment: Enrollment {
                class: EventType::parse(&class).unwrap_or(EventType::Cough),
                embedding: blob_to_f32(&blob),
                is_negative: row.get::<_, i64>("is_negative")? != 0,
                negative_scoped: row.get::<_, i64>("negative_scoped")? != 0,
            },
            created_at: row.get("created_at")?,
            similarity: row.get("similarity")?,
            separation: row.get("separation")?,
            peak_dbfs: row.get("peak_dbfs")?,
            model_version: row.get("model_version")?,
            source_event_uuid: row.get("source_event_uuid")?,
            synced: synced_at.is_some(),
        })
    }

    /// Soft-delete one enrollment example. Soft, not hard, so the removal can be
    /// propagated to the PHR instead of just vanishing from this machine.
    pub fn delete_enrollment(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE enrollment_examples SET deleted = 1 WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Soft-delete all positive and negative examples for one class.
    pub fn delete_enrollments_for_class(&self, class: EventType) -> Result<usize> {
        Ok(self.conn.execute(
            "UPDATE enrollment_examples SET deleted = 1 WHERE class = ?1 AND deleted = 0",
            params![class.as_str()],
        )?)
    }

    /// Soft-delete every local enrollment example. Event history is untouched.
    pub fn delete_all_enrollments(&self) -> Result<usize> {
        Ok(self.conn.execute(
            "UPDATE enrollment_examples SET deleted = 1 WHERE deleted = 0",
            [],
        )?)
    }

    /// Count of negative examples (learned false-positive suppressions).
    pub fn negative_enrollment_count(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM enrollment_examples WHERE is_negative = 1 AND deleted = 0",
            [],
            |r| r.get(0),
        )?)
    }

    /// Forget every learned false-positive suppression, keeping taught takes.
    pub fn delete_negative_enrollments(&self) -> Result<usize> {
        Ok(self.conn.execute(
            "UPDATE enrollment_examples SET deleted = 1 WHERE is_negative = 1 AND deleted = 0",
            [],
        )?)
    }

    // ----- enrollment sync ----------------------------------------------------

    /// Live enrollments the server has not seen yet.
    pub fn pending_enrollments(&self, limit: usize) -> Result<Vec<StoredEnrollment>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, uuid, class, embedding, is_negative, created_at, similarity, separation,
                    peak_dbfs, model_version, source_event_uuid, synced_at, negative_scoped
             FROM enrollment_examples
             WHERE deleted = 0 AND synced_at IS NULL
             ORDER BY id LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], Self::row_to_enrollment)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Uuids of enrollments deleted here that the server still has.
    pub fn pending_enrollment_deletions(&self, limit: usize) -> Result<Vec<Vec<u8>>> {
        let mut stmt = self.conn.prepare(
            "SELECT uuid FROM enrollment_examples
             WHERE deleted = 1 AND synced_at IS NOT NULL
             ORDER BY id LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| r.get::<_, Vec<u8>>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn mark_enrollments_synced(&self, uuids: &[Vec<u8>]) -> Result<()> {
        let now = now_stamp();
        for uuid in uuids {
            self.conn.execute(
                "UPDATE enrollment_examples SET synced_at = ?2 WHERE uuid = ?1",
                params![uuid, now],
            )?;
        }
        Ok(())
    }

    /// Drop rows whose deletion the server has confirmed — the tombstone has
    /// done its job.
    pub fn purge_enrollments(&self, uuids: &[Vec<u8>]) -> Result<()> {
        for uuid in uuids {
            self.conn.execute(
                "DELETE FROM enrollment_examples WHERE uuid = ?1 AND deleted = 1",
                params![uuid],
            )?;
        }
        Ok(())
    }

    /// Insert an enrollment pulled from the server, ignoring one already known
    /// (including one deleted here — a local delete must not be resurrected by
    /// the next pull). Returns whether a row was actually added.
    pub fn upsert_remote_enrollment(&self, remote: &RemoteEnrollment) -> Result<bool> {
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO enrollment_examples
                (uuid, class, embedding, is_negative, created_at, similarity, separation,
                 peak_dbfs, model_version, source_event_uuid, synced_at, negative_scoped)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                remote.uuid,
                remote.class.as_str(),
                f32_to_blob(&remote.embedding),
                remote.is_negative as i64,
                remote.created_at,
                remote.similarity,
                remote.separation,
                remote.peak_dbfs,
                remote.model_version,
                remote.source_event_uuid,
                now_stamp(),
                remote.negative_scoped as i64,
            ],
        )?;
        Ok(changed > 0)
    }

    /// Count of positive (non-negative) examples per class.
    pub fn enrollment_counts(&self) -> Result<std::collections::HashMap<EventType, i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT class, COUNT(*) FROM enrollment_examples
             WHERE is_negative = 0 AND deleted = 0 GROUP BY class",
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
            peak_dbfs: Some(-12.0),
            mean_dbfs: Some(-24.0),
            noise_floor_dbfs: Some(-55.0),
            model_version: "band-heuristic@0".to_string(),
            source: Source::DesktopMac,
            device_id: "dev-1".to_string(),
            uploaded_at: None,
            deleted: false,
            false_positive_at: None,
            corrected_to: None,
            corrected_at: None,
            reject_count: 0,
            rejected_at: None,
        }
    }

    #[test]
    fn migrations_apply_and_are_idempotent() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.schema_version().unwrap(), 10);
        // Re-running migrate on the same connection is a no-op.
        let mut store = store;
        store.migrate().unwrap();
        assert_eq!(store.schema_version().unwrap(), 10);
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
    fn rejected_events_drop_out_of_pending_after_max_rejections() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("r")).unwrap();
        assert_eq!(store.pending_events(10).unwrap().len(), 1);

        // Two rejections: still pending (below the cap).
        for _ in 0..(MAX_REJECTIONS - 1) {
            store.mark_rejected(&["r".to_string()], Utc::now()).unwrap();
        }
        assert_eq!(store.pending_events(10).unwrap().len(), 1);

        // The MAX_REJECTIONS-th rejection removes it from the pending set, but the
        // row is retained (with its count/timestamp) for the history UI.
        store.mark_rejected(&["r".to_string()], Utc::now()).unwrap();
        assert!(store.pending_events(10).unwrap().is_empty());
        let got = store.get_event("r").unwrap().unwrap();
        assert_eq!(got.reject_count, MAX_REJECTIONS);
        assert!(got.rejected_at.is_some());
    }

    #[test]
    fn corrupt_rows_are_skipped_not_coerced() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("good")).unwrap();
        // Hand-insert two corrupt rows: an unknown event_type and a malformed
        // occurred_at. Neither should surface as a (mis-typed) event.
        store
            .conn
            .execute(
                "INSERT INTO events (uuid, event_type, occurred_at, tz_offset_min, duration_ms,
                    confidence, burst_count, model_version, source, device_id, deleted)
                 VALUES ('bad_type','not_a_class','2026-01-01T00:00:00Z',0,1,0.5,1,'m','desktop-mac','d',0)",
                [],
            )
            .unwrap();
        store
            .conn
            .execute(
                "INSERT INTO events (uuid, event_type, occurred_at, tz_offset_min, duration_ms,
                    confidence, burst_count, model_version, source, device_id, deleted)
                 VALUES ('bad_date','cough','not-a-date',0,1,0.5,1,'m','desktop-mac','d',0)",
                [],
            )
            .unwrap();

        // Only the good row comes back; the corrupt ones are skipped (and logged).
        let pending = store.pending_events(10).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].uuid, "good");
        assert!(store.get_event("bad_type").unwrap().is_none());
        assert!(store.get_event("bad_date").unwrap().is_none());
    }

    /// A reported misdetection is retained but stops counting; a correction
    /// keeps counting, under the new class. Both stay visible in the history
    /// list so the user can undo them.
    #[test]
    fn false_positives_stop_counting_but_stay_visible() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("fp")).unwrap();
        store.insert_event(&sample_event("fix")).unwrap();
        store.insert_event(&sample_event("keep")).unwrap();

        store.mark_false_positive("fp").unwrap();
        store.recharacterize("fix", EventType::NoseBlow).unwrap();

        let from = Utc::now() - chrono::Duration::days(1);
        let to = Utc::now() + chrono::Duration::days(1);

        // Counts exclude the misdetection but keep the correction.
        let counted = store.events_in_range(from, to).unwrap();
        assert_eq!(counted.len(), 2);
        assert!(counted.iter().all(|e| e.uuid != "fp"));
        assert_eq!(store.event_count().unwrap(), 2);

        // History still shows all three, so the report can be undone.
        let recent = store.recent_events(from, to).unwrap();
        assert_eq!(recent.len(), 3);

        // A correction relabels rather than erases.
        let fixed = store.get_event("fix").unwrap().unwrap();
        assert_eq!(fixed.effective_type(), EventType::NoseBlow);
        assert!(fixed.counts());

        // Undo restores it to the counted set.
        store.clear_flag("fp").unwrap();
        assert_eq!(store.events_in_range(from, to).unwrap().len(), 3);
    }

    #[test]
    fn intensity_round_trips_and_tolerates_older_rows() {
        let store = Store::open_in_memory().unwrap();
        store.insert_event(&sample_event("loud")).unwrap();
        let got = store.get_event("loud").unwrap().unwrap();
        assert_eq!(got.peak_dbfs, Some(-12.0));
        assert_eq!(got.mean_dbfs, Some(-24.0));
        assert_eq!(got.noise_floor_dbfs, Some(-55.0));

        // Rows written before loudness existed simply have none.
        let mut quiet = sample_event("legacy");
        quiet.peak_dbfs = None;
        quiet.mean_dbfs = None;
        quiet.noise_floor_dbfs = None;
        store.insert_event(&quiet).unwrap();
        assert_eq!(store.get_event("legacy").unwrap().unwrap().peak_dbfs, None);
    }

    /// Deletes are soft so they can be propagated to the PHR, and a pull must
    /// not resurrect something deleted here.
    #[test]
    fn enrollment_deletes_are_soft_and_sync() {
        let store = Store::open_in_memory().unwrap();
        let id = store
            .add_enrollment(EventType::Hawk, &[0.1, 0.2], false)
            .unwrap();

        let pending = store.pending_enrollments(10).unwrap();
        assert_eq!(pending.len(), 1);
        let uuid = pending[0].uuid.clone();
        assert_eq!(uuid.len(), 16, "uuid must be raw 16 bytes");

        store
            .mark_enrollments_synced(std::slice::from_ref(&uuid))
            .unwrap();
        assert!(store.pending_enrollments(10).unwrap().is_empty());

        store.delete_enrollment(id).unwrap();
        assert!(store.enrollments().unwrap().is_empty());
        assert_eq!(
            store.pending_enrollment_deletions(10).unwrap(),
            vec![uuid.clone()]
        );

        // A pull must not resurrect it while the tombstone is still in flight.
        let resurrect = RemoteEnrollment {
            uuid: uuid.clone(),
            class: EventType::Hawk,
            embedding: vec![0.1, 0.2],
            is_negative: false,
            created_at: Utc::now().to_rfc3339(),
            similarity: None,
            separation: None,
            peak_dbfs: None,
            model_version: None,
            source_event_uuid: None,
            negative_scoped: false,
        };
        assert!(!store.upsert_remote_enrollment(&resurrect).unwrap());
        assert!(store.enrollments().unwrap().is_empty());

        store.purge_enrollments(&[uuid]).unwrap();
        assert!(store.pending_enrollment_deletions(10).unwrap().is_empty());
    }

    #[test]
    fn settings_doc_carries_only_synced_keys() {
        let store = Store::open_in_memory().unwrap();
        store.setting_set("sensitivity", "0.7").unwrap();
        store.setting_set("quiet_start", "22").unwrap();
        // Device-local: must never travel.
        store
            .setting_set("server_url", "https://example.test")
            .unwrap();
        store.setting_set("patient_id", "42").unwrap();
        store.setting_set("mode", "offline-strict").unwrap();

        let doc = store.settings_doc().unwrap();
        assert_eq!(doc.values.len(), 2);
        assert_eq!(
            doc.values.get("sensitivity").map(String::as_str),
            Some("0.7")
        );
        assert_eq!(
            doc.values.get("quiet_start").map(String::as_str),
            Some("22")
        );
        assert!(!doc.values.contains_key("server_url"));
        assert!(!doc.values.contains_key("mode"));
        assert!(doc.updated_at.is_some());
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
        assert!(!all[0].created_at.is_empty());
        assert_eq!(all[0].similarity, None);
        let counts = store.enrollment_counts().unwrap();
        assert_eq!(counts[&EventType::Hawk], 2); // negatives excluded
        store.delete_enrollment(id).unwrap();
        assert_eq!(store.enrollments().unwrap().len(), 2);
        assert_eq!(
            store.delete_enrollments_for_class(EventType::Hawk).unwrap(),
            2
        );
        assert!(store.enrollments().unwrap().is_empty());
        store
            .add_enrollment_with_quality(
                EventType::Sniffle,
                &[0.2, 0.3],
                false,
                Some(0.88),
                Some(0.12),
            )
            .unwrap();
        let quality = store.enrollments().unwrap();
        assert_eq!(quality[0].similarity, Some(0.88));
        assert_eq!(quality[0].separation, Some(0.12));
        assert_eq!(store.delete_all_enrollments().unwrap(), 1);
    }

    #[test]
    fn event_embeddings_roundtrip_and_prune() {
        let store = Store::open_in_memory().unwrap();
        let mut recent = sample_event("recent");
        recent.occurred_at = Utc::now();
        let mut old = sample_event("old");
        old.occurred_at = Utc::now() - chrono::Duration::days(60);
        store.insert_event(&recent).unwrap();
        store.insert_event(&old).unwrap();

        store.put_event_embedding("recent", &[0.1, 0.2]).unwrap();
        store.put_event_embedding("old", &[0.3, 0.4]).unwrap();
        // Empty embeddings are not stored.
        store.put_event_embedding("recent2", &[]).unwrap();
        assert_eq!(store.get_event_embedding("recent2").unwrap(), None);
        assert_eq!(
            store.get_event_embedding("recent").unwrap().unwrap(),
            vec![0.1, 0.2]
        );

        // Prune drops the aged-out event's embedding, keeps the recent one.
        let cutoff = Utc::now() - chrono::Duration::days(30);
        assert_eq!(store.prune_event_embeddings(cutoff).unwrap(), 1);
        assert!(store.get_event_embedding("old").unwrap().is_none());
        assert!(store.get_event_embedding("recent").unwrap().is_some());

        // A tombstoned event's embedding is also prunable.
        store.mark_deleted("recent").unwrap();
        assert_eq!(store.prune_event_embeddings(cutoff).unwrap(), 1);

        store.put_event_embedding("recent", &[0.5]).unwrap();
        store.delete_event_embedding("recent").unwrap();
        assert!(store.get_event_embedding("recent").unwrap().is_none());
    }

    #[test]
    fn negative_enrollments_count_and_reset_independently() {
        let store = Store::open_in_memory().unwrap();
        store
            .add_enrollment(EventType::Hawk, &[0.1, 0.2], false)
            .unwrap();
        store
            .add_enrollment(EventType::Cough, &[0.3, 0.4], true)
            .unwrap();
        store
            .add_enrollment(EventType::Sniffle, &[0.5, 0.6], true)
            .unwrap();
        assert_eq!(store.negative_enrollment_count().unwrap(), 2);
        assert_eq!(store.delete_negative_enrollments().unwrap(), 2);
        assert_eq!(store.negative_enrollment_count().unwrap(), 0);
        // The positive take survives.
        assert_eq!(store.enrollments().unwrap().len(), 1);
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

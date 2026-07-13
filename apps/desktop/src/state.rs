//! Non-UI app state (SPEC §6). Everything here is unit-tested; the egui layer is
//! a thin renderer over it. Handles today's counts, the 7-day histogram + the
//! congestion score, and the pause state machine.

use std::collections::HashMap;

use chrono::{DateTime, Duration, NaiveDate, Utc};
use sinus_core::store::Store;
use sinus_core::types::EventType;

/// Count events per class in `[from, to)`.
pub fn counts_in_range(
    store: &Store,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> HashMap<EventType, i64> {
    let mut counts: HashMap<EventType, i64> = HashMap::new();
    if let Ok(events) = store.events_in_range(from, to) {
        for e in events {
            *counts.entry(e.event_type).or_insert(0) += 1;
        }
    }
    counts
}

/// Counts for the UTC day containing `now`.
pub fn today_counts(store: &Store, now: DateTime<Utc>) -> HashMap<EventType, i64> {
    let start = now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc();
    let end = start + Duration::days(1);
    counts_in_range(store, start, end)
}

/// One day's per-class counts for the trend chart.
#[derive(Debug, Clone, PartialEq)]
pub struct DayCount {
    pub date: NaiveDate,
    pub counts: HashMap<EventType, i64>,
}

impl DayCount {
    pub fn total(&self) -> i64 {
        self.counts.values().sum()
    }
}

/// The last `days` days (oldest first), each with per-class counts. The final
/// entry is the day containing `now`.
pub fn daily_histogram(store: &Store, days: i64, now: DateTime<Utc>) -> Vec<DayCount> {
    let today = now.date_naive();
    (0..days)
        .rev()
        .map(|d| {
            let date = today - Duration::days(d);
            let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
            let end = start + Duration::days(1);
            DayCount {
                date,
                counts: counts_in_range(store, start, end),
            }
        })
        .collect()
}

/// Daily congestion score (SPEC §2): weighted event sum normalized per monitored
/// hour. Blockage indicators weigh double, clearing actions 1, cough 0.5.
pub fn congestion_score(counts: &HashMap<EventType, i64>, monitored_hours: f64) -> f64 {
    let weighted: f64 = counts
        .iter()
        .map(|(et, &n)| et.congestion_weight() * n as f64)
        .sum();
    if monitored_hours <= 0.0 {
        return 0.0;
    }
    weighted / monitored_hours
}

/// Pause state (SPEC §6 tray: pause 15 min / 1 h / until resumed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseState {
    Running,
    PausedUntil(DateTime<Utc>),
    PausedIndefinite,
}

impl PauseState {
    /// Whether capture is currently paused at `now`.
    pub fn is_paused(&self, now: DateTime<Utc>) -> bool {
        match self {
            PauseState::Running => false,
            PauseState::PausedIndefinite => true,
            PauseState::PausedUntil(t) => now < *t,
        }
    }

    /// Remaining pause duration, if any.
    pub fn remaining(&self, now: DateTime<Utc>) -> Option<Duration> {
        match self {
            PauseState::PausedUntil(t) if now < *t => Some(*t - now),
            _ => None,
        }
    }

    /// Auto-clears an expired timed pause back to `Running`.
    pub fn normalized(self, now: DateTime<Utc>) -> PauseState {
        match self {
            PauseState::PausedUntil(t) if now >= t => PauseState::Running,
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinus_core::types::Source;

    fn event(et: EventType, at: DateTime<Utc>) -> sinus_core::types::Event {
        sinus_core::types::Event {
            uuid: uuid::Uuid::new_v4().to_string(),
            event_type: et,
            occurred_at: at,
            tz_offset_min: 0,
            duration_ms: 500,
            confidence: 0.7,
            burst_count: 1,
            model_version: "test@0".into(),
            source: Source::DesktopMac,
            device_id: "d".into(),
            uploaded_at: None,
            deleted: false,
            reject_count: 0,
            rejected_at: None,
        }
    }

    #[test]
    fn today_counts_group_by_class() {
        let store = Store::open_in_memory().unwrap();
        let now = Utc::now();
        store.insert_event(&event(EventType::Cough, now)).unwrap();
        store.insert_event(&event(EventType::Cough, now)).unwrap();
        store.insert_event(&event(EventType::Sniffle, now)).unwrap();
        let counts = today_counts(&store, now);
        assert_eq!(counts[&EventType::Cough], 2);
        assert_eq!(counts[&EventType::Sniffle], 1);
    }

    #[test]
    fn histogram_buckets_by_day() {
        let store = Store::open_in_memory().unwrap();
        let now = Utc::now();
        let yesterday = now - Duration::days(1);
        store.insert_event(&event(EventType::Cough, now)).unwrap();
        store
            .insert_event(&event(EventType::Hawk, yesterday))
            .unwrap();
        let hist = daily_histogram(&store, 7, now);
        assert_eq!(hist.len(), 7);
        assert_eq!(hist[6].total(), 1); // today
        assert_eq!(hist[5].total(), 1); // yesterday
        assert_eq!(hist[0].total(), 0); // 6 days ago
    }

    #[test]
    fn congestion_weights_blockage_double() {
        let mut counts = HashMap::new();
        counts.insert(EventType::Sniffle, 2); // 2 * 2.0 = 4.0
        counts.insert(EventType::Cough, 4); // 4 * 0.5 = 2.0
                                            // total weighted 6.0 over 3 monitored hours = 2.0
        assert_eq!(congestion_score(&counts, 3.0), 2.0);
        assert_eq!(congestion_score(&counts, 0.0), 0.0);
    }

    #[test]
    fn pause_state_machine() {
        let now = Utc::now();
        assert!(!PauseState::Running.is_paused(now));
        assert!(PauseState::PausedIndefinite.is_paused(now));

        let future = PauseState::PausedUntil(now + Duration::minutes(15));
        assert!(future.is_paused(now));
        assert!(future.remaining(now).is_some());
        assert_eq!(future.normalized(now), future);

        let past = PauseState::PausedUntil(now - Duration::minutes(1));
        assert!(!past.is_paused(now));
        assert_eq!(past.normalized(now), PauseState::Running);
    }
}

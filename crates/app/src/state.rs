//! UI-independent application state (SPEC §6). Handles today's counts, the
//! seven-day histogram, congestion score, and pause state machine.

use std::collections::HashMap;

use chrono::{DateTime, Duration, FixedOffset, NaiveDate, TimeZone, Utc};
use sinus_core::store::Store;
use sinus_core::types::EventType;

/// Count events per class in `[from, to)`.
pub fn counts_in_range(
    store: &Store,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> HashMap<EventType, i64> {
    let mut counts: HashMap<EventType, i64> = HashMap::new();
    // `events_in_range` already drops reported misdetections. Corrections do
    // still count — under the class the user corrected them to, matching the
    // PHR's `COALESCE(corrected_to_event_type, event_type)`.
    if let Ok(events) = store.events_in_range(from, to) {
        for e in events {
            *counts.entry(e.effective_type()).or_insert(0) += 1;
        }
    }
    counts
}

/// Counts for the UTC day containing `now`.
pub fn today_counts(store: &Store, now: DateTime<Utc>) -> HashMap<EventType, i64> {
    today_counts_at_offset(store, now, 0)
}

/// Counts for the local day containing `now` at a fixed UTC offset.
pub fn today_counts_at_offset(
    store: &Store,
    now: DateTime<Utc>,
    offset_minutes: i32,
) -> HashMap<EventType, i64> {
    let offset = fixed_offset(offset_minutes);
    let date = now.with_timezone(&offset).date_naive();
    let start = local_midnight_utc(date, offset);
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
    // The chart now stacks per-class series, so only tests still sum a day.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn total(&self) -> i64 {
        self.counts.values().sum()
    }
}

/// The last `days` days (oldest first), each with per-class counts. The final
/// entry is the day containing `now`.
pub fn daily_histogram(store: &Store, days: i64, now: DateTime<Utc>) -> Vec<DayCount> {
    daily_histogram_at_offset(store, days, now, 0)
}

/// Local-day buckets at a fixed UTC offset, oldest first.
pub fn daily_histogram_at_offset(
    store: &Store,
    days: i64,
    now: DateTime<Utc>,
    offset_minutes: i32,
) -> Vec<DayCount> {
    let offset = fixed_offset(offset_minutes);
    let today = now.with_timezone(&offset).date_naive();
    (0..days)
        .rev()
        .map(|d| {
            let date = today - Duration::days(d);
            let start = local_midnight_utc(date, offset);
            let end = start + Duration::days(1);
            DayCount {
                date,
                counts: counts_in_range(store, start, end),
            }
        })
        .collect()
}

fn fixed_offset(offset_minutes: i32) -> FixedOffset {
    FixedOffset::east_opt(offset_minutes.clamp(-1_439, 1_439) * 60)
        .expect("clamped timezone offset is valid")
}

pub fn local_midnight_at_offset(now: DateTime<Utc>, offset_minutes: i32) -> DateTime<Utc> {
    let offset = fixed_offset(offset_minutes);
    local_midnight_utc(now.with_timezone(&offset).date_naive(), offset)
}

fn local_midnight_utc(date: NaiveDate, offset: FixedOffset) -> DateTime<Utc> {
    offset
        .from_local_datetime(&date.and_hms_opt(0, 0, 0).expect("midnight is valid"))
        .single()
        .expect("fixed offsets have no ambiguous local times")
        .with_timezone(&Utc)
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
            peak_dbfs: Some(-15.0),
            mean_dbfs: Some(-28.0),
            noise_floor_dbfs: Some(-55.0),
            model_version: "test@0".into(),
            source: Source::DesktopMac,
            device_id: "d".into(),
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
    fn local_day_uses_the_supplied_offset() {
        let store = Store::open_in_memory().unwrap();
        let now = "2026-07-24T01:00:00Z".parse::<DateTime<Utc>>().unwrap();
        store
            .insert_event(&event(EventType::Cough, now - Duration::hours(2)))
            .unwrap();

        assert_eq!(
            today_counts_at_offset(&store, now, -7 * 60)
                .get(&EventType::Cough)
                .copied(),
            Some(1)
        );
        assert!(today_counts_at_offset(&store, now, 0).is_empty());
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

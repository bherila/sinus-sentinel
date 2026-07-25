//! Event-flagging policy: reporting misdetections, correcting them, and
//! undoing either — shared by every shell so the rules that feed the
//! prototype matcher cannot drift between platforms.
//!
//! Keyed by uuid rather than `&Event`: the Apple FFI boundary only ever hands
//! across a uuid, so loading the event here (rather than requiring the caller
//! to have one in hand) keeps this the single place the rules live.

use sinus_core::error::{Error, Result};
use sinus_core::store::{EnrollmentInsert, Store};
use sinus_core::types::EventType;

/// What a flag operation changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FlagOutcome {
    /// An enrollment was written, so the detector was actually adjusted — and,
    /// equivalently, the running matcher's prototypes are now stale and must be
    /// reloaded. A report on an event whose embedding was already pruned flags
    /// the event but trains nothing, which is worth telling the user.
    pub trained: bool,
}

fn outcome(trained: bool) -> FlagOutcome {
    FlagOutcome { trained }
}

fn load_event(store: &Store, event_uuid: &str) -> Result<sinus_core::types::Event> {
    store
        .get_event(event_uuid)?
        .ok_or_else(|| Error::Config(format!("no such event: {event_uuid}")))
}

/// Enroll this event's sound as "not `class`", if its embedding was retained.
///
/// `scoped` decides how far the veto reaches: a plain false-positive report
/// is unscoped (the sound is suppressed under every label, so a borderline
/// sound cannot simply re-fire as a sibling class), while the negative half
/// of a correction is scoped, because a positive for the corrected class
/// carries the same embedding.
fn enroll_negative(
    store: &Store,
    event: &sinus_core::types::Event,
    class: EventType,
    scoped: bool,
) -> bool {
    let Ok(Some(embedding)) = store.get_event_embedding(&event.uuid) else {
        return false;
    };
    store
        .add_enrollment_full(EnrollmentInsert {
            class,
            embedding: &embedding,
            is_negative: true,
            similarity: None,
            separation: None,
            peak_dbfs: event.peak_dbfs,
            model_version: Some(&event.model_version),
            source_event_uuid: Some(&event.uuid),
            negative_scoped: scoped,
        })
        .is_ok()
}

/// Report a misdetection: enroll the event's stored embedding as a negative
/// for the class that fired (so the detector stops calling that sound
/// *that*), then flag the event.
///
/// The event is flagged, not deleted: a health record should keep the fact
/// that the classifier got something wrong. It stops counting everywhere and
/// the flag syncs to the PHR, where it is likewise retained rather than
/// erased. The embedding is deliberately kept — the user may follow up by
/// saying what the sound actually was.
pub fn report_false_positive(store: &Store, event_uuid: &str) -> Result<FlagOutcome> {
    let event = load_event(store, event_uuid)?;
    let trained = enroll_negative(store, &event, event.event_type, false);
    store.mark_false_positive(&event.uuid)?;
    Ok(outcome(trained))
}

/// Record what a misdetected sound actually was.
///
/// Enrolls the embedding twice: as a negative against the class that fired,
/// and as a positive for the corrected one. Negatives are class-scoped, so
/// the wrong label is suppressed while the corrected label stays free to
/// fire — a class-blind negative would silence the sound entirely, which is
/// worse than the original mistake.
pub fn recharacterize(
    store: &Store,
    event_uuid: &str,
    corrected: EventType,
) -> Result<FlagOutcome> {
    let event = load_event(store, event_uuid)?;

    // Correcting back to what the classifier originally said is an undo, not
    // a correction. Treating it as one would enroll a negative *and* a
    // positive for that class from the same embedding, and the veto would
    // then suppress the very label the user just confirmed.
    if corrected == event.event_type {
        return clear_flag(store, event_uuid);
    }

    let embedding = store.get_event_embedding(&event.uuid).ok().flatten();

    let mut trained = false;
    if let Some(embedding) = &embedding {
        trained = enroll_negative(store, &event, event.event_type, true);
        let positive = store.add_enrollment_full(EnrollmentInsert {
            class: corrected,
            embedding,
            is_negative: false,
            similarity: None,
            separation: None,
            peak_dbfs: event.peak_dbfs,
            model_version: Some(&event.model_version),
            source_event_uuid: Some(&event.uuid),
            negative_scoped: false,
        });
        trained |= positive.is_ok();
    }

    store.recharacterize(&event.uuid, corrected)?;
    Ok(outcome(trained))
}

/// Undo a false-positive report or a correction. Training is deliberately
/// kept — the embedding-derived enrollments stay put unless the user removes
/// them explicitly (Teach mode), because the sound they describe was real
/// even if the flag on this particular event was a mistake.
pub fn clear_flag(store: &Store, event_uuid: &str) -> Result<FlagOutcome> {
    // Confirms the uuid exists (and reports it the same way the other two
    // operations do) even though the event itself isn't otherwise needed.
    load_event(store, event_uuid)?;
    store.clear_flag(event_uuid)?;
    Ok(FlagOutcome::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};
    use sinus_core::store::Store;
    use sinus_core::types::{Event, Source};

    fn event(et: EventType, at: DateTime<Utc>) -> Event {
        Event {
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
    fn correcting_to_the_original_class_is_an_undo() {
        let store = Store::open_in_memory().unwrap();
        let e = event(EventType::Cough, Utc::now());
        store.insert_event(&e).unwrap();
        store
            .put_event_embedding(&e.uuid, &[0.1, 0.2, 0.3])
            .unwrap();

        let outcome = recharacterize(&store, &e.uuid, EventType::Cough).unwrap();
        assert!(!outcome.trained);
        assert_eq!(store.enrollments().unwrap().len(), 0);

        let stored = store.get_event(&e.uuid).unwrap().unwrap();
        assert!(stored.false_positive_at.is_none());
        assert!(stored.corrected_to.is_none());
    }

    #[test]
    fn report_without_a_retained_embedding_still_flags() {
        let store = Store::open_in_memory().unwrap();
        let e = event(EventType::Cough, Utc::now());
        store.insert_event(&e).unwrap();

        let outcome = report_false_positive(&store, &e.uuid).unwrap();
        assert!(!outcome.trained);

        let stored = store.get_event(&e.uuid).unwrap().unwrap();
        assert!(stored.false_positive_at.is_some());
    }

    #[test]
    fn correction_writes_a_scoped_negative_and_a_positive() {
        let store = Store::open_in_memory().unwrap();
        let e = event(EventType::Cough, Utc::now());
        store.insert_event(&e).unwrap();
        store
            .put_event_embedding(&e.uuid, &[0.1, 0.2, 0.3])
            .unwrap();

        let outcome = recharacterize(&store, &e.uuid, EventType::Sniffle).unwrap();
        assert!(outcome.trained);

        let enrollments = store.enrollments().unwrap();
        assert_eq!(enrollments.len(), 2);

        let negative = enrollments
            .iter()
            .find(|en| en.enrollment.is_negative)
            .expect("negative enrollment");
        assert_eq!(negative.enrollment.class, EventType::Cough);
        assert!(negative.enrollment.negative_scoped);
        assert_eq!(negative.source_event_uuid.as_deref(), Some(e.uuid.as_str()));

        let positive = enrollments
            .iter()
            .find(|en| !en.enrollment.is_negative)
            .expect("positive enrollment");
        assert_eq!(positive.enrollment.class, EventType::Sniffle);
        assert_eq!(positive.source_event_uuid.as_deref(), Some(e.uuid.as_str()));

        let stored = store.get_event(&e.uuid).unwrap().unwrap();
        assert_eq!(stored.corrected_to, Some(EventType::Sniffle));
    }

    #[test]
    fn report_writes_an_unscoped_negative() {
        let store = Store::open_in_memory().unwrap();
        let e = event(EventType::Cough, Utc::now());
        store.insert_event(&e).unwrap();
        store
            .put_event_embedding(&e.uuid, &[0.1, 0.2, 0.3])
            .unwrap();

        let outcome = report_false_positive(&store, &e.uuid).unwrap();
        assert!(outcome.trained);

        let enrollments = store.enrollments().unwrap();
        assert_eq!(enrollments.len(), 1);
        assert!(enrollments[0].enrollment.is_negative);
        assert!(!enrollments[0].enrollment.negative_scoped);
        assert_eq!(enrollments[0].enrollment.class, EventType::Cough);
    }

    #[test]
    fn unknown_uuid_errors() {
        let store = Store::open_in_memory().unwrap();
        let error = report_false_positive(&store, "not-a-real-uuid").unwrap_err();
        assert!(error.to_string().contains("not-a-real-uuid"));
    }
}

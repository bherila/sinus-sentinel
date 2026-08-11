//! Event-feedback policy shared by every shell.
//!
//! The store owns the atomic persistence boundary. This module translates
//! user intent into a canonical feedback state and turns the store's raw
//! mutation facts into the stable domain result exposed through UniFFI.

use sinus_core::classify::proto::MIN_POSITIVE_EXAMPLES;
use sinus_core::error::{Error, Result};
use sinus_core::store::{EventFeedbackState, FeedbackMutationInput, FeedbackMutationResult, Store};
use sinus_core::types::EventType;

use crate::settings;

/// How a feedback operation affected personalized detection.
///
/// This is deliberately independent of [`FeedbackOutcome::classifier_changed`].
/// Re-correcting an event whose replacement embedding is unavailable can remove
/// the old classifier effect (`classifier_changed = true`) while still reporting
/// `Unavailable` for the requested replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingEffect {
    Applied,
    Removed,
    Unavailable,
    Unchanged,
}

/// Positive-class activation progress after a feedback operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingProgress {
    pub class: EventType,
    pub positive_examples: u32,
    pub activation_threshold: u32,
}

/// Independent facts about one canonical event-feedback mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedbackOutcome {
    /// The canonical feedback/effective event state changed.
    pub event_changed: bool,
    /// The virtual enrollment view changed and a live matcher must reload.
    pub classifier_changed: bool,
    /// Durable work for this event remains on a currently supported sync path.
    pub sync_required: bool,
    pub training_effect: TrainingEffect,
    /// Present for feedback whose desired classifier effect has a positive class.
    pub progress: Option<TrainingProgress>,
}

/// Result of clearing every event-owned feedback group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BulkFeedbackOutcome {
    pub groups_changed: usize,
    pub classifier_changed: bool,
    pub sync_required: bool,
}

fn apply_feedback(
    store: &Store,
    event_uuid: &str,
    desired_state: EventFeedbackState,
    target_class: Option<EventType>,
) -> Result<FeedbackOutcome> {
    let actor_device_id = settings::ensure_device_id(store);
    let mutation = store.apply_event_feedback(
        event_uuid,
        FeedbackMutationInput {
            desired_state,
            target_class,
            actor_device_id,
        },
    )?;
    outcome(store, desired_state, mutation)
}

fn outcome(
    store: &Store,
    desired_state: EventFeedbackState,
    mutation: FeedbackMutationResult,
) -> Result<FeedbackOutcome> {
    let training_effect = if !mutation.event_changed && !mutation.classifier_changed {
        TrainingEffect::Unchanged
    } else if desired_state == EventFeedbackState::None {
        if mutation.classifier_changed {
            TrainingEffect::Removed
        } else {
            TrainingEffect::Unchanged
        }
    } else if mutation.embedding_available {
        TrainingEffect::Applied
    } else {
        TrainingEffect::Unavailable
    };

    let progress = mutation
        .positive_class
        .map(|class| -> Result<TrainingProgress> {
            let positive_examples = store
                .classifier_enrollment_counts()?
                .get(&class)
                .copied()
                .unwrap_or(0)
                .clamp(0, u32::MAX as i64) as u32;
            Ok(TrainingProgress {
                class,
                positive_examples,
                activation_threshold: MIN_POSITIVE_EXAMPLES as u32,
            })
        })
        .transpose()?;

    Ok(FeedbackOutcome {
        event_changed: mutation.event_changed,
        classifier_changed: mutation.classifier_changed,
        sync_required: mutation.sync_required,
        training_effect,
        progress,
    })
}

/// Explicitly confirm the detector's current effective label.
///
/// Confirmation never changes what the event counts as. When an embedding is
/// recoverable, it derives one positive classifier example for that class.
pub fn confirm_event(store: &Store, event_uuid: &str) -> Result<FeedbackOutcome> {
    apply_feedback(store, event_uuid, EventFeedbackState::Confirmed, None)
}

/// Report a misdetection. The event remains in history but stops counting, and
/// its embedding (when recoverable) becomes an unscoped negative.
pub fn report_false_positive(store: &Store, event_uuid: &str) -> Result<FeedbackOutcome> {
    apply_feedback(store, event_uuid, EventFeedbackState::FalsePositive, None)
}

/// Correct the event to another class.
///
/// Correcting back to the classifier's original class is Undo, not a
/// confirmation: it clears feedback and every classifier effect owned by the
/// event rather than deriving contradictory positive and negative examples.
pub fn recharacterize(
    store: &Store,
    event_uuid: &str,
    corrected: EventType,
) -> Result<FeedbackOutcome> {
    let event = store
        .get_event(event_uuid)?
        .ok_or_else(|| Error::Config(format!("no such event: {event_uuid}")))?;
    if corrected == event.event_type {
        return clear_flag(store, event_uuid);
    }
    apply_feedback(
        store,
        event_uuid,
        EventFeedbackState::Corrected,
        Some(corrected),
    )
}

/// Undo confirmation, correction, or a false-positive report.
///
/// The canonical `none` document is retained so #21 can later propagate a
/// clearing state to other devices. Guided Teach rows are never part of this
/// event-owned mutation.
pub fn clear_flag(store: &Store, event_uuid: &str) -> Result<FeedbackOutcome> {
    apply_feedback(store, event_uuid, EventFeedbackState::None, None)
}

/// Clear every feedback-derived training unit while leaving guided Teach rows
/// untouched. The store performs the complete reset atomically.
pub fn clear_all_feedback(store: &Store) -> Result<BulkFeedbackOutcome> {
    let actor_device_id = settings::ensure_device_id(store);
    let mutation = store.clear_all_event_feedback(&actor_device_id)?;
    Ok(BulkFeedbackOutcome {
        groups_changed: mutation.groups_changed,
        classifier_changed: mutation.classifier_changed,
        sync_required: mutation.sync_required,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};
    use sinus_core::store::EnrollmentInsert;
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
            device_id: "capture-device".into(),
            uploaded_at: None,
            deleted: false,
            false_positive_at: None,
            corrected_to: None,
            corrected_at: None,
            reject_count: 0,
            rejected_at: None,
        }
    }

    fn stored_event(store: &Store, class: EventType, with_embedding: bool) -> Event {
        let event = event(class, Utc::now());
        store.insert_event(&event).unwrap();
        if with_embedding {
            store
                .put_event_embedding(&event.uuid, &[0.1, 0.2, 0.3])
                .unwrap();
        }
        event
    }

    fn assert_flags(
        outcome: FeedbackOutcome,
        event_changed: bool,
        classifier_changed: bool,
        sync_required: bool,
        effect: TrainingEffect,
    ) {
        assert_eq!(outcome.event_changed, event_changed);
        assert_eq!(outcome.classifier_changed, classifier_changed);
        assert_eq!(outcome.sync_required, sync_required);
        assert_eq!(outcome.training_effect, effect);
    }

    #[test]
    fn correction_derives_a_scoped_negative_and_positive_with_progress() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);

        let outcome = recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();
        assert_flags(outcome, true, true, true, TrainingEffect::Applied);
        assert_eq!(
            outcome.progress,
            Some(TrainingProgress {
                class: EventType::Sniffle,
                positive_examples: 1,
                activation_threshold: MIN_POSITIVE_EXAMPLES as u32,
            })
        );

        let enrollments = store.classifier_enrollments().unwrap();
        assert_eq!(enrollments.len(), 2);
        let negative = enrollments.iter().find(|item| item.is_negative).unwrap();
        assert_eq!(negative.class, EventType::Cough);
        assert!(negative.negative_scoped);
        let positive = enrollments.iter().find(|item| !item.is_negative).unwrap();
        assert_eq!(positive.class, EventType::Sniffle);

        let stored = store.get_event(&event.uuid).unwrap().unwrap();
        assert_eq!(stored.corrected_to, Some(EventType::Sniffle));
    }

    #[test]
    fn correction_without_an_embedding_changes_the_event_but_is_unavailable() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, false);

        let outcome = recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();
        assert_flags(outcome, true, false, true, TrainingEffect::Unavailable);
        assert_eq!(
            outcome.progress,
            Some(TrainingProgress {
                class: EventType::Sniffle,
                positive_examples: 0,
                activation_threshold: MIN_POSITIVE_EXAMPLES as u32,
            })
        );
        assert!(store.classifier_enrollments().unwrap().is_empty());
        assert_eq!(
            store.get_event(&event.uuid).unwrap().unwrap().corrected_to,
            Some(EventType::Sniffle)
        );
    }

    #[test]
    fn repeated_correction_is_idempotent_but_reports_existing_pending_sync() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);
        recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();

        let repeated = recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();
        assert_flags(repeated, false, false, true, TrainingEffect::Unchanged);
        assert_eq!(store.classifier_enrollments().unwrap().len(), 2);
    }

    #[test]
    fn recorrection_reuses_the_feedback_embedding_after_event_embedding_expiry() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);
        recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();
        store.delete_event_embedding(&event.uuid).unwrap();

        let outcome = recharacterize(&store, &event.uuid, EventType::NoseBlow).unwrap();
        assert_flags(outcome, true, true, true, TrainingEffect::Applied);
        let enrollments = store.classifier_enrollments().unwrap();
        assert_eq!(enrollments.len(), 2);
        assert!(enrollments.iter().any(|item| {
            item.is_negative && item.negative_scoped && item.class == EventType::Cough
        }));
        assert!(enrollments
            .iter()
            .any(|item| !item.is_negative && item.class == EventType::NoseBlow));
        assert!(!enrollments
            .iter()
            .any(|item| !item.is_negative && item.class == EventType::Sniffle));
    }

    #[test]
    fn recorrection_without_any_embedding_removes_no_longer_valid_effects() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, false);
        recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();

        let outcome = recharacterize(&store, &event.uuid, EventType::NoseBlow).unwrap();
        assert_flags(outcome, true, false, true, TrainingEffect::Unavailable);
        assert!(store.classifier_enrollments().unwrap().is_empty());
    }

    #[test]
    fn undo_removes_event_owned_training_and_keeps_a_none_document() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);
        recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();

        let outcome = clear_flag(&store, &event.uuid).unwrap();
        assert_flags(outcome, true, true, true, TrainingEffect::Removed);
        assert!(outcome.progress.is_none());
        assert!(store.classifier_enrollments().unwrap().is_empty());

        let feedback = store.event_feedback(&event.uuid).unwrap().unwrap();
        assert_eq!(feedback.state, EventFeedbackState::None);
        let stored = store.get_event(&event.uuid).unwrap().unwrap();
        assert!(stored.false_positive_at.is_none());
        assert!(stored.corrected_to.is_none());
    }

    #[test]
    fn repeated_undo_is_an_idempotent_noop_with_pending_sync_truth() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);
        report_false_positive(&store, &event.uuid).unwrap();
        clear_flag(&store, &event.uuid).unwrap();

        let repeated = clear_flag(&store, &event.uuid).unwrap();
        assert_flags(repeated, false, false, true, TrainingEffect::Unchanged);
    }

    #[test]
    fn false_positive_derives_one_unscoped_negative() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);

        let outcome = report_false_positive(&store, &event.uuid).unwrap();
        assert_flags(outcome, true, true, true, TrainingEffect::Applied);
        assert!(outcome.progress.is_none());
        let enrollments = store.classifier_enrollments().unwrap();
        assert_eq!(enrollments.len(), 1);
        assert!(enrollments[0].is_negative);
        assert!(!enrollments[0].negative_scoped);
        assert_eq!(enrollments[0].class, EventType::Cough);
    }

    #[test]
    fn confirmation_is_device_local_until_canonical_sync_exists() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);

        let outcome = confirm_event(&store, &event.uuid).unwrap();
        assert_flags(outcome, true, true, false, TrainingEffect::Applied);
        assert_eq!(outcome.progress.unwrap().class, EventType::Cough);
        let stored = store.get_event(&event.uuid).unwrap().unwrap();
        assert!(stored.false_positive_at.is_none());
        assert!(stored.corrected_to.is_none());

        let repeated = confirm_event(&store, &event.uuid).unwrap();
        assert_flags(repeated, false, false, false, TrainingEffect::Unchanged);
        assert_eq!(store.classifier_enrollments().unwrap().len(), 1);
    }

    #[test]
    fn repeated_false_positive_is_idempotent_with_pending_sync_truth() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);
        report_false_positive(&store, &event.uuid).unwrap();

        let repeated = report_false_positive(&store, &event.uuid).unwrap();
        assert_flags(repeated, false, false, true, TrainingEffect::Unchanged);
        assert_eq!(store.classifier_enrollments().unwrap().len(), 1);
    }

    #[test]
    fn clearing_an_event_that_never_had_feedback_is_unchanged() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);

        let outcome = clear_flag(&store, &event.uuid).unwrap();
        assert_flags(outcome, false, false, false, TrainingEffect::Unchanged);
        assert!(store.event_feedback(&event.uuid).unwrap().is_none());
    }

    #[test]
    fn correcting_to_the_original_class_is_undo() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);
        recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();

        let outcome = recharacterize(&store, &event.uuid, EventType::Cough).unwrap();
        assert_flags(outcome, true, true, true, TrainingEffect::Removed);
        assert!(store.classifier_enrollments().unwrap().is_empty());
    }

    #[test]
    fn undo_never_removes_guided_teach_rows() {
        let store = Store::open_in_memory().unwrap();
        let event = stored_event(&store, EventType::Cough, true);
        store
            .add_enrollment_full(EnrollmentInsert {
                class: EventType::Sniffle,
                embedding: &[0.9, 0.1, 0.0],
                is_negative: false,
                similarity: None,
                separation: None,
                peak_dbfs: None,
                model_version: Some("test@0"),
                source_event_uuid: None,
                negative_scoped: false,
            })
            .unwrap();

        let corrected = recharacterize(&store, &event.uuid, EventType::Sniffle).unwrap();
        assert_eq!(corrected.progress.unwrap().positive_examples, 2);
        clear_flag(&store, &event.uuid).unwrap();

        let remaining = store.classifier_enrollments().unwrap();
        assert_eq!(remaining.len(), 1);
        assert!(!remaining[0].is_negative);
        assert_eq!(remaining[0].class, EventType::Sniffle);
        assert_eq!(store.enrollments().unwrap().len(), 1);
    }

    #[test]
    fn clear_all_feedback_is_atomic_and_preserves_guided_takes() {
        let store = Store::open_in_memory().unwrap();
        let corrected = stored_event(&store, EventType::Cough, true);
        let reported = stored_event(&store, EventType::Sneeze, true);
        recharacterize(&store, &corrected.uuid, EventType::Sniffle).unwrap();
        report_false_positive(&store, &reported.uuid).unwrap();
        store
            .add_enrollment(EventType::Hawk, &[0.9, 0.1, 0.0], false)
            .unwrap();

        let outcome = clear_all_feedback(&store).unwrap();
        assert_eq!(outcome.groups_changed, 2);
        assert!(outcome.classifier_changed);
        assert!(outcome.sync_required);

        let remaining = store.classifier_enrollments().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].class, EventType::Hawk);
        assert!(!remaining[0].is_negative);
    }

    #[test]
    fn unknown_uuid_errors() {
        let store = Store::open_in_memory().unwrap();
        let error = report_false_positive(&store, "not-a-real-uuid").unwrap_err();
        assert!(error.to_string().contains("not-a-real-uuid"));
    }
}

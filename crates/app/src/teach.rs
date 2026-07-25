//! Teach-mode policy (SPEC §5 Phase B-lite): the recording countdown, take
//! scoring against the personalized matcher, and the per-class training status
//! shown in Settings. Owned here rather than in a platform shell because the
//! SwiftUI client needs identical scoring — a divergent reimplementation would
//! mislabel training quality (calling a class "ready" on one platform and not
//! the other) even though both read the same store.

use sinus_core::classify::embed::Embedder;
use sinus_core::classify::proto::{Enrollment, PrototypeMatcher};
use sinus_core::error::{Error, Result};
use sinus_core::mel::loudest_patch;
use sinus_core::pipeline::StreamingPipeline;
use sinus_core::store::{EnrollmentInsert, Store, StoredEnrollment};
use sinus_core::types::{EventType, SAMPLE_RATE};

use crate::monitor::{PROTOTYPE_NEGATIVE_MARGIN, PROTOTYPE_SIM_THRESHOLD};

/// Silent lead-in before recording starts, giving the user a beat to react to
/// "get ready" before the mic starts counting toward the take.
pub const TEACH_COUNTDOWN_SAMPLES: usize = SAMPLE_RATE as usize;
/// Length of one recorded take.
pub const TEACH_TAKE_SAMPLES: usize = SAMPLE_RATE as usize * 3;
/// Takes needed before a personalized class is eligible to fire on its own —
/// matches `proto::MIN_POSITIVE_EXAMPLES`, kept as its own constant here so
/// Teach-mode UI copy doesn't have to reach into the matcher internals.
pub const MIN_TAKES: usize = 3;
/// Same-class repeat similarity a take must clear to count as "good".
pub const GOOD_SIMILARITY: f32 = 0.75;
/// Margin over the closest other class a take must clear to count as "good".
pub const GOOD_SEPARATION: f32 = 0.05;

/// Buffers one Teach-mode take: discards the countdown lead-in, then keeps
/// exactly `TEACH_TAKE_SAMPLES` of microphone audio.
pub struct TakeBuffer {
    class: EventType,
    samples: Vec<f32>,
    countdown_remaining: usize,
}

impl TakeBuffer {
    pub fn new(class: EventType) -> Self {
        TakeBuffer {
            class,
            samples: Vec::with_capacity(TEACH_TAKE_SAMPLES),
            countdown_remaining: TEACH_COUNTDOWN_SAMPLES,
        }
    }

    /// Consume microphone samples, returning true exactly when the countdown
    /// crosses into recording. Samples before that edge are discarded.
    pub fn push(&mut self, input: &[f32]) -> bool {
        let was_counting_down = self.countdown_remaining > 0;
        let countdown_samples = self.countdown_remaining.min(input.len());
        self.countdown_remaining -= countdown_samples;
        let started_recording = was_counting_down && self.countdown_remaining == 0;

        if self.countdown_remaining == 0 {
            let remaining = TEACH_TAKE_SAMPLES.saturating_sub(self.samples.len());
            let available = input.len().saturating_sub(countdown_samples).min(remaining);
            self.samples.extend_from_slice(
                &input[countdown_samples..countdown_samples.saturating_add(available)],
            );
        }
        started_recording
    }

    pub fn is_complete(&self) -> bool {
        self.samples.len() >= TEACH_TAKE_SAMPLES
    }

    pub fn class(&self) -> EventType {
        self.class
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

/// The outcome of enrolling one take.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeachResult {
    pub class: EventType,
    pub examples: usize,
    /// Negative when this was the class's first take — no validation possible
    /// against a prototype that does not exist yet.
    pub similarity: f32,
    pub separation: f32,
    pub peak_dbfs: Option<f32>,
}

impl TeachResult {
    /// Whether this take is strong enough, on its own, to call the class
    /// trained — the same bar `class_status` applies to a class's latest take.
    pub fn is_good(&self) -> bool {
        self.examples >= MIN_TAKES
            && self.similarity >= GOOD_SIMILARITY
            && self.separation >= GOOD_SEPARATION
    }
}

/// Enroll one recorded take: pick its loudest analysis window, embed it, score
/// it against the existing prototypes, and store it. Does not reload the live
/// detector's prototypes — that needs `&mut StreamingPipeline` and is the
/// caller's job once the store write lands.
pub fn enroll_take<E: Embedder>(
    store: &Store,
    pipeline: &StreamingPipeline<E>,
    class: EventType,
    samples: &[f32],
) -> Result<TeachResult> {
    let patch = loudest_patch(samples)
        .ok_or_else(|| Error::Config("no complete analysis window captured".to_string()))?;
    let embedding = pipeline.embed_patch(&patch, true)?;
    let (similarity, separation) = score_embedding(store, class, &embedding)?;

    // The very first take of a class has nothing to score against; leave the
    // quality columns null rather than recording a meaningless -1/0 pair that
    // would otherwise look like a real (and terrible) score.
    let quality = similarity >= 0.0;
    let peak = peak_dbfs(samples);
    let model_version = pipeline.model_version();
    store.add_enrollment_full(EnrollmentInsert {
        class,
        embedding: &embedding,
        is_negative: false,
        similarity: quality.then_some(similarity),
        separation: quality.then_some(separation),
        peak_dbfs: peak,
        model_version: Some(&model_version),
        source_event_uuid: None,
        // Irrelevant for a positive take; scoping only governs how far a
        // negative's veto reaches.
        negative_scoped: false,
    })?;

    let examples = store.enrollment_counts()?.get(&class).copied().unwrap_or(0) as usize;

    Ok(TeachResult {
        class,
        examples,
        similarity,
        separation,
        peak_dbfs: peak,
    })
}

/// Same-class score and separation for an already-computed embedding, against
/// every non-deleted enrollment in the store. `(-1.0, 0.0)` when no positive
/// example of `class` is enrolled yet — there is nothing to compare against.
pub fn score_embedding(store: &Store, class: EventType, embedding: &[f32]) -> Result<(f32, f32)> {
    let all_enrollments: Vec<Enrollment> = store
        .enrollments()?
        .into_iter()
        .map(|stored| stored.enrollment)
        .collect();
    let has_same_class = all_enrollments
        .iter()
        .any(|example| example.class == class && !example.is_negative);
    if !has_same_class {
        return Ok((-1.0, 0.0));
    }

    let matcher = PrototypeMatcher::from_enrollments(
        &all_enrollments,
        PROTOTYPE_SIM_THRESHOLD,
        PROTOTYPE_NEGATIVE_MARGIN,
    );
    let similarities = matcher.similarities(embedding);
    let same = similarities
        .iter()
        .find(|(candidate, _)| *candidate == class)
        .map(|(_, score)| *score)
        .unwrap_or(-1.0);
    let other = similarities
        .iter()
        .filter(|(candidate, _)| *candidate != class)
        .map(|(_, score)| *score)
        .fold(-1.0f32, f32::max);
    Ok((same, same - other))
}

/// Loudest 50 ms hop in a buffer, dBFS — the same measure events record, so a
/// take that matches poorly can be checked against how loud it actually was.
/// `None` for a buffer shorter than one hop.
pub fn peak_dbfs(samples: &[f32]) -> Option<f32> {
    const HOP: usize = SAMPLE_RATE as usize / 20;

    samples
        .chunks_exact(HOP)
        .map(|hop| {
            let sum_sq: f64 = hop.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let rms = (sum_sq / hop.len() as f64).sqrt();
            20.0 * (rms + 1e-9).log10() as f32
        })
        .fold(None, |best: Option<f32>, db| {
            Some(best.map_or(db, |b| b.max(db)))
        })
}

/// Where a class stands in Settings — how many takes it has and whether it is
/// good enough to fire on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassStatus {
    Untrained,
    Inactive { needed: usize },
    Active,
    Ready,
}

/// Status for one class from its positive takes, oldest first (as `Store`
/// returns them) so "latest" unambiguously means "just recorded". Shared by
/// the per-class badge and `training_snapshot` so the two cannot disagree
/// about what counts as the newest take.
pub fn class_status(takes: &[&StoredEnrollment]) -> ClassStatus {
    let count = takes.len();
    if count == 0 {
        return ClassStatus::Untrained;
    }
    if count < MIN_TAKES {
        return ClassStatus::Inactive {
            needed: MIN_TAKES - count,
        };
    }
    let latest_good = takes.last().is_some_and(|latest| {
        latest
            .similarity
            .is_some_and(|value| value >= GOOD_SIMILARITY)
            && latest
                .separation
                .is_some_and(|value| value >= GOOD_SEPARATION)
    });
    if latest_good {
        ClassStatus::Ready
    } else {
        ClassStatus::Active
    }
}

pub struct ClassTraining {
    pub class: EventType,
    pub status: ClassStatus,
    pub takes: Vec<StoredEnrollment>,
}

/// Every class in `EventType::ALL` order, with its positive takes and derived
/// status, plus the count of learned false-positive suppressions.
pub fn training_snapshot(store: &Store) -> Result<(Vec<ClassTraining>, usize)> {
    let enrollments = store.enrollments()?;
    let mut classes = Vec::with_capacity(EventType::ALL.len());
    for class in EventType::ALL {
        let takes: Vec<StoredEnrollment> = enrollments
            .iter()
            .filter(|stored| stored.enrollment.class == class && !stored.enrollment.is_negative)
            .cloned()
            .collect();
        let status = class_status(&takes.iter().collect::<Vec<_>>());
        classes.push(ClassTraining {
            class,
            status,
            takes,
        });
    }
    let negative_count = enrollments
        .iter()
        .filter(|stored| stored.enrollment.is_negative)
        .count();
    Ok((classes, negative_count))
}

/// What to remove from stored training.
#[derive(Debug, Clone, Copy)]
pub enum Deletion {
    One(i64),
    Class(EventType),
    Negatives,
    All,
}

/// Remove enrollments per `what`, returning how many rows were removed.
pub fn delete(store: &Store, what: Deletion) -> Result<usize> {
    match what {
        Deletion::One(id) => {
            store.delete_enrollment(id)?;
            Ok(1)
        }
        Deletion::Class(class) => store.delete_enrollments_for_class(class),
        Deletion::Negatives => store.delete_negative_enrollments(),
        Deletion::All => store.delete_all_enrollments(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sinus_core::classify::embed::BandHeuristicEmbedder;
    use sinus_core::pipeline::PipelineConfig;
    use sinus_core::synth;

    fn pipeline() -> StreamingPipeline<BandHeuristicEmbedder> {
        StreamingPipeline::new(PipelineConfig::default(), BandHeuristicEmbedder)
    }

    /// A clean tone loud enough for `loudest_patch` to select, long enough to
    /// cover one full analysis patch.
    fn take_signal() -> Vec<f32> {
        synth::sine(TEACH_TAKE_SAMPLES, SAMPLE_RATE, 300.0, 0.6)
    }

    #[test]
    fn take_buffer_discards_countdown_and_keeps_exactly_one_take() {
        let mut buf = TakeBuffer::new(EventType::Sniffle);
        assert!(!buf.push(&vec![0.1; TEACH_COUNTDOWN_SAMPLES - 1]));
        assert!(buf.samples().is_empty());
        assert!(buf.push(&[0.2, 0.3]));
        assert_eq!(buf.samples(), &[0.3]);

        assert!(!buf.push(&vec![0.4; TEACH_TAKE_SAMPLES + 100]));
        assert!(buf.is_complete());
        assert_eq!(buf.samples().len(), TEACH_TAKE_SAMPLES);
    }

    #[test]
    fn first_take_has_no_baseline_to_score_against() {
        let store = Store::open_in_memory().unwrap();
        let pipeline = pipeline();

        let result = enroll_take(&store, &pipeline, EventType::Hawk, &take_signal()).unwrap();
        assert_eq!(result.examples, 1);
        assert!(result.similarity < 0.0);

        let stored = store.enrollments().unwrap();
        assert_eq!(stored.len(), 1);
        assert!(stored[0].similarity.is_none());
        assert!(stored[0].separation.is_none());
    }

    #[test]
    fn second_take_scores_against_the_first() {
        let store = Store::open_in_memory().unwrap();
        let pipeline = pipeline();

        enroll_take(&store, &pipeline, EventType::Hawk, &take_signal()).unwrap();
        let result = enroll_take(&store, &pipeline, EventType::Hawk, &take_signal()).unwrap();

        assert_eq!(result.examples, 2);
        assert!(result.similarity >= 0.0);

        let stored = store.enrollments().unwrap();
        let second = &stored[1];
        assert_eq!(second.similarity, Some(result.similarity));
        assert_eq!(second.separation, Some(result.separation));
    }

    #[test]
    fn class_status_transitions_as_takes_accumulate() {
        let store = Store::open_in_memory().unwrap();
        let pipeline = pipeline();

        let status_of = |store: &Store| {
            let (classes, _) = training_snapshot(store).unwrap();
            classes
                .into_iter()
                .find(|c| c.class == EventType::Hawk)
                .unwrap()
                .status
        };

        assert_eq!(status_of(&store), ClassStatus::Untrained);

        enroll_take(&store, &pipeline, EventType::Hawk, &take_signal()).unwrap();
        assert_eq!(status_of(&store), ClassStatus::Inactive { needed: 2 });

        enroll_take(&store, &pipeline, EventType::Hawk, &take_signal()).unwrap();
        assert_eq!(status_of(&store), ClassStatus::Inactive { needed: 1 });

        enroll_take(&store, &pipeline, EventType::Hawk, &take_signal()).unwrap();
        // Three identical takes score a near-perfect repeat similarity against
        // each other with no competing class enrolled, so the class reads ready.
        assert_eq!(status_of(&store), ClassStatus::Ready);
    }

    #[test]
    fn deleting_a_class_leaves_others_untouched() {
        let store = Store::open_in_memory().unwrap();
        let pipeline = pipeline();

        enroll_take(&store, &pipeline, EventType::Hawk, &take_signal()).unwrap();
        enroll_take(&store, &pipeline, EventType::Hawk, &take_signal()).unwrap();
        enroll_take(&store, &pipeline, EventType::SnortSuck, &take_signal()).unwrap();

        let removed = delete(&store, Deletion::Class(EventType::Hawk)).unwrap();
        assert_eq!(removed, 2);

        let remaining = store.enrollments().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].enrollment.class, EventType::SnortSuck);
    }
}

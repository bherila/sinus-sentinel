//! Few-shot personalized matcher (SPEC §5 Phase B-lite). The user enrolls their own
//! `hawk` / `nose_blow` / `snort_suck` (and personalized native) examples; each is
//! stored as its 1024-d YAMNet embedding. At runtime, nearest-example cosine
//! similarity preserves multimodal sounds better than one averaged centroid; the
//! winner must clear a threshold and beat competing classes/enrolled negatives.
//! No training loop — updates are instant.

use crate::types::EventType;

/// A personalized class stays inactive until it has enough independent examples
/// to avoid turning one unusual sound (or a bit of speech) into a broad template.
pub const MIN_POSITIVE_EXAMPLES: usize = 3;

/// L2-normalize a vector (returns zeros if the input is all-zero).
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return vec![0.0; v.len()];
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Cosine similarity between two vectors (assumes equal length).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = normalize(a);
    let nb = normalize(b);
    na.iter().zip(&nb).map(|(x, y)| x * y).sum()
}

/// A class prototype: the mean of the enrolled (normalized) embeddings.
#[derive(Debug, Clone)]
pub struct Prototype {
    pub class: EventType,
    pub centroid: Vec<f32>,
    pub examples: usize,
}

/// One enrolled example.
#[derive(Debug, Clone)]
pub struct Enrollment {
    pub class: EventType,
    pub embedding: Vec<f32>,
    pub is_negative: bool,
}

/// The prototype matcher.
#[derive(Debug, Clone)]
pub struct PrototypeMatcher {
    prototypes: Vec<Prototype>,
    /// Normalized positives retained for nearest-example matching. A centroid can
    /// blur a multimodal class, such as a short sniff and a longer inhale.
    positives: Vec<(EventType, Vec<f32>)>,
    /// Normalized negative embeddings.
    negatives: Vec<Vec<f32>>,
    /// Minimum cosine similarity to accept a match.
    pub sim_threshold: f32,
    /// The best-class similarity must beat the nearest negative by this margin.
    /// The same margin is required over the runner-up positive class so similar
    /// nasal sounds do not get forced into an arbitrary enrolled label.
    pub negative_margin: f32,
}

impl PrototypeMatcher {
    /// Build a matcher from enrolled examples. Positive examples are normalized
    /// and retained for nearest-example matching; class centroids are also kept
    /// for counts/inspection. Negatives are stored individually for the
    /// must-beat-negative gate.
    pub fn from_enrollments(
        enrollments: &[Enrollment],
        sim_threshold: f32,
        negative_margin: f32,
    ) -> Self {
        use std::collections::BTreeMap;
        let mut sums: BTreeMap<&'static str, (EventType, Vec<f32>, usize)> = BTreeMap::new();
        let mut positives = Vec::new();
        let mut negatives = Vec::new();

        for e in enrollments {
            if e.is_negative {
                negatives.push(normalize(&e.embedding));
                continue;
            }
            let norm = normalize(&e.embedding);
            positives.push((e.class, norm.clone()));
            let entry = sums
                .entry(e.class.as_str())
                .or_insert_with(|| (e.class, vec![0.0; norm.len()], 0));
            if entry.1.len() != norm.len() {
                // Skip mismatched-dimension enrollments defensively.
                continue;
            }
            for (acc, x) in entry.1.iter_mut().zip(&norm) {
                *acc += x;
            }
            entry.2 += 1;
        }

        let prototypes = sums
            .into_values()
            .map(|(class, sum, count)| Prototype {
                class,
                centroid: normalize(&sum), // mean direction
                examples: count,
            })
            .collect();

        PrototypeMatcher {
            prototypes,
            positives,
            negatives,
            sim_threshold,
            negative_margin,
        }
    }

    pub fn is_empty(&self) -> bool {
        !self
            .prototypes
            .iter()
            .any(|prototype| prototype.examples >= MIN_POSITIVE_EXAMPLES)
    }

    pub fn prototypes(&self) -> &[Prototype] {
        &self.prototypes
    }

    /// Similarity of an embedding to every prototype, in prototype order.
    pub fn similarities(&self, embedding: &[f32]) -> Vec<(EventType, f32)> {
        self.prototypes
            .iter()
            .map(|prototype| {
                let similarity = self
                    .positives
                    .iter()
                    .filter(|(class, _)| *class == prototype.class)
                    .map(|(_, example)| cosine(example, embedding))
                    .fold(-1.0f32, f32::max);
                (prototype.class, similarity)
            })
            .collect()
    }

    /// Cosine similarity to the nearest enrolled negative (−1 if none).
    pub fn nearest_negative(&self, embedding: &[f32]) -> f32 {
        self.negatives
            .iter()
            .map(|n| cosine(n, embedding))
            .fold(-1.0f32, f32::max)
    }

    /// Best matching class for an embedding, applying both gates: the similarity
    /// must clear `sim_threshold` and beat both the nearest negative and runner-up
    /// positive class by `negative_margin`. Returns `None` if nothing qualifies.
    pub fn best_match(&self, embedding: &[f32]) -> Option<(EventType, f32)> {
        let neg = self.nearest_negative(embedding);
        let mut similarities = self.similarities(embedding);
        similarities.retain(|(class, _)| {
            self.prototypes.iter().any(|prototype| {
                prototype.class == *class && prototype.examples >= MIN_POSITIVE_EXAMPLES
            })
        });
        similarities.sort_by(|a, b| b.1.total_cmp(&a.1));
        let &(class, best) = similarities.first()?;
        let runner_up = similarities.get(1).map(|(_, sim)| *sim).unwrap_or(-1.0);
        (best >= self.sim_threshold
            && best >= neg + self.negative_margin
            && best >= runner_up + self.negative_margin)
            .then_some((class, best))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a length-8 embedding from a small seed pattern, so clusters are easy
    /// to reason about.
    fn emb(pattern: [f32; 4]) -> Vec<f32> {
        vec![
            pattern[0], pattern[1], pattern[2], pattern[3], pattern[0], pattern[1], pattern[2],
            pattern[3],
        ]
    }

    fn matcher() -> PrototypeMatcher {
        let enrollments = vec![
            // Hawk cluster ≈ direction [1,0,0,0].
            Enrollment {
                class: EventType::Hawk,
                embedding: emb([1.0, 0.1, 0.0, 0.0]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::Hawk,
                embedding: emb([0.9, 0.0, 0.1, 0.0]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::Hawk,
                embedding: emb([1.0, 0.0, 0.1, 0.0]),
                is_negative: false,
            },
            // Nose-blow cluster ≈ direction [0,0,1,0].
            Enrollment {
                class: EventType::NoseBlow,
                embedding: emb([0.0, 0.1, 1.0, 0.0]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::NoseBlow,
                embedding: emb([0.1, 0.0, 0.9, 0.0]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::NoseBlow,
                embedding: emb([0.0, 0.0, 1.0, 0.1]),
                is_negative: false,
            },
            // Negative (keyboard-like) ≈ direction [0,0,0,1].
            Enrollment {
                class: EventType::Hawk,
                embedding: emb([0.0, 0.0, 0.0, 1.0]),
                is_negative: true,
            },
        ];
        PrototypeMatcher::from_enrollments(&enrollments, 0.6, 0.1)
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn prototype_is_mean_direction_and_counts_examples() {
        let m = matcher();
        let hawk = m.prototypes().iter().find(|p| p.class == EventType::Hawk);
        assert_eq!(hawk.unwrap().examples, 3);
    }

    #[test]
    fn matches_nearest_prototype() {
        let m = matcher();
        let probe = emb([0.95, 0.05, 0.0, 0.0]); // near hawk
        assert_eq!(m.best_match(&probe).unwrap().0, EventType::Hawk);
        let probe = emb([0.0, 0.0, 1.0, 0.0]); // near nose-blow
        assert_eq!(m.best_match(&probe).unwrap().0, EventType::NoseBlow);
    }

    #[test]
    fn rejects_when_below_threshold() {
        let m = matcher();
        // Orthogonal-ish to every prototype.
        let probe = emb([0.0, 1.0, 0.0, 0.0]);
        assert!(m.best_match(&probe).is_none());
    }

    #[test]
    fn rejects_when_a_negative_is_closer() {
        let m = matcher();
        // Close to the negative [0,0,0,1]; must be rejected despite any weak
        // prototype similarity.
        let probe = emb([0.2, 0.0, 0.1, 0.95]);
        assert!(
            m.best_match(&probe).is_none(),
            "should be blocked by the nearest negative"
        );
    }

    #[test]
    fn rejects_ambiguous_positive_classes() {
        let enrollments = [
            (EventType::Hawk, [1.0, 0.0, 0.0, 0.0]),
            (EventType::SnortSuck, [0.98, 0.2, 0.0, 0.0]),
        ]
        .into_iter()
        .flat_map(|(class, pattern)| {
            (0..MIN_POSITIVE_EXAMPLES).map(move |_| Enrollment {
                class,
                embedding: emb(pattern),
                is_negative: false,
            })
        })
        .collect::<Vec<_>>();
        let matcher = PrototypeMatcher::from_enrollments(&enrollments, 0.6, 0.05);
        assert!(matcher.best_match(&emb([1.0, 0.1, 0.0, 0.0])).is_none());
    }

    #[test]
    fn nearest_example_preserves_a_multimodal_class() {
        let enrollments = vec![
            Enrollment {
                class: EventType::Sniffle,
                embedding: emb([1.0, 0.0, 0.0, 0.0]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::Sniffle,
                embedding: emb([0.0, 1.0, 0.0, 0.0]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::Sniffle,
                embedding: emb([0.05, 0.95, 0.0, 0.0]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::Hawk,
                embedding: emb([0.0, 0.0, 1.0, 0.0]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::Hawk,
                embedding: emb([0.0, 0.0, 1.0, 0.05]),
                is_negative: false,
            },
            Enrollment {
                class: EventType::Hawk,
                embedding: emb([0.0, 0.05, 1.0, 0.0]),
                is_negative: false,
            },
        ];
        let matcher = PrototypeMatcher::from_enrollments(&enrollments, 0.8, 0.1);
        assert_eq!(
            matcher.best_match(&emb([0.0, 1.0, 0.0, 0.0])),
            Some((EventType::Sniffle, 1.0))
        );
    }

    #[test]
    fn class_stays_inactive_until_three_positive_examples() {
        let matcher = PrototypeMatcher::from_enrollments(
            &[
                Enrollment {
                    class: EventType::Hawk,
                    embedding: emb([1.0, 0.0, 0.0, 0.0]),
                    is_negative: false,
                },
                Enrollment {
                    class: EventType::Hawk,
                    embedding: emb([0.9, 0.1, 0.0, 0.0]),
                    is_negative: false,
                },
            ],
            0.6,
            0.05,
        );

        assert!(matcher.is_empty());
        assert!(matcher.best_match(&emb([1.0, 0.0, 0.0, 0.0])).is_none());
    }
}

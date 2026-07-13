//! Deterministic synthetic signal generators. Used by unit tests and by the CLI
//! `testdata` corpus builder (SPEC §4.1 accuracy loop). Everything is seeded so
//! the golden corpus is byte-for-byte reproducible; these are *clearly synthetic*
//! placeholders — real recordings come from the user later.

use std::f32::consts::PI;

/// A tiny deterministic xorshift PRNG so the corpus never depends on `rand`'s
/// internals or platform entropy.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in [-1.0, 1.0).
    pub fn next_bipolar(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        v * 2.0 - 1.0
    }
}

/// A sine wave of `n` samples at `freq` Hz and amplitude `amp`.
pub fn sine(n: usize, sample_rate: u32, freq: f32, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

/// White noise of `n` samples with amplitude `amp`, seeded.
pub fn white_noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| amp * rng.next_bipolar()).collect()
}

/// `hops` hops of `hop` samples of a sine wave.
pub fn sine_hops(hops: usize, hop: usize, sample_rate: u32, freq: f32, amp: f32) -> Vec<f32> {
    sine(hops * hop, sample_rate, freq, amp)
}

/// `hops` hops of `hop` samples of white noise.
pub fn white_noise_hops(hops: usize, hop: usize, amp: f32, seed: u64) -> Vec<f32> {
    white_noise(hops * hop, amp, seed)
}

/// `n` samples of digital silence.
pub fn silence(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

/// A raised-cosine-enveloped tone burst — a crude but deterministic stand-in for
/// a transient event (cough/sneeze-like). `dur_s` seconds at `freq` Hz.
pub fn tone_burst(sample_rate: u32, freq: f32, dur_s: f32, amp: f32) -> Vec<f32> {
    let n = (sample_rate as f32 * dur_s) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let env = 0.5 - 0.5 * (2.0 * PI * t).cos(); // Hann envelope
            amp * env * (2.0 * PI * freq * i as f32 / sample_rate as f32).sin()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let a: Vec<f32> = (0..8).map(|_| Rng::new(42).next_bipolar()).collect();
        let b = Rng::new(42).next_bipolar();
        assert_eq!(a[0], b);
        assert!(a.iter().all(|&v| (-1.0..1.0).contains(&v)));
    }

    #[test]
    fn sine_has_expected_amplitude() {
        let s = sine(16_000, 16_000, 440.0, 0.5);
        let peak = s.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.01, "peak = {peak}");
    }

    #[test]
    fn tone_burst_length_matches() {
        let b = tone_burst(16_000, 500.0, 0.3, 0.7);
        assert_eq!(b.len(), 4800);
    }
}

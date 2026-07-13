//! Stage ② — log-mel frontend (SPEC §4.1). This is YAMNet's *exact* recipe and
//! must not be "improved": 16 kHz mono → STFT with a 25 ms periodic-Hann window
//! and 10 ms hop → 64 mel bands spanning 125–7500 Hz → `log(mel + 0.001)`. A
//! classifier patch is 0.96 s = 96 mel frames × 64 bands; patches hop 0.5 s while
//! the gate is open. There is deliberately no per-window normalization / AGC —
//! YAMNet was trained on this scaling and silently changing it wrecks calibration.

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

/// STFT frame length: 25 ms @ 16 kHz.
pub const FRAME_LEN: usize = 400;
/// FFT length: next power of two ≥ frame length (matches `tf.signal.stft`).
pub const FFT_LEN: usize = 512;
/// STFT hop: 10 ms @ 16 kHz.
pub const HOP_LEN: usize = 160;
/// Number of mel bands.
pub const N_MEL: usize = 64;
/// Number of one-sided spectrogram bins.
pub const N_SPEC_BINS: usize = FFT_LEN / 2 + 1;
/// Mel band range (Hz).
pub const MEL_MIN_HZ: f32 = 125.0;
pub const MEL_MAX_HZ: f32 = 7500.0;
/// Stabilizing offset inside the log (SPEC §4.1).
pub const LOG_OFFSET: f32 = 0.001;
/// Frames per classifier patch: 0.96 s.
pub const PATCH_FRAMES: usize = 96;
/// Patch hop in frames: 0.5 s (SPEC §4.1).
pub const PATCH_HOP_FRAMES: usize = 50;

/// HTK-style mel scale, matching `tf.signal.linear_to_mel_weight_matrix`
/// (natural log, break frequency 700 Hz, Q = 1127).
pub fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// A single classifier patch of log-mel features, row-major `[frames][bands]`.
#[derive(Debug, Clone)]
pub struct MelPatch {
    pub frames: usize,
    pub bands: usize,
    /// Length `frames * bands`, row-major.
    pub data: Vec<f32>,
}

impl MelPatch {
    /// Mean log-mel energy across the patch, per band — a cheap summary used by
    /// deterministic test classifiers and calibration.
    pub fn band_means(&self) -> Vec<f32> {
        let mut means = vec![0.0f32; self.bands];
        for frame in self.data.chunks(self.bands) {
            for (m, &v) in means.iter_mut().zip(frame) {
                *m += v;
            }
        }
        for m in &mut means {
            *m /= self.frames as f32;
        }
        means
    }
}

/// Precomputed mel filterbank, shape `[N_MEL][N_SPEC_BINS]`.
fn build_filterbank(sample_rate: u32) -> Vec<[f32; N_SPEC_BINS]> {
    let nyquist = sample_rate as f32 / 2.0;
    // Linear bin center frequencies. tf zeroes the DC bin (bands_to_zero = 1).
    let bin_hz: Vec<f32> = (0..N_SPEC_BINS)
        .map(|i| i as f32 * nyquist / (N_SPEC_BINS - 1) as f32)
        .collect();
    let bin_mel: Vec<f32> = bin_hz.iter().map(|&f| hz_to_mel(f)).collect();

    // N_MEL + 2 mel band edges, linearly spaced in mel.
    let lo = hz_to_mel(MEL_MIN_HZ);
    let hi = hz_to_mel(MEL_MAX_HZ);
    let edges: Vec<f32> = (0..N_MEL + 2)
        .map(|j| lo + (hi - lo) * j as f32 / (N_MEL + 1) as f32)
        .collect();

    let mut fb = vec![[0.0f32; N_SPEC_BINS]; N_MEL];
    for (j, row) in fb.iter_mut().enumerate() {
        let lower = edges[j];
        let center = edges[j + 1];
        let upper = edges[j + 2];
        // Skip the DC bin (i = 0) — left at weight 0.
        for i in 1..N_SPEC_BINS {
            let m = bin_mel[i];
            let lower_slope = (m - lower) / (center - lower);
            let upper_slope = (upper - m) / (upper - center);
            row[i] = lower_slope.min(upper_slope).max(0.0);
        }
    }
    fb
}

/// Periodic Hann window of length `FRAME_LEN` (matches `tf.signal.hann_window`).
fn hann_window() -> [f32; FRAME_LEN] {
    let mut w = [0.0f32; FRAME_LEN];
    for (n, wn) in w.iter_mut().enumerate() {
        *wn = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / FRAME_LEN as f32).cos();
    }
    w
}

/// The log-mel frontend. Holds the FFT plan, window, and filterbank so repeated
/// calls allocate nothing beyond scratch buffers.
pub struct MelFrontend {
    sample_rate: u32,
    window: [f32; FRAME_LEN],
    filterbank: Vec<[f32; N_SPEC_BINS]>,
    r2c: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
}

impl MelFrontend {
    pub fn new(sample_rate: u32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(FFT_LEN);
        MelFrontend {
            sample_rate,
            window: hann_window(),
            filterbank: build_filterbank(sample_rate),
            r2c,
        }
    }

    /// Number of STFT frames a signal of `n` samples produces (no end padding,
    /// matching `tf.signal.stft(pad_end=False)`).
    pub fn num_frames(n: usize) -> usize {
        if n < FRAME_LEN {
            0
        } else {
            (n - FRAME_LEN) / HOP_LEN + 1
        }
    }

    /// Compute log-mel frames for a mono 16 kHz signal. Each frame is `N_MEL`
    /// bands. Returns one `[f32; N_MEL]` per STFT frame.
    pub fn log_mel_frames(&self, samples: &[f32]) -> Vec<[f32; N_MEL]> {
        let n_frames = Self::num_frames(samples.len());
        let mut out = Vec::with_capacity(n_frames);
        let mut indata = self.r2c.make_input_vec();
        let mut spectrum = self.r2c.make_output_vec();
        let mut mag = [0.0f32; N_SPEC_BINS];

        for f in 0..n_frames {
            let start = f * HOP_LEN;
            let frame = &samples[start..start + FRAME_LEN];
            for (i, slot) in indata.iter_mut().enumerate() {
                *slot = if i < FRAME_LEN {
                    frame[i] * self.window[i]
                } else {
                    0.0
                };
            }
            self.r2c
                .process(&mut indata, &mut spectrum)
                .expect("realfft: fixed-size buffers");
            for (m, c) in mag.iter_mut().zip(spectrum.iter()) {
                *m = Complex::norm(*c);
            }
            let mut row = [0.0f32; N_MEL];
            for (j, band) in row.iter_mut().enumerate() {
                let fb = &self.filterbank[j];
                let mut acc = 0.0f32;
                for i in 0..N_SPEC_BINS {
                    acc += mag[i] * fb[i];
                }
                *band = (acc + LOG_OFFSET).ln();
            }
            out.push(row);
        }
        out
    }

    /// Assemble 96-frame patches (0.5 s hop) from a signal. Segments shorter than
    /// one patch yield none.
    pub fn patches(&self, samples: &[f32]) -> Vec<MelPatch> {
        let frames = self.log_mel_frames(samples);
        Self::frames_to_patches(&frames)
    }

    /// Group log-mel frames into 96-frame patches with a 50-frame hop.
    pub fn frames_to_patches(frames: &[[f32; N_MEL]]) -> Vec<MelPatch> {
        let mut patches = Vec::new();
        if frames.len() < PATCH_FRAMES {
            return patches;
        }
        let mut start = 0;
        while start + PATCH_FRAMES <= frames.len() {
            let mut data = Vec::with_capacity(PATCH_FRAMES * N_MEL);
            for row in &frames[start..start + PATCH_FRAMES] {
                data.extend_from_slice(row);
            }
            patches.push(MelPatch {
                frames: PATCH_FRAMES,
                bands: N_MEL,
                data,
            });
            start += PATCH_HOP_FRAMES;
        }
        patches
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    #[test]
    fn hz_to_mel_reference_values() {
        // Hand-computed: 1127 * ln(1 + f/700).
        assert!(
            (hz_to_mel(125.0) - 185.09).abs() < 0.1,
            "{}",
            hz_to_mel(125.0)
        );
        assert!(
            (hz_to_mel(7500.0) - 2773.4).abs() < 1.0,
            "{}",
            hz_to_mel(7500.0)
        );
        assert_eq!(hz_to_mel(0.0), 0.0);
    }

    #[test]
    fn filterbank_shape_and_dc_zero() {
        let fb = build_filterbank(16_000);
        assert_eq!(fb.len(), N_MEL);
        for row in &fb {
            assert_eq!(row[0], 0.0, "DC bin must be zeroed");
        }
    }

    #[test]
    fn each_filter_is_triangular_with_unit_peak() {
        let fb = build_filterbank(16_000);
        for (j, row) in fb.iter().enumerate() {
            let peak = row.iter().cloned().fold(0.0f32, f32::max);
            assert!(
                peak > 0.4 && peak <= 1.0001,
                "filter {j} peak {peak} out of range"
            );
        }
    }

    #[test]
    fn framing_math_matches_tf_stft() {
        // 1 s @ 16 kHz → floor((16000-400)/160)+1 = 98 frames.
        assert_eq!(MelFrontend::num_frames(16_000), 98);
        assert_eq!(MelFrontend::num_frames(399), 0);
        assert_eq!(MelFrontend::num_frames(400), 1);
        assert_eq!(MelFrontend::num_frames(560), 2);
    }

    #[test]
    fn patch_framing() {
        // 98 frames, 96-frame patch, 50-frame hop → exactly 1 patch.
        let frames = vec![[0.0f32; N_MEL]; 98];
        assert_eq!(MelFrontend::frames_to_patches(&frames).len(), 1);
        // 146 frames → 2 patches (starts 0 and 50).
        let frames = vec![[0.0f32; N_MEL]; 146];
        assert_eq!(MelFrontend::frames_to_patches(&frames).len(), 2);
        // Too short → none.
        let frames = vec![[0.0f32; N_MEL]; 95];
        assert_eq!(MelFrontend::frames_to_patches(&frames).len(), 0);
    }

    #[test]
    fn pure_tone_lands_in_the_expected_mel_band() {
        let fe = MelFrontend::new(16_000);
        let tone = synth::sine(16_000, 16_000, 1000.0, 0.8);
        let patches = fe.patches(&tone);
        assert_eq!(patches.len(), 1);
        let means = patches[0].band_means();
        let argmax = means
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // Expected band: the filter whose center is nearest 1000 Hz.
        let lo = hz_to_mel(MEL_MIN_HZ);
        let hi = hz_to_mel(MEL_MAX_HZ);
        let target_mel = hz_to_mel(1000.0);
        let expected = (0..N_MEL)
            .min_by(|&a, &b| {
                let ca = lo + (hi - lo) * (a as f32 + 1.0) / (N_MEL + 1) as f32;
                let cb = lo + (hi - lo) * (b as f32 + 1.0) / (N_MEL + 1) as f32;
                (ca - target_mel)
                    .abs()
                    .partial_cmp(&(cb - target_mel).abs())
                    .unwrap()
            })
            .unwrap();
        assert!(
            (argmax as i32 - expected as i32).abs() <= 2,
            "argmax {argmax}, expected ~{expected}"
        );
    }
}

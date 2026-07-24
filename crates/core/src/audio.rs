//! Audio capture & transport (SPEC §4.1). An [`AudioSource`] yields 16 kHz mono
//! `f32` samples; tests feed WAV/synthetic sources through the *identical*
//! pipeline as the live microphone. The live implementation ([`CpalAudioSource`],
//! behind the `live-audio` feature) opens the raw input via `cpal`, moves samples
//! across a lock-free `ringbuf` SPSC queue from the real-time callback, and
//! resamples to 16 kHz with `rubato` when the device rate differs.

#[cfg(feature = "live-audio")]
use crate::error::Error;
use crate::error::Result;
use crate::types::SAMPLE_RATE;

/// A source of 16 kHz mono `f32` audio. `read` fills `out` and returns the number
/// of samples written; `0` means end-of-stream.
pub trait AudioSource {
    /// Output sample rate — always [`SAMPLE_RATE`] for pipeline sources (they
    /// resample internally).
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize>;
}

/// Cheap linear-interpolation resampler. Used for the WAV path (the live path
/// uses `rubato`, see [`CpalAudioSource`]). Adequate for offline analysis of
/// already-recorded files.
pub fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = input[idx.min(input.len() - 1)];
        let b = input[(idx + 1).min(input.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Downmix interleaved multi-channel samples to mono by averaging channels.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// An [`AudioSource`] backed by in-memory 16 kHz mono samples — WAV files and
/// synthetic test signals.
#[derive(Debug, Clone)]
pub struct BufferedAudioSource {
    samples: Vec<f32>,
    pos: usize,
}

impl BufferedAudioSource {
    /// From samples already at 16 kHz mono.
    pub fn from_samples(samples: Vec<f32>) -> Self {
        BufferedAudioSource { samples, pos: 0 }
    }

    /// Load a WAV file, downmix to mono, and resample to 16 kHz.
    pub fn open_wav(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let channels = spec.channels as usize;

        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / max))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
        };

        let mono = downmix_to_mono(&interleaved, channels);
        let resampled = resample_linear(&mono, spec.sample_rate, SAMPLE_RATE);
        Ok(BufferedAudioSource::from_samples(resampled))
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl AudioSource for BufferedAudioSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let remaining = self.samples.len() - self.pos;
        let n = remaining.min(out.len());
        out[..n].copy_from_slice(&self.samples[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Write mono 16 kHz `f32` samples to a WAV file (used by the CLI corpus builder).
pub fn write_wav_16k_mono(path: impl AsRef<std::path::Path>, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(clamped)?;
    }
    writer.finalize()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Live capture (feature `live-audio`)
// ---------------------------------------------------------------------------

/// A `rubato`-backed resampler to 16 kHz that accepts arbitrary-length input by
/// buffering into fixed chunks (SPEC §4.1 — resample when the device won't open
/// at 16 kHz).
#[cfg(feature = "live-audio")]
pub struct RubatoResampler {
    inner: rubato::FastFixedIn<f32>,
    chunk: usize,
    in_buf: std::collections::VecDeque<f32>,
    input_chunk: Vec<f32>,
    output: Vec<Vec<f32>>,
}

#[cfg(feature = "live-audio")]
impl RubatoResampler {
    pub fn new(from: u32, to: u32) -> Result<Self> {
        use rubato::{FastFixedIn, PolynomialDegree, Resampler};
        let chunk = 1024;
        let ratio = to as f64 / from as f64;
        let inner = FastFixedIn::<f32>::new(ratio, 1.1, PolynomialDegree::Linear, chunk, 1)
            .map_err(|e| Error::Audio(format!("rubato init: {e}")))?;
        let output = inner.output_buffer_allocate(true);
        Ok(RubatoResampler {
            inner,
            chunk,
            in_buf: std::collections::VecDeque::new(),
            input_chunk: vec![0.0; chunk],
            output,
        })
    }

    /// Push input samples; returns any resampled output produced from full chunks.
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        self.push_into(input, &mut output)?;
        Ok(output)
    }

    /// Allocation-reusing live variant. `output` is appended to and may be reused
    /// by the caller across every microphone read.
    pub fn push_into(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<()> {
        use rubato::Resampler;
        self.in_buf.extend(input.iter().copied());
        while self.in_buf.len() >= self.chunk {
            for slot in &mut self.input_chunk {
                *slot = self.in_buf.pop_front().expect("length checked above");
            }
            let (_, written) = self
                .inner
                .process_into_buffer(
                    std::slice::from_ref(&self.input_chunk),
                    &mut self.output,
                    None,
                )
                .map_err(|e| Error::Audio(format!("rubato process: {e}")))?;
            output.extend_from_slice(&self.output[0][..written]);
        }
        Ok(())
    }
}

/// Live microphone capture via `cpal` + `ringbuf` (SPEC §4.1). The real-time
/// callback only downmixes to mono and pushes into the SPSC ring buffer — no
/// allocation, no locks, no inference. `read` drains and resamples to 16 kHz.
#[cfg(feature = "live-audio")]
pub struct CpalAudioSource {
    consumer: ringbuf::HeapCons<f32>,
    resampler: Option<RubatoResampler>,
    pending: std::collections::VecDeque<f32>,
    drained: Vec<f32>,
    resampled: Vec<f32>,
    _stream: cpal::Stream,
}

#[cfg(feature = "live-audio")]
impl CpalAudioSource {
    /// Open the default input device in raw mono mode. Returns [`Error::Audio`] on
    /// any device/stream error — never panics.
    pub fn open_default() -> Result<Self> {
        use cpal::traits::{DeviceTrait, HostTrait};
        use ringbuf::traits::{Producer, Split};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| Error::Audio("no default input device".into()))?;
        let config = device
            .default_input_config()
            .map_err(|e| Error::Audio(format!("default input config: {e}")))?;
        let device_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        // 3 s ring buffer at the device rate (SPEC §4.1).
        let rb = ringbuf::HeapRb::<f32>::new((device_rate as usize) * 3);
        let (mut producer, consumer) = rb.split();
        let capture_thread = std::thread::current();
        let samples_since_wake = Arc::new(AtomicUsize::new(0));
        let callback_wake_count = Arc::clone(&samples_since_wake);
        let wake_after = (device_rate as usize / 20).max(1);

        let err_fn = |e| eprintln!("cpal stream error: {e}");
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Real-time-safe: downmix + push only.
                    let mut frames = 0usize;
                    for frame in data.chunks(channels) {
                        let mono = frame.iter().sum::<f32>() / channels as f32;
                        let _ = producer.try_push(mono);
                        frames += 1;
                    }
                    let accumulated = callback_wake_count.fetch_add(frames, Ordering::Relaxed);
                    if accumulated + frames >= wake_after {
                        callback_wake_count.store(0, Ordering::Relaxed);
                        capture_thread.unpark();
                    }
                },
                err_fn,
                None,
            ),
            other => {
                return Err(Error::Audio(format!(
                    "unsupported sample format {other:?}; open the raw f32 input"
                )))
            }
        }
        .map_err(|e| Error::Audio(format!("build input stream: {e}")))?;

        use cpal::traits::StreamTrait;
        stream
            .play()
            .map_err(|e| Error::Audio(format!("stream play: {e}")))?;

        let resampler = if device_rate == SAMPLE_RATE {
            None
        } else {
            Some(RubatoResampler::new(device_rate, SAMPLE_RATE)?)
        };

        Ok(CpalAudioSource {
            consumer,
            resampler,
            pending: std::collections::VecDeque::new(),
            drained: Vec::with_capacity((device_rate as usize / 20).max(1024)),
            resampled: Vec::with_capacity(SAMPLE_RATE as usize / 20),
            _stream: stream,
        })
    }
}

#[cfg(feature = "live-audio")]
impl AudioSource for CpalAudioSource {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        use ringbuf::traits::Consumer;

        // One gate hop per worker wake keeps latency at 50 ms while avoiding the
        // old 20 ms polling loop. A timeout lets control changes and broken audio
        // devices return to the outer loop even if callbacks stop entirely.
        let target = out.len().min(SAMPLE_RATE as usize / 20).max(1);
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
        loop {
            self.drained.clear();
            while let Some(sample) = self.consumer.try_pop() {
                self.drained.push(sample);
            }
            if let Some(resampler) = &mut self.resampler {
                self.resampled.clear();
                resampler.push_into(&self.drained, &mut self.resampled)?;
                self.pending.extend(self.resampled.iter().copied());
            } else {
                self.pending.extend(self.drained.iter().copied());
            }

            if self.pending.len() >= target || std::time::Instant::now() >= deadline {
                let n = out.len().min(self.pending.len());
                for slot in out.iter_mut().take(n) {
                    *slot = self.pending.pop_front().expect("length checked above");
                }
                return Ok(n);
            }

            std::thread::park_timeout(
                deadline.saturating_duration_since(std::time::Instant::now()),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_resample_changes_length_by_ratio() {
        let input = vec![0.0f32; 48_000]; // 1 s @ 48 kHz
        let out = resample_linear(&input, 48_000, 16_000);
        assert!((out.len() as i64 - 16_000).abs() <= 1);
    }

    #[test]
    fn resample_noop_when_rates_match() {
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_linear(&input, 16_000, 16_000), input);
    }

    #[test]
    fn downmix_averages_channels() {
        let stereo = vec![1.0, 3.0, 2.0, 4.0]; // frames (1,3) and (2,4)
        assert_eq!(downmix_to_mono(&stereo, 2), vec![2.0, 3.0]);
    }

    #[test]
    fn buffered_source_reads_then_eof() {
        let mut src = BufferedAudioSource::from_samples(vec![0.1, 0.2, 0.3]);
        let mut buf = [0.0f32; 2];
        assert_eq!(src.read(&mut buf).unwrap(), 2);
        assert_eq!(buf, [0.1, 0.2]);
        assert_eq!(src.read(&mut buf).unwrap(), 1);
        assert_eq!(src.read(&mut buf).unwrap(), 0); // EOF
    }

    #[test]
    fn wav_roundtrip_through_disk() {
        let dir = std::env::temp_dir().join(format!("sinus-wav-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let samples = crate::synth::sine(16_000, 16_000, 440.0, 0.5);
        write_wav_16k_mono(&path, &samples).unwrap();
        let mut src = BufferedAudioSource::open_wav(&path).unwrap();
        assert_eq!(src.sample_rate(), 16_000);
        let mut buf = vec![0.0f32; 16_000];
        let n = src.read(&mut buf).unwrap();
        assert_eq!(n, 16_000);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "live-audio")]
    #[test]
    fn rubato_resampler_downsamples() {
        let mut r = RubatoResampler::new(48_000, 16_000).unwrap();
        let input = crate::synth::sine(48_000, 48_000, 440.0, 0.5);
        let out = r.push(&input).unwrap();
        // ~1/3 the samples (minus buffering remainder).
        assert!(
            out.len() > 14_000 && out.len() < 17_000,
            "len {}",
            out.len()
        );
    }
}

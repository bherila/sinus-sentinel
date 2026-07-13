//! Headless harness (SPEC §4.1 accuracy loop, §9). Runs the *identical* core
//! pipeline on WAV files and synthetic signals:
//!
//! - `classify <file.wav>` — per-window scores + final sessionized events.
//! - `soak [--secs N]` — quiet-room CPU soak: gate over N s of silence must never
//!   open (SPEC §9).
//! - `calibrate <dir>` — derive per-class thresholds from a labeled corpus.
//! - `gen-testdata <dir>` — write the synthetic golden corpus (clearly synthetic).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sinus_core::audio::{write_wav_16k_mono, AudioSource, BufferedAudioSource};
use sinus_core::classify::embed::BandHeuristicEmbedder;
use sinus_core::gate::{Gate, GateConfig};
use sinus_core::pipeline::{Pipeline, PipelineConfig, PipelineResult};
use sinus_core::synth;
use sinus_core::types::EventType;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    let rest: &[String] = args.get(1..).unwrap_or(&[]);

    let result = match cmd {
        Some("classify") => cmd_classify(rest),
        Some("soak") => cmd_soak(rest),
        Some("calibrate") => cmd_calibrate(rest),
        Some("gen-testdata") => cmd_gen_testdata(rest),
        Some("--help") | Some("-h") | Some("help") | None => {
            print_usage();
            Ok(())
        }
        Some(other) => Err(format!("unknown command: {other}")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!(
        "sinus-cli (core {})\n\n\
         USAGE:\n  \
         sinus-cli classify <file.wav>\n  \
         sinus-cli soak [--secs N]\n  \
         sinus-cli calibrate <dir>\n  \
         sinus-cli gen-testdata <dir>\n",
        sinus_core::CORE_VERSION
    );
}

/// Build the offline pipeline with the deterministic, model-free backbone. (With
/// the `onnx` feature and a model file, swap in `YamnetOnnx` — see model/README.)
fn build_pipeline() -> Pipeline<BandHeuristicEmbedder> {
    Pipeline::new(PipelineConfig::default(), BandHeuristicEmbedder)
}

fn run_wav(path: &Path) -> Result<PipelineResult, String> {
    let mut src = BufferedAudioSource::open_wav(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let mut samples = Vec::new();
    let mut buf = vec![0.0f32; 16_000];
    loop {
        let n = src.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        samples.extend_from_slice(&buf[..n]);
    }
    build_pipeline()
        .process(&samples)
        .map_err(|e| e.to_string())
}

fn cmd_classify(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("usage: classify <file.wav>")?;
    let path = PathBuf::from(path);
    let result = run_wav(&path)?;

    println!("== per-window scores ({}) ==", path.display());
    for w in &result.windows {
        if !w.active {
            continue;
        }
        let top: Vec<String> = w
            .scores
            .iter()
            .take(3)
            .map(|(et, s)| format!("{}={:.2}", et.as_str(), s))
            .collect();
        let hit = w
            .hit
            .map(|h| format!(" -> {}", h.event_type.as_str()))
            .unwrap_or_default();
        println!(
            "  t={:>6}ms peak={} speech={:.2} [{}]{}",
            w.time_ms,
            if w.energy_peak { "Y" } else { "n" },
            w.speech,
            top.join(" "),
            hit
        );
    }

    println!("== events ==");
    if result.events.is_empty() {
        println!("  (none)");
    }
    for e in &result.events {
        println!(
            "  {:<15} start={:>6}ms dur={:>5}ms conf={:.2} bursts={}",
            e.event_type.as_str(),
            e.start_ms,
            e.duration_ms,
            e.confidence,
            e.burst_count
        );
    }
    Ok(())
}

fn cmd_soak(args: &[String]) -> Result<(), String> {
    let mut secs = 600u64;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--secs" {
            secs = args
                .get(i + 1)
                .and_then(|s| s.parse().ok())
                .ok_or("--secs needs a number")?;
            i += 2;
        } else {
            return Err(format!("unexpected arg: {}", args[i]));
        }
    }

    let cfg = GateConfig::default();
    let hop = cfg.hop_samples();
    let mut gate = Gate::new(cfg.clone());
    // Very-low-level noise stands in for a "quiet room" (not digital-zero).
    let silence_hop = synth::white_noise(hop, 0.0005, 12345);

    let total_hops = (secs * cfg.sample_rate as u64) / hop as u64;
    let start = std::time::Instant::now();
    let mut opens = 0u64;
    for _ in 0..total_hops {
        let r = gate.process_hop(&silence_hop);
        if r.open {
            opens += 1;
        }
    }
    let elapsed = start.elapsed();

    println!("soak: {secs}s silence, {total_hops} hops");
    println!("  gate opens: {opens} (want 0)");
    println!(
        "  processing wall time: {:.3}s ({:.0} hops/s)",
        elapsed.as_secs_f64(),
        total_hops as f64 / elapsed.as_secs_f64().max(1e-9)
    );
    if opens > 0 {
        return Err(format!(
            "gate opened {opens} times on silence — quiet-room budget violated"
        ));
    }
    println!("  OK: gate stayed closed the entire soak");
    Ok(())
}

/// Extract the leading class label from a filename like `cough_01.wav` or
/// `throat_clearing.wav`.
fn label_from_filename(path: &Path) -> Option<EventType> {
    let stem = path.file_stem()?.to_str()?;
    for et in EventType::ALL {
        if stem == et.as_str() || stem.starts_with(&format!("{}_", et.as_str())) {
            return Some(et);
        }
    }
    None
}

fn cmd_calibrate(args: &[String]) -> Result<(), String> {
    let dir = PathBuf::from(args.first().ok_or("usage: calibrate <dir>")?);
    let mut positives: BTreeMap<&'static str, Vec<f32>> = BTreeMap::new();

    let entries = std::fs::read_dir(&dir).map_err(|e| format!("read {dir:?}: {e}"))?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("wav") {
            continue;
        }
        let Some(label) = label_from_filename(&path) else {
            continue;
        };
        let result = run_wav(&path)?;
        // Highest confidence among events of the labeled class in this file.
        let conf = result
            .events
            .iter()
            .filter(|e| e.event_type == label)
            .map(|e| e.confidence)
            .fold(0.0f32, f32::max);
        positives.entry(label.as_str()).or_default().push(conf);
    }

    if positives.is_empty() {
        return Err(format!(
            "no labeled *.wav files in {dir:?} (name them e.g. cough_01.wav)"
        ));
    }

    // Suggest θ_c = 70% of the weakest true positive, floored at 0.1.
    let mut thresholds: BTreeMap<String, f32> = BTreeMap::new();
    for (class, confs) in &positives {
        let min = confs.iter().cloned().fold(f32::INFINITY, f32::min);
        let thr = (min * 0.7).max(0.1);
        thresholds.insert(class.to_string(), (thr * 100.0).round() / 100.0);
    }

    println!("{}", serde_json::to_string_pretty(&thresholds).unwrap());
    Ok(())
}

fn cmd_gen_testdata(args: &[String]) -> Result<(), String> {
    let dir = PathBuf::from(args.first().ok_or("usage: gen-testdata <dir>")?);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let files = generate_corpus(&dir)?;
    for f in &files {
        println!("wrote {}", f.display());
    }
    Ok(())
}

/// Write the synthetic golden corpus. Positive tone bursts map deterministically
/// to classes via [`BandHeuristicEmbedder`]; quiet negatives keep the gate shut.
/// Clearly synthetic — real recordings come from the user later.
pub fn generate_corpus(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let sr = sinus_core::types::SAMPLE_RATE;
    let mut written = Vec::new();
    let mut write = |name: &str, samples: &[f32]| -> Result<(), String> {
        let path = dir.join(name);
        write_wav_16k_mono(&path, samples).map_err(|e| e.to_string())?;
        written.push(path);
        Ok(())
    };

    // Positives: ~1 s quiet, 1.5 s loud tone in the class band, ~1 s quiet.
    let positives = [
        ("cough.wav", 300.0f32),
        ("throat_clearing.wav", 700.0),
        ("sneeze.wav", 1500.0),
        ("sniffle.wav", 4500.0),
    ];
    for (name, freq) in positives {
        let mut sig = synth::white_noise(sr as usize, 0.003, 1);
        sig.extend(synth::sine((sr as usize) * 3 / 2, sr, freq, 0.6));
        sig.extend(synth::white_noise(sr as usize, 0.003, 2));
        write(name, &sig)?;
    }

    // Quiet negatives: gate should never open → zero events.
    write("silence.wav", &synth::silence((sr as usize) * 2))?;
    write(
        "quiet_room.wav",
        &synth::white_noise((sr as usize) * 2, 0.0006, 9),
    )?;

    // Hard negative (loud broadband) — kept for real-model precision scoring; the
    // deterministic backbone can't reject it (that is YAMNet's job, plus enrolled
    // negatives + the speech guard, covered by unit tests).
    let mut noise = synth::white_noise(sr as usize, 0.003, 3);
    noise.extend(synth::white_noise(sr as usize, 0.5, 4));
    noise.extend(synth::white_noise(sr as usize, 0.003, 5));
    write("noise_burst.wav", &noise)?;

    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_parse_from_filenames() {
        assert_eq!(
            label_from_filename(Path::new("cough_01.wav")),
            Some(EventType::Cough)
        );
        assert_eq!(
            label_from_filename(Path::new("throat_clearing.wav")),
            Some(EventType::ThroatClearing)
        );
        assert_eq!(
            label_from_filename(Path::new("nose_blow_3.wav")),
            Some(EventType::NoseBlow)
        );
        assert_eq!(label_from_filename(Path::new("random.wav")), None);
    }
}

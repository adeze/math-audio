//! Waterfall decay grids + 60 ms resonance picking (report primitive M4).
//!
//! Feeds roomeq report section R4: STFT decay slices over −5…500 ms plus
//! prominent-resonance detection at the 60 ms slice with per-resonance
//! decay time.
//!
//! Conventions (stated next to the code per the cross-cutting requirement):
//! - Window: Hann, [`WATERFALL_WINDOW_MS`] = 32 ms, hop [`WATERFALL_HOP_MS`]
//!   = 2 ms (~94 % overlap), symmetric Hann weights. Frames are centred;
//!   frame time 0 = the direct reference (argmax `|ir|`, or caller-supplied
//!   index via [`waterfall_grid_at`]).
//! - Span: [`WATERFALL_PRE_MS`] = −5 ms … [`WATERFALL_POST_MS`] = 500 ms.
//! - Magnitudes in dB rel. the grid peak, floored at [`WATERFALL_FLOOR_DB`]
//!   = −100 dB.
//! - Decimation for HTML-sized grids: [`WaterfallConfig::max_frames`] /
//!   `max_bins` cap the grid; excess frames/bins are max-pooled in the
//!   linear-power domain (peaks survive, unlike averaging), with time and
//!   frequency labels tracking the argmax cell (not the pool mean).
//! - Resonance picking at the 60 ms slice ([`RESONANCE_SLICE_MS`]):
//!   local maxima above [`WaterfallConfig::threshold_db`] rel. the slice
//!   peak, minimum separation [`WaterfallConfig::min_separation_hz`]
//!   (greedy, loudest first). Per-resonance decay time comes from a
//!   least-squares line fit of that bin's dB decay over
//!   [`DECAY_FIT_FROM_MS`]…[`DECAY_FIT_TO_MS`]; `T60 = −60 / slope`.

use rustfft::{FftPlanner, num_complex::Complex};

/// STFT analysis window length (ms): 32 ms resolves room modes down to
/// ~30 Hz (an 8 ms window is shorter than one period at 110 Hz, so bass
/// content leaks into the DC bin and wins every slice). Time localisation
/// still comes from the 2 ms hop.
pub const WATERFALL_WINDOW_MS: f64 = 32.0;

/// STFT hop between frames (ms).
pub const WATERFALL_HOP_MS: f64 = 2.0;

/// Waterfall start rel. direct (ms, negative = pre-direct).
pub const WATERFALL_PRE_MS: f64 = -5.0;

/// Waterfall end rel. direct (ms).
pub const WATERFALL_POST_MS: f64 = 500.0;

/// dB floor rel. the grid peak.
pub const WATERFALL_FLOOR_DB: f64 = -100.0;

/// Slice time for resonance picking (ms after direct).
pub const RESONANCE_SLICE_MS: f64 = 60.0;

/// Decay-fit window start (ms after direct).
pub const DECAY_FIT_FROM_MS: f64 = 20.0;

/// Decay-fit window end (ms after direct).
pub const DECAY_FIT_TO_MS: f64 = 200.0;

/// Default resonance threshold rel. the 60 ms slice peak (dB).
pub const DEFAULT_RESONANCE_THRESHOLD_DB: f64 = 12.0;

/// Default minimum resonance separation (Hz).
pub const DEFAULT_RESONANCE_SEPARATION_HZ: f64 = 25.0;

/// Default grid caps for HTML-sized output.
pub const DEFAULT_MAX_FRAMES: usize = 256;
/// Default grid caps for HTML-sized output.
pub const DEFAULT_MAX_BINS: usize = 128;

/// Configuration for [`waterfall_grid_at`] / [`detect_resonances`].
#[derive(Debug, Clone, Copy)]
pub struct WaterfallConfig {
    /// Max frames after decimation (max-pooling).
    pub max_frames: usize,
    /// Max frequency bins after decimation (max-pooling).
    pub max_bins: usize,
    /// Resonance threshold rel. slice peak (dB).
    pub threshold_db: f64,
    /// Minimum resonance separation (Hz).
    pub min_separation_hz: f64,
}

impl Default for WaterfallConfig {
    fn default() -> Self {
        Self {
            max_frames: DEFAULT_MAX_FRAMES,
            max_bins: DEFAULT_MAX_BINS,
            threshold_db: DEFAULT_RESONANCE_THRESHOLD_DB,
            min_separation_hz: DEFAULT_RESONANCE_SEPARATION_HZ,
        }
    }
}

/// Decimated STFT decay grid. `mags_db[frame][bin]`, dB rel. grid peak.
#[derive(Debug, Clone)]
pub struct WaterfallGrid {
    /// Frame centre times rel. direct (ms).
    pub times_ms: Vec<f64>,
    /// Bin centre frequencies (Hz).
    pub freqs_hz: Vec<f64>,
    /// Magnitudes, dB rel. grid peak, floored.
    pub mags_db: Vec<Vec<f32>>,
}

/// Analysis settings for a shared spectrogram/waterfall grid without max-pooling.
///
/// The default 128 ms Hann window has a nominal resolution of 7.8125 Hz;
/// zero-padding to a power of two only interpolates its spectrum. Centred
/// windows use zero-padding at the recording edges, but frame centres never
/// extend outside the saved impulse response. Times are relative to `direct_sample`.
#[derive(Debug, Clone, Copy)]
pub struct WaterfallAnalysisConfig {
    pub window_ms: f64,
    pub hop_ms: f64,
    pub start_ms: f64,
    pub end_ms: f64,
    /// Requested output budget, capped at four million cells before allocation.
    pub max_cells: usize,
}

impl Default for WaterfallAnalysisConfig {
    fn default() -> Self {
        Self {
            window_ms: 128.0,
            hop_ms: 4.0,
            start_ms: 0.0,
            end_ms: 500.0,
            max_cells: 4_000_000,
        }
    }
}

/// Shared view limits; changing these does not rerun or renormalize analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedDisplayControls {
    pub min_hz: f64,
    pub max_hz: f64,
    pub start_ms: f64,
    pub end_ms: f64,
    pub floor_db: f64,
    pub ceiling_db: f64,
}

/// Invalid input, unsupported analysis size, or an empty display selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaterfallError(pub &'static str);

impl std::fmt::Display for WaterfallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for WaterfallError {}

impl SharedDisplayControls {
    pub fn validate(&self) -> Result<(), WaterfallError> {
        if ![
            self.min_hz,
            self.max_hz,
            self.start_ms,
            self.end_ms,
            self.floor_db,
            self.ceiling_db,
        ]
        .iter()
        .all(|v| v.is_finite())
            || self.min_hz < 0.0
            || self.max_hz <= self.min_hz
            || self.end_ms <= self.start_ms
            || self.ceiling_db <= self.floor_db
        {
            return Err(WaterfallError("invalid frequency, time, or level limits"));
        }
        Ok(())
    }

    /// Select the same rectangular grid for both views. Magnitudes retain the
    /// analysis-wide reference; the renderer applies floor/ceiling as display limits.
    pub fn crop(&self, grid: &WaterfallGrid) -> Result<WaterfallGrid, WaterfallError> {
        self.validate()?;
        let ordered = |axis: &[f64]| {
            !axis.is_empty()
                && axis.iter().all(|v| v.is_finite())
                && axis.windows(2).all(|p| p[0] < p[1])
        };
        if !ordered(&grid.times_ms)
            || !ordered(&grid.freqs_hz)
            || grid.freqs_hz[0] < 0.0
            || grid.mags_db.len() != grid.times_ms.len()
            || grid
                .mags_db
                .iter()
                .any(|row| row.len() != grid.freqs_hz.len() || row.iter().any(|v| !v.is_finite()))
        {
            return Err(WaterfallError("invalid waterfall grid axes or magnitudes"));
        }
        let ts = grid.times_ms.partition_point(|&t| t < self.start_ms);
        let te = grid.times_ms.partition_point(|&t| t <= self.end_ms);
        let fs = grid.freqs_hz.partition_point(|&f| f < self.min_hz);
        let fe = grid.freqs_hz.partition_point(|&f| f <= self.max_hz);
        if ts == te || fs == fe {
            return Err(WaterfallError("display limits contain no analysis samples"));
        }
        Ok(WaterfallGrid {
            times_ms: grid.times_ms[ts..te].to_vec(),
            freqs_hz: grid.freqs_hz[fs..fe].to_vec(),
            mags_db: grid.mags_db[ts..te]
                .iter()
                .map(|row| row[fs..fe].to_vec())
                .collect(),
        })
    }
}

/// Checked STFT for interactive decay views. All cells share one peak reference
/// from the complete requested analysis, before display cropping: these are
/// relative spectral dB, not calibrated SPL or a noise-compensated decay estimate.
/// Existing report primitives retain their legacy pooling and defaults.
pub fn analyze_waterfall_at(
    ir: &[f32],
    sample_rate: f64,
    direct_sample: usize,
    config: &WaterfallAnalysisConfig,
) -> Result<WaterfallGrid, WaterfallError> {
    if ir.is_empty()
        || ir.len() > 16_777_216
        || direct_sample >= ir.len()
        || ir.iter().any(|v| !v.is_finite())
        || !sample_rate.is_finite()
        || sample_rate <= 0.0
        || ![
            config.window_ms,
            config.hop_ms,
            config.start_ms,
            config.end_ms,
        ]
        .iter()
        .all(|v| v.is_finite())
        || config.window_ms <= 0.0
        || config.hop_ms <= 0.0
        || config.end_ms <= config.start_ms
    {
        return Err(WaterfallError(
            "invalid impulse response or analysis settings",
        ));
    }
    let win_samples = (config.window_ms * sample_rate / 1000.0).round();
    let hop_samples = (config.hop_ms * sample_rate / 1000.0).round();
    if !(4.0..=262_144.0).contains(&win_samples) || !(1.0..=16_777_216.0).contains(&hop_samples) {
        return Err(WaterfallError(
            "analysis window or hop exceeds supported sample range",
        ));
    }
    let window_len = win_samples as usize;
    let hop = hop_samples as usize;
    let fft_len = window_len.next_power_of_two();
    let first = (direct_sample as f64 + config.start_ms * sample_rate / 1000.0)
        .ceil()
        .max(0.0);
    let last = (direct_sample as f64 + config.end_ms * sample_rate / 1000.0)
        .floor()
        .min((ir.len() - 1) as f64);
    if !first.is_finite() || !last.is_finite() || first > last {
        return Err(WaterfallError(
            "analysis time span does not overlap the impulse response",
        ));
    }
    let first = first as usize;
    let frame_count = (last as usize - first) / hop + 1;
    let bin_count = fft_len / 2 + 1;
    if frame_count
        .checked_mul(bin_count)
        .is_none_or(|cells| cells > config.max_cells.min(4_000_000))
    {
        return Err(WaterfallError(
            "analysis grid exceeds cell budget; increase the hop or shorten the span",
        ));
    }
    let window = hann_symmetric(window_len);
    let fft = FftPlanner::new().plan_fft_forward(fft_len);
    let mut buf = vec![Complex::new(0.0, 0.0); fft_len];
    let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    let mut mags_db = Vec::with_capacity(frame_count);
    let mut times_ms = Vec::with_capacity(frame_count);
    let mut peak = 0.0_f64;
    for frame in 0..frame_count {
        let centre = first + frame * hop;
        let start = centre as i64 - (window_len / 2) as i64;
        buf.fill(Complex::new(0.0, 0.0));
        for (i, &weight) in window.iter().enumerate() {
            let index = start + i as i64;
            if index >= 0 && (index as usize) < ir.len() {
                buf[i].re = f64::from(ir[index as usize]) * weight;
            }
        }
        fft.process_with_scratch(&mut buf, &mut scratch);
        let row: Vec<f32> = buf[..bin_count]
            .iter()
            .map(|v| {
                let magnitude = v.norm();
                peak = peak.max(magnitude);
                if magnitude > 0.0 {
                    (20.0 * magnitude.log10()) as f32
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        mags_db.push(row);
        times_ms.push((centre as f64 - direct_sample as f64) * 1000.0 / sample_rate);
    }
    let reference_db = if peak > 0.0 {
        (20.0 * peak.log10()) as f32
    } else {
        0.0
    };
    for row in &mut mags_db {
        for value in row {
            *value = (*value - reference_db).max(WATERFALL_FLOOR_DB as f32);
        }
    }
    Ok(WaterfallGrid {
        times_ms,
        freqs_hz: (0..bin_count)
            .map(|bin| bin as f64 * sample_rate / fft_len as f64)
            .collect(),
        mags_db,
    })
}

#[cfg(test)]
mod shared_decay_tests {
    use super::*;

    #[test]
    fn checked_grid_resolves_decaying_tone_and_silence_with_uniform_axes() {
        let sr = 8000.0;
        let ir: Vec<f32> = (0..4000)
            .map(|i| {
                let t = i as f64 / sr;
                ((2.0 * std::f64::consts::PI * 250.0 * t).sin() * (-8.0 * t).exp()) as f32
            })
            .collect();
        let grid = analyze_waterfall_at(&ir, sr, 0, &WaterfallAnalysisConfig::default()).unwrap();
        assert!(*grid.times_ms.last().unwrap() <= 499.875);
        assert!(
            grid.times_ms
                .windows(2)
                .all(|w| (w[1] - w[0] - 4.0).abs() < 1e-10)
        );
        assert!(
            grid.freqs_hz
                .windows(2)
                .all(|w| (w[1] - w[0] - 7.8125).abs() < 1e-10)
        );
        let bin = grid.freqs_hz.iter().position(|&f| f == 250.0).unwrap();
        let late = grid.times_ms.iter().position(|&t| t == 300.0).unwrap();
        let early = grid.times_ms.iter().position(|&t| t == 100.0).unwrap();
        assert!(
            (f64::from(grid.mags_db[late][bin] - grid.mags_db[early][bin]) + 13.8974).abs() < 0.01
        );
        let silence =
            analyze_waterfall_at(&vec![0.0; 1000], sr, 0, &WaterfallAnalysisConfig::default())
                .unwrap();
        assert!(silence.mags_db.iter().flatten().all(|&db| db == -100.0));
    }

    #[test]
    fn checked_grid_rejects_invalid_inputs_and_preflights_budget() {
        let config = WaterfallAnalysisConfig::default();
        assert!(analyze_waterfall_at(&[], 48000.0, 0, &config).is_err());
        assert!(analyze_waterfall_at(&[f32::NAN], 48000.0, 0, &config).is_err());
        assert!(analyze_waterfall_at(&[0.0], f64::INFINITY, 0, &config).is_err());
        assert!(analyze_waterfall_at(&[0.0], 48000.0, 1, &config).is_err());
        for invalid in [
            WaterfallAnalysisConfig {
                window_ms: f64::MAX,
                ..config
            },
            WaterfallAnalysisConfig {
                hop_ms: 0.0,
                ..config
            },
            WaterfallAnalysisConfig {
                end_ms: -1.0,
                ..config
            },
            WaterfallAnalysisConfig {
                start_ms: 1000.0,
                end_ms: 1500.0,
                ..config
            },
            WaterfallAnalysisConfig {
                max_cells: 1,
                ..config
            },
        ] {
            assert!(analyze_waterfall_at(&[0.0; 1000], 8000.0, 0, &invalid).is_err());
        }
    }

    #[test]
    fn linked_display_crop_preserves_reference_and_validates_axes() {
        let grid = WaterfallGrid {
            times_ms: vec![0.0, 4.0, 8.0],
            freqs_hz: vec![0.0, 100.0, 200.0, 300.0],
            mags_db: vec![vec![0.0, -2.0, -4.0, -8.0], vec![-10.0; 4], vec![-20.0; 4]],
        };
        let controls = SharedDisplayControls {
            min_hz: 100.0,
            max_hz: 200.0,
            start_ms: 4.0,
            end_ms: 8.0,
            floor_db: -60.0,
            ceiling_db: 0.0,
        };
        let cropped = controls.crop(&grid).unwrap();
        assert_eq!(cropped.times_ms, [4.0, 8.0]);
        assert_eq!(cropped.freqs_hz, [100.0, 200.0]);
        assert_eq!(cropped.mags_db, [vec![-10.0; 2], vec![-20.0; 2]]);
        assert!(
            SharedDisplayControls {
                min_hz: f64::NAN,
                ..controls
            }
            .crop(&grid)
            .is_err()
        );
        assert!(
            SharedDisplayControls {
                min_hz: 400.0,
                max_hz: 500.0,
                ..controls
            }
            .crop(&grid)
            .is_err()
        );
        let malformed = WaterfallGrid {
            mags_db: vec![vec![0.0]],
            ..grid
        };
        assert!(controls.crop(&malformed).is_err());
    }
}

/// One prominent resonance at the 60 ms slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resonance {
    /// Frequency (Hz).
    pub freq_hz: f64,
    /// Level at the 60 ms slice (dB rel. grid peak).
    pub level_db: f64,
    /// Decay time from the bin's 20…200 ms fit (s); NaN if unfittable.
    pub decay_time_s: f64,
}

fn hann_symmetric(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let d = (n - 1) as f64;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / d).cos()))
        .collect()
}

/// Build the waterfall grid with an automatic direct reference (argmax `|ir|`).
pub fn waterfall_grid(ir: &[f32], sample_rate: f64, config: &WaterfallConfig) -> WaterfallGrid {
    let direct = ir
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(i, _)| i);
    waterfall_grid_at(ir, sample_rate, direct, config)
}

/// Build the waterfall grid around an explicit direct sample.
#[allow(clippy::cast_precision_loss)]
pub fn waterfall_grid_at(
    ir: &[f32],
    sample_rate: f64,
    direct_sample: usize,
    config: &WaterfallConfig,
) -> WaterfallGrid {
    let empty = WaterfallGrid {
        times_ms: Vec::new(),
        freqs_hz: Vec::new(),
        mags_db: Vec::new(),
    };
    if ir.is_empty() || sample_rate <= 0.0 || direct_sample >= ir.len() {
        return empty;
    }
    let win_n = ((WATERFALL_WINDOW_MS * sample_rate / 1000.0).round() as usize).max(16);
    let hop_n = ((WATERFALL_HOP_MS * sample_rate / 1000.0).round() as usize).max(1);
    let n_fft = win_n.next_power_of_two();
    let window = hann_symmetric(win_n);

    let t0 = direct_sample as f64 + WATERFALL_PRE_MS * sample_rate / 1000.0;
    let t1 = direct_sample as f64 + WATERFALL_POST_MS * sample_rate / 1000.0;
    if t1 <= 0.0 {
        return empty;
    }
    // Frame centres from max(t0, 0): the first frame is centred exactly at
    // the span start (zero-padded where it overhangs the data) so the grid
    // honestly covers −5 ms.
    let mut centres: Vec<f64> = Vec::new();
    let mut c = t0.max(0.0);
    if c > t1 {
        return empty;
    }
    while c <= t1 {
        centres.push(c);
        c += hop_n as f64;
    }
    if centres.is_empty() {
        return empty;
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let half = n_fft / 2;
    let df = sample_rate / n_fft as f64;
    let mut frames: Vec<Vec<f64>> = Vec::with_capacity(centres.len());
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    for &ctr in &centres {
        let start = (ctr - win_n as f64 / 2.0).round() as i64;
        for v in buf.iter_mut() {
            *v = Complex::new(0.0, 0.0);
        }
        for (i, &w) in window.iter().enumerate() {
            let idx = start + i as i64;
            if idx >= 0 && (idx as usize) < ir.len() {
                buf[i].re = f64::from(ir[idx as usize]) * w;
            }
        }
        fft.process(&mut buf);
        frames.push((0..=half).map(|k| buf[k].norm_sqr()).collect());
    }

    // dB rel. grid peak.
    let peak = frames
        .iter()
        .flat_map(|f| f.iter())
        .copied()
        .fold(0.0f64, f64::max);
    let to_db = |p: f64| {
        if peak > 0.0 && p > 0.0 {
            (10.0 * (p / peak).log10()).max(WATERFALL_FLOOR_DB)
        } else {
            WATERFALL_FLOOR_DB
        }
    };

    // Decimate (max-pool in power domain) to the HTML-sized caps.
    let nf = centres.len().min(config.max_frames.max(1));
    let nb = (half + 1).min(config.max_bins.max(1));
    let frame_stride = (centres.len() as f64 / nf as f64).max(1.0);
    let bin_stride = ((half + 1) as f64 / nb as f64).max(1.0);
    let mut times_ms = Vec::with_capacity(nf);
    let mut freqs_hz = Vec::with_capacity(nb);
    let mut mags_db = Vec::with_capacity(nf);
    for fi in 0..nf {
        let f0 = (fi as f64 * frame_stride) as usize;
        let f1 = (((fi + 1) as f64 * frame_stride) as usize).min(centres.len());
        let mut row = Vec::with_capacity(nb);
        for bi in 0..nb {
            let b0 = (bi as f64 * bin_stride) as usize;
            let b1 = (((bi + 1) as f64 * bin_stride) as usize).min(half + 1);
            let mut pmax = 0.0f64;
            let mut fbest = b0.min(half);
            for b in b0..b1.max(b0 + 1) {
                let bb = b.min(half);
                for f in f0..f1.max(f0 + 1) {
                    let ff = f.min(centres.len() - 1);
                    if frames[ff][bb] > pmax {
                        pmax = frames[ff][bb];
                        fbest = bb;
                    }
                }
            }
            if fi == 0 {
                // Frequency label = argmax bin (not the pool mean, which
                // mislabels the peak — e.g. 46.9 Hz for a 110 Hz mode).
                freqs_hz.push(fbest as f64 * df);
            }
            row.push(to_db(pmax) as f32);
        }
        // Time label = centre of the frame-group peak across all bins
        // (magnitudes are max-pooled, so a mean label would smear onsets).
        let mut tlabel = centres[f0.min(centres.len() - 1)];
        let mut pbest = 0.0f64;
        let f1c = f1.max(f0 + 1);
        for (fr, &tc) in frames
            .iter()
            .zip(centres.iter())
            .skip(f0)
            .take(f1c.saturating_sub(f0))
        {
            for &p in fr {
                if p > pbest {
                    pbest = p;
                    tlabel = tc;
                }
            }
        }
        times_ms.push((tlabel - direct_sample as f64) * 1000.0 / sample_rate);
        mags_db.push(row);
    }

    WaterfallGrid {
        times_ms,
        freqs_hz,
        mags_db,
    }
}

/// Detect prominent resonances at the 60 ms slice with per-resonance decay.
///
/// `grid` is typically built by [`waterfall_grid`] from the same IR. The
/// slice nearest [`RESONANCE_SLICE_MS`] is peak-picked; each candidate's
/// decay time comes from a least-squares fit of its bin over
/// [`DECAY_FIT_FROM_MS`]…[`DECAY_FIT_TO_MS`].
pub fn detect_resonances(grid: &WaterfallGrid, config: &WaterfallConfig) -> Vec<Resonance> {
    if grid.times_ms.is_empty() || grid.freqs_hz.is_empty() {
        return Vec::new();
    }
    // Nearest frame to the 60 ms slice.
    let slice = grid
        .times_ms
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (*a.1 - RESONANCE_SLICE_MS)
                .abs()
                .total_cmp(&(*b.1 - RESONANCE_SLICE_MS).abs())
        })
        .map_or(0, |(i, _)| i);
    let row = &grid.mags_db[slice];
    let slice_peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !slice_peak.is_finite() {
        return Vec::new();
    }
    let floor = slice_peak - config.threshold_db as f32;
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for (b, &v) in row.iter().enumerate() {
        if v < floor {
            continue;
        }
        let left = if b > 0 { row[b - 1] } else { f32::NEG_INFINITY };
        let right = if b + 1 < row.len() {
            row[b + 1]
        } else {
            f32::NEG_INFINITY
        };
        if v >= left && v > right {
            candidates.push((b, v));
        }
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut accepted: Vec<(usize, f32)> = Vec::new();
    for (b, v) in candidates {
        let f = grid.freqs_hz[b.min(grid.freqs_hz.len() - 1)];
        if accepted.iter().all(|&(ab, _)| {
            (grid.freqs_hz[ab.min(grid.freqs_hz.len() - 1)] - f).abs() >= config.min_separation_hz
        }) {
            accepted.push((b, v));
        }
    }

    // Fit window frames for the decay slope.
    let fit_frames: Vec<usize> = grid
        .times_ms
        .iter()
        .enumerate()
        .filter(|(_, t)| **t >= DECAY_FIT_FROM_MS && **t <= DECAY_FIT_TO_MS)
        .map(|(i, _)| i)
        .collect();

    let mut out: Vec<Resonance> = accepted
        .into_iter()
        .map(|(b, v)| {
            let freq_hz = grid.freqs_hz[b.min(grid.freqs_hz.len() - 1)];
            let mut xs: Vec<f64> = Vec::new();
            let mut ys: Vec<f64> = Vec::new();
            for &f in &fit_frames {
                let m = grid.mags_db[f][b.min(grid.mags_db[f].len() - 1)];
                if f64::from(m) > WATERFALL_FLOOR_DB + 1.0 {
                    xs.push(grid.times_ms[f] / 1000.0);
                    ys.push(f64::from(m));
                }
            }
            let decay_time_s = linear_t60(&xs, &ys);
            Resonance {
                freq_hz,
                level_db: f64::from(v),
                decay_time_s,
            }
        })
        .collect();
    out.sort_by(|a, b| b.level_db.total_cmp(&a.level_db));
    out
}

fn linear_t60(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let n_f = n as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let denom = n_f * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return f64::NAN;
    }
    let slope = (n_f * sxy - sx * sy) / denom;
    if slope < -1e-9 {
        -60.0 / slope
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modal_ir(sr: f64, modes: &[(f64, f64, f64)]) -> Vec<f32> {
        // Sum of exponentially decaying sines: (freq, t60, amp).
        let n = (0.6 * sr) as usize;
        let mut ir = vec![0.0f32; n];
        ir[0] = 1.0;
        for &(f, t60, amp) in modes {
            let alpha = 3.0 * std::f64::consts::LN_10 / t60;
            for (i, s) in ir.iter_mut().enumerate().skip(1) {
                let t = i as f64 / sr;
                *s +=
                    (amp * (-alpha * t).exp() * (2.0 * std::f64::consts::PI * f * t).sin()) as f32;
            }
        }
        ir
    }

    #[test]
    fn grid_spans_minus5_to_500ms() {
        let sr = 48000.0;
        let ir = modal_ir(sr, &[(120.0, 0.5, 0.5)]);
        let grid = waterfall_grid(&ir, sr, &WaterfallConfig::default());
        assert!(!grid.times_ms.is_empty());
        assert!(
            grid.times_ms[0] <= 0.0,
            "first frame {:?}",
            grid.times_ms[0]
        );
        assert!(
            grid.times_ms.last().copied().unwrap_or(0.0) >= 400.0,
            "last frame {:?}",
            grid.times_ms.last()
        );
        assert!(grid.mags_db.len() <= DEFAULT_MAX_FRAMES);
        assert!(grid.freqs_hz.len() <= DEFAULT_MAX_BINS);
    }

    #[test]
    fn detects_planted_resonance_with_decay() {
        let sr = 48000.0;
        let ir = modal_ir(sr, &[(110.0, 0.6, 1.0), (440.0, 0.2, 0.3)]);
        let cfg = WaterfallConfig::default();
        let grid = waterfall_grid(&ir, sr, &cfg);
        let res = detect_resonances(&grid, &cfg);
        assert!(!res.is_empty(), "expected resonances");
        let top = &res[0];
        // Frequency resolution after decimation is coarse; allow one bin.
        assert!(
            (top.freq_hz - 110.0).abs() < 60.0,
            "top resonance at {} Hz (expected ≈ 110)",
            top.freq_hz
        );
        assert!(
            (top.decay_time_s - 0.6).abs() < 0.3,
            "decay = {:.3} s (expected ≈ 0.6)",
            top.decay_time_s
        );
    }

    #[test]
    fn empty_input_yields_empty_grid() {
        let grid = waterfall_grid(&[], 48000.0, &WaterfallConfig::default());
        assert!(grid.times_ms.is_empty());
        assert!(detect_resonances(&grid, &WaterfallConfig::default()).is_empty());
        // Past-the-end direct reference is equally empty.
        let ir = modal_ir(48000.0, &[(110.0, 0.5, 0.5)]);
        let grid = waterfall_grid_at(&ir, 48000.0, ir.len(), &WaterfallConfig::default());
        assert!(grid.times_ms.is_empty());
    }

    #[test]
    fn explicit_direct_matches_auto() {
        // Unambiguous direct (2.0 delta at sample 500): the explicit
        // reference must reproduce the automatic grid exactly.
        let sr = 48000.0;
        let mut ir = vec![0.0f32; (0.6 * sr) as usize];
        ir[500] = 2.0;
        for (i, s) in ir.iter_mut().enumerate().skip(501) {
            let t = i as f64 / sr;
            *s += (0.5 * (-t * 10.0).exp() * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as f32;
        }
        let cfg = WaterfallConfig::default();
        let auto_grid = waterfall_grid(&ir, sr, &cfg);
        let exp_grid = waterfall_grid_at(&ir, sr, 500, &cfg);
        assert_eq!(auto_grid.times_ms, exp_grid.times_ms);
        assert_eq!(auto_grid.freqs_hz, exp_grid.freqs_hz);
        assert_eq!(auto_grid.mags_db, exp_grid.mags_db);
    }

    #[test]
    fn silent_ir_sits_at_floor() {
        // All zeros: no peak to normalise against, every cell at the floor.
        let grid = waterfall_grid(&vec![0.0f32; 24000], 48000.0, &WaterfallConfig::default());
        assert!(!grid.mags_db.is_empty());
        for row in &grid.mags_db {
            for &v in row {
                assert_eq!(v, WATERFALL_FLOOR_DB as f32);
            }
        }
    }

    #[test]
    fn close_peaks_keep_loudest_only() {
        // Hand-built grid (no STFT interference): the 60 ms slice has
        // local maxima at 400 Hz (−18 dB) and 420 Hz (−19 dB), 20 Hz apart
        // (< 25 Hz separation), plus a distant 500 Hz peak. Only 400 Hz
        // and 500 Hz may survive; the decay fit must read T60 = 0.2 s.
        let times_ms = vec![0.0, 20.0, 60.0, 100.0, 200.0];
        let freqs_hz = vec![400.0, 410.0, 420.0, 500.0];
        // −0.3 dB/ms slope in every bin; slice peaks at 400/420/500.
        let peaks = [0.0f64, -10.0, -1.0, -0.5];
        let mags_db: Vec<Vec<f32>> = times_ms
            .iter()
            .map(|&t| {
                freqs_hz
                    .iter()
                    .zip(peaks.iter())
                    .map(|(_, &p)| (p - 0.3 * t) as f32)
                    .collect()
            })
            .collect();
        let grid = WaterfallGrid {
            times_ms,
            freqs_hz,
            mags_db,
        };
        let cfg = WaterfallConfig::default();
        let res = detect_resonances(&grid, &cfg);
        assert_eq!(res.len(), 2, "expected 400 + 500 Hz, got {res:?}");
        assert!((res[0].freq_hz - 400.0).abs() < 1e-9);
        assert!((res[1].freq_hz - 500.0).abs() < 1e-9);
        assert!((res[0].decay_time_s - 0.2).abs() < 0.01);
        // All-NaN slice: no resonances.
        let mut nan_grid = grid.clone();
        for row in nan_grid.mags_db.iter_mut() {
            for v in row.iter_mut() {
                *v = f32::NAN;
            }
        }
        assert!(detect_resonances(&nan_grid, &cfg).is_empty());
    }

    #[test]
    fn linear_t60_edge_cases() {
        // Fewer than two points: unfittable.
        assert!(linear_t60(&[0.0], &[-1.0]).is_nan());
        assert!(linear_t60(&[], &[]).is_nan());
        // Degenerate time base: unfittable.
        assert!(linear_t60(&[1.0, 1.0], &[0.0, -10.0]).is_nan());
        // Rising (non-decaying) slope: not a decay.
        assert!(linear_t60(&[0.0, 1.0], &[-20.0, -10.0]).is_nan());
        // Exact −60 dB/s slope: T60 = 1 s.
        assert!((linear_t60(&[0.0, 1.0], &[0.0, -60.0]) - 1.0).abs() < 1e-9);
    }
}

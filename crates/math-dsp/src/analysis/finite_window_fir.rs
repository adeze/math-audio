//! Finite recorded-window energy of a causal FIR applied to an impulse response.
//! The anchor is supplied by the caller; no peak or acoustic event is inferred.
//! Projection semantics adapted from Open Room Calibration's
//! `FiniteWindowFIRObjective.swift` (Apache-2.0).

use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct FiniteWindowFirConfig {
    pub capture_rate_hz: u32,
    pub filter_rate_hz: u32,
    pub tap_count: usize,
    pub anchor_sample: usize,
    pub frequencies_hz: Vec<f64>,
    pub frequency_weights: Vec<f64>,
    pub window_seconds: f64,
    pub starts_seconds: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteWindowFirError {
    InvalidConfig,
    InsufficientRecording,
    InvalidTaps,
    InvalidEnergy,
}

impl std::fmt::Display for FiniteWindowFirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FiniteWindowFirError {}

#[derive(Clone, Debug)]
struct Projection {
    real: Vec<f64>,
    imaginary: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct FiniteWindowFirObjective {
    windows: Vec<Vec<Projection>>,
    weights: Vec<f64>,
    tap_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FiniteWindowFirComparison {
    pub early_change_db: f64,
    pub late_changes_db: Vec<f64>,
    pub relative_tail_changes_db: Vec<f64>,
}

impl FiniteWindowFirObjective {
    /// Precomputes Hann-windowed arbitrary-frequency projections of each FIR tap.
    /// Frequencies must lie below the filter Nyquist frequency; rates must have
    /// an integral capture/filter ratio. All windows must fit the saved impulse.
    pub fn new(
        impulse: &[f64],
        config: &FiniteWindowFirConfig,
    ) -> Result<Self, FiniteWindowFirError> {
        let bad = FiniteWindowFirError::InvalidConfig;
        if config.capture_rate_hz == 0
            || config.filter_rate_hz == 0
            || !config.capture_rate_hz.is_multiple_of(config.filter_rate_hz)
            || config.tap_count == 0
            || config.tap_count > isize::MAX as usize / std::mem::size_of::<f64>()
            || impulse.is_empty()
            || !impulse.iter().all(|x| x.is_finite())
            || config.anchor_sample >= impulse.len()
            || config.frequencies_hz.is_empty()
            || config.frequencies_hz.len() != config.frequency_weights.len()
            || !config
                .frequencies_hz
                .iter()
                .all(|&f| f.is_finite() && f > 0.0 && f < f64::from(config.filter_rate_hz) / 2.0)
            || !config
                .frequency_weights
                .iter()
                .all(|&w| w.is_finite() && w >= 0.0)
            || !config.frequency_weights.iter().sum::<f64>().is_finite()
            || config.frequency_weights.iter().sum::<f64>() <= 0.0
            || !config.window_seconds.is_finite()
            || config.window_seconds <= 0.0
            || config.starts_seconds.len() < 2
            || config.starts_seconds[0] != 0.0
            || config
                .starts_seconds
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || !config
                .starts_seconds
                .iter()
                .all(|&s| s.is_finite() && s >= 0.0)
        {
            return Err(bad);
        }
        let rate = f64::from(config.capture_rate_hz);
        let count_float = config.window_seconds * rate;
        if !count_float.is_finite() || count_float < 2.0 || count_float > usize::MAX as f64 {
            return Err(bad);
        }
        let count = count_float as usize;
        let stride = (config.capture_rate_hz / config.filter_rate_hz) as usize;
        (config.tap_count - 1).checked_mul(stride).ok_or(bad)?;
        let mut windows = Vec::with_capacity(config.starts_seconds.len());
        for &offset in &config.starts_seconds {
            let offset_float = offset * rate;
            if !offset_float.is_finite() || offset_float > usize::MAX as f64 {
                return Err(bad);
            }
            let start = config
                .anchor_sample
                .checked_add(offset_float as usize)
                .ok_or(bad)?;
            let end = start.checked_add(count).ok_or(bad)?;
            if end > impulse.len() {
                return Err(FiniteWindowFirError::InsufficientRecording);
            }
            let mut projections = Vec::with_capacity(config.frequencies_hz.len());
            for &frequency in &config.frequencies_hz {
                let window_weights: Vec<(f64, f64)> = (0..count)
                    .map(|index| {
                        let hann = 0.5 - 0.5 * (2.0 * PI * index as f64 / (count - 1) as f64).cos();
                        let phase = 2.0 * PI * frequency * index as f64 / rate;
                        (hann * phase.cos(), -hann * phase.sin())
                    })
                    .collect();
                let mut real = vec![0.0; config.tap_count];
                let mut imaginary = vec![0.0; config.tap_count];
                for tap in 0..config.tap_count {
                    let delay = tap * stride;
                    let first = delay.saturating_sub(start).min(count);
                    for (index, &(window_real, window_imaginary)) in
                        window_weights.iter().enumerate().skip(first)
                    {
                        let source = (start + index) - delay;
                        let sample = impulse[source];
                        real[tap] += sample * window_real;
                        imaginary[tap] += sample * window_imaginary;
                    }
                }
                projections.push(Projection { real, imaginary });
            }
            windows.push(projections);
        }
        Ok(Self {
            windows,
            weights: config.frequency_weights.clone(),
            tap_count: config.tap_count,
        })
    }

    /// Returns positive finite weighted spectral energy for each window.
    pub fn energy(&self, taps: &[f64]) -> Result<Vec<f64>, FiniteWindowFirError> {
        if taps.len() != self.tap_count || !taps.iter().all(|x| x.is_finite()) {
            return Err(FiniteWindowFirError::InvalidTaps);
        }
        self.windows
            .iter()
            .map(|projections| {
                let energy: f64 = projections
                    .iter()
                    .zip(&self.weights)
                    .map(|(projection, &weight)| {
                        let real: f64 = taps
                            .iter()
                            .zip(&projection.real)
                            .map(|(tap, value)| tap * value)
                            .sum();
                        let imaginary: f64 = taps
                            .iter()
                            .zip(&projection.imaginary)
                            .map(|(tap, value)| tap * value)
                            .sum();
                        weight * (real * real + imaginary * imaginary)
                    })
                    .sum();
                if energy.is_finite() && energy > 0.0 {
                    Ok(energy)
                } else {
                    Err(FiniteWindowFirError::InvalidEnergy)
                }
            })
            .collect()
    }

    /// Compares candidate against baseline, in dB of energy ratio.
    pub fn compare(
        &self,
        taps: &[f64],
        baseline: &[f64],
    ) -> Result<FiniteWindowFirComparison, FiniteWindowFirError> {
        let actual = self.energy(taps)?;
        let reference = self.energy(baseline)?;
        let changes: Vec<f64> = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| 10.0 * (a / b).log10())
            .collect();
        if !changes.iter().all(|x| x.is_finite()) {
            return Err(FiniteWindowFirError::InvalidEnergy);
        }
        let early_change_db = changes[0];
        let late_changes_db = changes[1..].to_vec();
        let relative_tail_changes_db = late_changes_db
            .iter()
            .map(|x| x - early_change_db)
            .collect();
        Ok(FiniteWindowFirComparison {
            early_change_db,
            late_changes_db,
            relative_tail_changes_db,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(tap_count: usize) -> (Vec<f64>, FiniteWindowFirConfig) {
        let mut impulse = vec![0.0; 240];
        impulse[15] = 1.0;
        impulse[23] = 0.4;
        impulse[47] = -0.2;
        impulse[100] = 0.1;
        let config = FiniteWindowFirConfig {
            capture_rate_hz: 1000,
            filter_rate_hz: 500,
            tap_count,
            anchor_sample: 15,
            frequencies_hz: vec![37.0, 91.0],
            frequency_weights: vec![1.0, 0.5],
            window_seconds: 0.064,
            starts_seconds: vec![0.0, 0.025, 0.070],
        };
        (impulse, config)
    }

    // Independent direct convolution and projection, including shifted arrivals.
    fn direct_energy(impulse: &[f64], config: &FiniteWindowFirConfig, taps: &[f64]) -> Vec<f64> {
        let stride = (config.capture_rate_hz / config.filter_rate_hz) as usize;
        let mut response = vec![0.0; impulse.len()];
        for (index, sample) in response.iter_mut().enumerate() {
            for (tap_index, tap) in taps.iter().enumerate() {
                if index >= tap_index * stride {
                    *sample += tap * impulse[index - tap_index * stride];
                }
            }
        }
        let count = (config.window_seconds * f64::from(config.capture_rate_hz)) as usize;
        config
            .starts_seconds
            .iter()
            .map(|offset| {
                let start =
                    config.anchor_sample + (offset * f64::from(config.capture_rate_hz)) as usize;
                config
                    .frequencies_hz
                    .iter()
                    .zip(&config.frequency_weights)
                    .map(|(&frequency, &weight)| {
                        let (mut real, mut imaginary) = (0.0, 0.0);
                        for index in 0..count {
                            let hann =
                                0.5 - 0.5 * (2.0 * PI * index as f64 / (count - 1) as f64).cos();
                            let phase = 2.0 * PI * frequency * index as f64
                                / f64::from(config.capture_rate_hz);
                            real += response[start + index] * hann * phase.cos();
                            imaginary -= response[start + index] * hann * phase.sin();
                        }
                        weight * (real * real + imaginary * imaginary)
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn matches_direct_convolution_with_shifted_impulse() {
        let (impulse, config) = fixture(5);
        let taps = [0.7, -0.2, 0.15, 0.0, 0.08];
        let objective = FiniteWindowFirObjective::new(&impulse, &config).unwrap();
        for (actual, expected) in objective
            .energy(&taps)
            .unwrap()
            .iter()
            .zip(direct_energy(&impulse, &config, &taps))
        {
            assert!((actual - expected).abs() <= 1e-10 * expected.max(1.0));
        }
        let mut shifted = impulse.clone();
        shifted.rotate_right(4);
        let mut shifted_config = config.clone();
        shifted_config.anchor_sample += 4;
        let shifted_objective = FiniteWindowFirObjective::new(&shifted, &shifted_config).unwrap();
        for (a, b) in objective
            .energy(&taps)
            .unwrap()
            .iter()
            .zip(shifted_objective.energy(&taps).unwrap())
        {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn gain_sign_and_relative_tail() {
        let (impulse, config) = fixture(2);
        let objective = FiniteWindowFirObjective::new(&impulse, &config).unwrap();
        let baseline = [1.0, 0.2];
        let positive = objective.compare(&[2.0, 0.4], &baseline).unwrap();
        let negative = objective.compare(&[-2.0, -0.4], &baseline).unwrap();
        assert_eq!(positive, negative);
        assert!((positive.early_change_db - 10.0 * 4.0_f64.log10()).abs() < 1e-12);
        assert!(
            positive
                .relative_tail_changes_db
                .iter()
                .all(|x| x.abs() < 1e-12)
        );
    }

    #[test]
    fn ratios_boundaries_and_long_filter() {
        let (mut impulse, mut config) = fixture(513);
        impulse.resize(1400, 0.0);
        let objective = FiniteWindowFirObjective::new(&impulse, &config).unwrap();
        let mut taps = vec![0.0; 513];
        taps[0] = 1.0;
        assert!(objective.energy(&taps).is_ok());
        impulse.resize(3000, 0.0);
        impulse[1500] = 1.0;
        let mut long_config = config.clone();
        long_config.tap_count = 1;
        long_config.window_seconds = 1.1;
        long_config.starts_seconds = vec![0.0, 1.2];
        assert!(
            FiniteWindowFirObjective::new(&impulse, &long_config)
                .unwrap()
                .energy(&[1.0])
                .is_ok()
        );
        config.capture_rate_hz = 1400;
        assert_eq!(
            FiniteWindowFirObjective::new(&impulse, &config).unwrap_err(),
            FiniteWindowFirError::InvalidConfig
        );
        config.capture_rate_hz = 1000;
        config.anchor_sample = impulse.len() - 1;
        assert_eq!(
            FiniteWindowFirObjective::new(&impulse, &config).unwrap_err(),
            FiniteWindowFirError::InsufficientRecording
        );
    }

    #[test]
    fn rejects_invalid_inputs_and_zero_energy() {
        let (impulse, mut config) = fixture(2);
        config.frequency_weights = vec![0.0, 0.0];
        assert_eq!(
            FiniteWindowFirObjective::new(&impulse, &config).unwrap_err(),
            FiniteWindowFirError::InvalidConfig
        );
        config.frequency_weights = vec![1.0, 1.0];
        config.starts_seconds = vec![0.0, 0.0];
        assert_eq!(
            FiniteWindowFirObjective::new(&impulse, &config).unwrap_err(),
            FiniteWindowFirError::InvalidConfig
        );
        config.starts_seconds = vec![0.0, 0.025, 0.070];
        let objective = FiniteWindowFirObjective::new(&impulse, &config).unwrap();
        assert_eq!(
            objective.energy(&[0.0, 0.0]),
            Err(FiniteWindowFirError::InvalidEnergy)
        );
        assert_eq!(
            objective.energy(&[f64::NAN, 1.0]),
            Err(FiniteWindowFirError::InvalidTaps)
        );
    }

    #[test]
    fn matches_swift_reference_fixture() {
        // Generated by the current Swift FiniteWindowFIRObjective with an explicit
        // peak at sample 7. f32 source values are serialized exactly as f64.
        let value: serde_json::Value =
            serde_json::from_str(include_str!("finite_window_fir_swift_fixture.json")).unwrap();
        let numbers = |key: &str| -> Vec<f64> {
            value[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        let config = FiniteWindowFirConfig {
            capture_rate_hz: value["capture_rate_hz"].as_u64().unwrap() as u32,
            filter_rate_hz: value["filter_rate_hz"].as_u64().unwrap() as u32,
            tap_count: numbers("taps").len(),
            anchor_sample: value["anchor_sample"].as_u64().unwrap() as usize,
            frequencies_hz: numbers("frequencies_hz"),
            frequency_weights: numbers("frequency_weights"),
            window_seconds: value["window_seconds"].as_f64().unwrap(),
            starts_seconds: numbers("starts_seconds"),
        };
        let objective = FiniteWindowFirObjective::new(&numbers("impulse"), &config).unwrap();
        let actual = objective.energy(&numbers("taps")).unwrap();
        for (a, b) in actual.iter().zip(numbers("energy")) {
            assert!((a - b).abs() <= 1e-10 * b.max(1.0));
        }
        let compared = objective
            .compare(&numbers("taps"), &numbers("baseline"))
            .unwrap();
        assert!(
            (compared.early_change_db - value["early_change_db"].as_f64().unwrap()).abs() < 1e-9
        );
        assert!((compared.late_changes_db[0] - numbers("late_changes_db")[0]).abs() < 1e-9);
        assert!(
            (compared.relative_tail_changes_db[0] - numbers("relative_tail_changes_db")[0]).abs()
                < 1e-9
        );
    }
}

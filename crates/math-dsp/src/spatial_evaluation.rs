//! Read-only ideal crossover and spatial DRC diagnostics.
//! Adapted from Open Room Calibration's `SpatialExperimentalModels.swift` and
//! `SpatialDRCSafetyEvaluator.swift` (Apache-2.0). These models do not describe
//! a measured receiver crossover or authorize filter promotion.

use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct CrossoverSeat<'a> {
    pub satellite: &'a [Complex64],
    pub subwoofer: &'a [Complex64],
}

#[derive(Debug, Clone, PartialEq)]
pub struct SeatSummation {
    pub energy_before_db: f64,
    pub energy_after_db: f64,
    pub gain_db: f64,
    pub cancellation_rmse_db: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiSeatCrossoverReport {
    pub seats: Vec<SeatSummation>,
    pub mean_gain_db: f64,
    pub min_seat_gain_db: f64,
    pub variance_before_db2: f64,
    pub variance_after_db2: f64,
    pub variance_reduction_percent: f64,
}

impl MultiSeatCrossoverReport {
    /// Caller-supplied model-screening thresholds; never an export decision.
    pub fn meets_gain_thresholds(&self, minimum_mean_db: f64, minimum_seat_db: f64) -> bool {
        minimum_mean_db.is_finite()
            && minimum_seat_db.is_finite()
            && self.mean_gain_db >= minimum_mean_db
            && self.min_seat_gain_db >= minimum_seat_db
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialEvaluationError {
    InvalidInput,
    EmptyCrossoverBand,
}

impl std::fmt::Display for SpatialEvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SpatialEvaluationError {}

fn variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

/// Evaluate the continuous ideal LR4 model used by the Swift reference.
/// Screening thresholds belong to the caller, not to this numerical kernel.
pub fn evaluate_ideal_multiseat_crossover(
    frequencies_hz: &[f64],
    seats: &[CrossoverSeat<'_>],
    crossover_hz: f64,
    subwoofer_delay_seconds: f64,
    subwoofer_inverted: bool,
) -> Result<MultiSeatCrossoverReport, SpatialEvaluationError> {
    if frequencies_hz.is_empty()
        || seats.is_empty()
        || !crossover_hz.is_finite()
        || crossover_hz <= 0.0
        || !subwoofer_delay_seconds.is_finite()
        || frequencies_hz.iter().any(|f| !f.is_finite() || *f < 0.0)
        || seats.iter().any(|seat| {
            seat.satellite.len() != frequencies_hz.len()
                || seat.subwoofer.len() != frequencies_hz.len()
                || seat
                    .satellite
                    .iter()
                    .chain(seat.subwoofer)
                    .any(|v| !v.re.is_finite() || !v.im.is_finite())
        })
    {
        return Err(SpatialEvaluationError::InvalidInput);
    }
    let band: Vec<_> = frequencies_hz
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, f)| *f >= crossover_hz * 0.5 && *f <= crossover_hz * 2.0)
        .collect();
    if band.is_empty() {
        return Err(SpatialEvaluationError::EmptyCrossoverBand);
    }
    let mut reports = Vec::with_capacity(seats.len());
    for seat in seats {
        let (mut before, mut after, mut loss_squared) = (0.0, 0.0, 0.0);
        for &(index, frequency) in &band {
            let ratio = frequency / crossover_hz;
            let denominator = Complex64::new(1.0 - ratio * ratio, 2.0_f64.sqrt() * ratio).powu(2);
            let satellite =
                seat.satellite[index] * Complex64::new(ratio.powi(4), 0.0) / denominator;
            let subwoofer = seat.subwoofer[index] / denominator;
            let baseline = satellite + subwoofer;
            let polarity = if subwoofer_inverted { -1.0 } else { 1.0 };
            let aligned = subwoofer
                * Complex64::from_polar(polarity, -2.0 * PI * frequency * subwoofer_delay_seconds);
            let candidate = satellite + aligned;
            before += baseline.norm_sqr();
            after += candidate.norm_sqr();
            let loss = (20.0
                * ((satellite.norm() + aligned.norm()).max(1e-9) / candidate.norm().max(1e-9))
                    .log10())
            .max(0.0);
            loss_squared += loss * loss;
        }
        let count = band.len() as f64;
        let energy_before_db = 10.0 * (before / count).max(1e-12).log10();
        let energy_after_db = 10.0 * (after / count).max(1e-12).log10();
        reports.push(SeatSummation {
            energy_before_db,
            energy_after_db,
            gain_db: energy_after_db - energy_before_db,
            cancellation_rmse_db: (loss_squared / count).sqrt(),
        });
    }
    let mean_gain_db = reports.iter().map(|seat| seat.gain_db).sum::<f64>() / reports.len() as f64;
    let min_seat_gain_db = reports
        .iter()
        .map(|seat| seat.gain_db)
        .fold(f64::INFINITY, f64::min);
    let variance_before_db2 = variance(
        &reports
            .iter()
            .map(|seat| seat.energy_before_db)
            .collect::<Vec<_>>(),
    );
    let variance_after_db2 = variance(
        &reports
            .iter()
            .map(|seat| seat.energy_after_db)
            .collect::<Vec<_>>(),
    );
    Ok(MultiSeatCrossoverReport {
        mean_gain_db,
        min_seat_gain_db,
        variance_before_db2,
        variance_after_db2,
        variance_reduction_percent: ((variance_before_db2 - variance_after_db2)
            / variance_before_db2.max(1e-9)
            * 100.0)
            .max(0.0),
        seats: reports,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadroomRisk {
    Safe,
    Moderate,
    Severe,
}

#[derive(Debug, Clone)]
pub struct DrcSafetyConfig {
    pub schroeder_transition_hz: f64,
    pub max_q_modal: f64,
    pub max_q_high_variance: f64,
    pub variance_threshold_db2: f64,
}

impl Default for DrcSafetyConfig {
    fn default() -> Self {
        Self {
            schroeder_transition_hz: 180.0,
            max_q_modal: 8.0,
            max_q_high_variance: 2.0,
            variance_threshold_db2: 9.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DrcSafetyReport {
    pub mean_spl_db: Vec<f64>,
    pub spatial_variance_db2: Vec<f64>,
    pub dynamic_q_caps: Vec<f64>,
    pub max_allowable_boost_db: Vec<f64>,
    pub recommended_modal_cuts_db: Vec<f64>,
    pub headroom_risk: HeadroomRisk,
    pub peak_constructive_gain_db: f64,
}

/// Evaluate pressure spectra without modifying filters. A zero boost limit is
/// policy embedded in the reference evaluator, not a headroom measurement.
pub fn evaluate_spatial_drc_safety(
    frequencies_hz: &[f64],
    seat_pressures: &[&[Complex64]],
    target_curve_db: Option<&[f64]>,
    config: &DrcSafetyConfig,
) -> Result<DrcSafetyReport, SpatialEvaluationError> {
    let count = frequencies_hz.len();
    if count == 0
        || seat_pressures.is_empty()
        || target_curve_db
            .is_some_and(|target| target.len() != count || target.iter().any(|v| !v.is_finite()))
        || frequencies_hz.iter().any(|f| !f.is_finite() || *f < 0.0)
        || seat_pressures.iter().any(|seat| {
            seat.len() != count || seat.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        })
        || !config.schroeder_transition_hz.is_finite()
        || config.schroeder_transition_hz <= 0.0
        || !config.max_q_modal.is_finite()
        || config.max_q_modal <= 0.0
        || !config.max_q_high_variance.is_finite()
        || config.max_q_high_variance <= 0.0
        || !config.variance_threshold_db2.is_finite()
        || config.variance_threshold_db2 <= 0.0
    {
        return Err(SpatialEvaluationError::InvalidInput);
    }
    let mut report = DrcSafetyReport {
        mean_spl_db: Vec::with_capacity(count),
        spatial_variance_db2: Vec::with_capacity(count),
        dynamic_q_caps: Vec::with_capacity(count),
        max_allowable_boost_db: vec![0.0; count],
        recommended_modal_cuts_db: Vec::with_capacity(count),
        headroom_risk: HeadroomRisk::Safe,
        peak_constructive_gain_db: f64::NEG_INFINITY,
    };
    for (index, &frequency) in frequencies_hz.iter().enumerate() {
        let levels: Vec<_> = seat_pressures
            .iter()
            .map(|seat| 20.0 * seat[index].norm().max(1e-6).log10())
            .collect();
        let mean = levels.iter().sum::<f64>() / levels.len() as f64;
        let spread = variance(&levels);
        let mut q_cap = if spread > config.variance_threshold_db2 {
            config.max_q_high_variance
        } else {
            config.max_q_modal
        };
        if frequency > config.schroeder_transition_hz {
            q_cap = q_cap.min(3.0);
        }
        let error = mean - target_curve_db.map_or(0.0, |target| target[index]);
        let cut = if frequency <= config.schroeder_transition_hz {
            if error > 0.5 {
                -error / (1.0 + spread / config.variance_threshold_db2)
            } else {
                0.0
            }
        } else if error > 2.0 && spread < config.variance_threshold_db2 {
            -(error * 0.5).min(3.0)
        } else {
            0.0
        };
        report.mean_spl_db.push(mean);
        report.spatial_variance_db2.push(spread);
        report.dynamic_q_caps.push(q_cap);
        report.recommended_modal_cuts_db.push(cut);
        report.peak_constructive_gain_db = report.peak_constructive_gain_db.max(error);
    }
    report.headroom_risk = if report.peak_constructive_gain_db > 6.0 {
        HeadroomRisk::Severe
    } else if report.peak_constructive_gain_db > 3.0 {
        HeadroomRisk::Moderate
    } else {
        HeadroomRisk::Safe
    };
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverted_sub_is_constructive_but_destructive_flip_is_rejected() {
        let frequencies = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0, 240.0];
        let satellite = [Complex64::new(1.0, 0.0); 8];
        let inverted = [Complex64::new(-1.0, 0.0); 8];
        let normal = [Complex64::new(1.0, 0.0); 8];
        let gain = evaluate_ideal_multiseat_crossover(
            &frequencies,
            &[CrossoverSeat {
                satellite: &satellite,
                subwoofer: &inverted,
            }; 3],
            120.0,
            0.0,
            true,
        )
        .unwrap();
        assert!(gain.meets_gain_thresholds(0.25, -0.5) && gain.min_seat_gain_db > 0.25);
        let loss = evaluate_ideal_multiseat_crossover(
            &frequencies,
            &[CrossoverSeat {
                satellite: &satellite,
                subwoofer: &normal,
            }; 2],
            120.0,
            0.0,
            true,
        )
        .unwrap();
        assert!(!loss.meets_gain_thresholds(0.25, -0.5) && loss.mean_gain_db < 0.0);
    }

    #[test]
    fn drc_limits_boost_and_caps_q() {
        let frequencies = [50.0, 100.0, 150.0, 250.0];
        let loud = [Complex64::new(10.0, 0.0); 4];
        let quiet = [Complex64::new(2.0, 0.0); 4];
        let report = evaluate_spatial_drc_safety(
            &frequencies,
            &[&loud, &quiet],
            None,
            &DrcSafetyConfig::default(),
        )
        .unwrap();
        assert!(report.max_allowable_boost_db.iter().all(|&v| v == 0.0));
        assert!(report.recommended_modal_cuts_db.iter().all(|&v| v <= 0.0));
        assert_eq!(report.dynamic_q_caps[0], 2.0);
        assert!(report.dynamic_q_caps[3] <= 3.0);
        assert_eq!(report.headroom_risk, HeadroomRisk::Severe);
    }

    #[test]
    fn invalid_shapes_and_nonfinite_inputs_fail() {
        let pressure = [Complex64::new(1.0, 0.0)];
        assert_eq!(
            evaluate_ideal_multiseat_crossover(
                &[10.0],
                &[CrossoverSeat {
                    satellite: &pressure,
                    subwoofer: &[]
                }],
                100.0,
                0.0,
                false
            )
            .unwrap_err(),
            SpatialEvaluationError::InvalidInput
        );
        assert!(
            evaluate_spatial_drc_safety(
                &[f64::NAN],
                &[&pressure],
                None,
                &DrcSafetyConfig::default()
            )
            .is_err()
        );
    }
}

//! Sweep timing from bounded spectral weighting and a correlation envelope.

// Rust guideline compliant 2026-02-21

use super::misc::next_power_of_two;
use super::plan::{plan_fft_forward, plan_fft_inverse};
use super::types::LagEstimate;
use rustfft::num_complex::Complex;

/// Minimum excess-envelope confidence for an acoustic sweep lock.
///
/// This is a detection score, not a probability. A normalized envelope must
/// exceed its competing peaks by enough energy to clear this threshold.
pub const MIN_SWEEP_LAG_CONFIDENCE: f32 = 0.2;

// Partial whitening is intentionally bounded: a notch or an absent driver
// must not give arbitrarily high weight to microphone noise.
const MAX_WEIGHT_RATIO: f64 = 10.0;
// Each retained octave must carry at least 0.5% of reference energy and
// 0.001% of capture energy. This excludes sweep leakage and absent bands.
const MIN_REFERENCE_BAND_SHARE: f64 = 0.005;
const MIN_CAPTURE_BAND_SHARE: f64 = 0.00001;

/// Estimate acoustic sweep lag without letting bass resonances dominate.
///
/// Uses octave-smoothed, bounded partial whitening of both signals followed
/// by an analytic correlation envelope. Only frequencies supported by both
/// signals contribute. Whitening is detection-only; neither input is modified.
/// The lag describes the strongest resolvable matched arrival, not necessarily
/// the earliest direct sound. Narrowband responses have lower timing resolution.
///
/// Unlike the generic waveform estimator, confidence is the normalized envelope
/// peak times the fraction of peak energy exceeding the largest distant peak.
/// Scores from the two methods must not be interpreted as probabilities.
///
/// # Errors
/// Returns an error for invalid inputs, missing supported energy, an incomplete
/// reference overlap, or insufficient/ambiguous sweep evidence.
pub fn estimate_sweep_lag_with_confidence(
    reference: &[f32],
    recorded: &[f32],
    sample_rate: u32,
) -> Result<LagEstimate, String> {
    if sample_rate == 0 || reference.is_empty() || recorded.is_empty() {
        return Err("Sweep lag requires nonempty signals and a positive sample rate".into());
    }
    if reference.iter().chain(recorded).any(|x| !x.is_finite()) {
        return Err("Sweep lag requires finite samples".into());
    }
    let peak = reference.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if peak == 0.0 || recorded.iter().all(|x| *x == 0.0) {
        return Err("Sweep lag: one signal has no energy".into());
    }
    // Ignore only tiny fade samples when determining complete stimulus coverage.
    let active_start = reference
        .iter()
        .position(|x| x.abs() > peak * 1e-4)
        .unwrap();
    let active_end = reference
        .iter()
        .rposition(|x| x.abs() > peak * 1e-4)
        .unwrap()
        + 1;
    if recorded.len() < active_end - active_start {
        return Err("Sweep lag: recording is shorter than the active stimulus".into());
    }
    let fft_size = next_power_of_two(reference.len() + recorded.len() - 1);
    let mut x = vec![Complex::new(0.0_f32, 0.0); fft_size];
    let mut y = x.clone();
    for (dst, &src) in x.iter_mut().zip(reference) {
        dst.re = src;
    }
    for (dst, &src) in y.iter_mut().zip(recorded) {
        dst.re = src;
    }
    let fft = plan_fft_forward(fft_size);
    fft.process(&mut x);
    fft.process(&mut y);
    let bins = fft_size / 2 + 1;
    let energy = |v: &[Complex<f32>]| {
        v[1..bins]
            .iter()
            .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
            .sum::<f64>()
    };
    let x_energy = energy(&x);
    let y_energy = energy(&y);
    if x_energy <= 0.0 || y_energy <= 0.0 {
        return Err("Sweep lag: no alternating signal energy".into());
    }
    let hz_per_bin = sample_rate as f64 / fft_size as f64;
    let mut bands = Vec::new();
    // Ten Hz is below ordinary acoustic sweep starts; reference support removes
    // this band for higher starts. No fixed high-frequency requirement excludes subs.
    let mut low = 10.0_f64.min(sample_rate as f64 / 4.0).max(hz_per_bin);
    while low < sample_rate as f64 / 2.0 {
        let high = (low * 2.0).min(sample_rate as f64 / 2.0);
        let start = (low / hz_per_bin).ceil() as usize;
        let end = ((high / hz_per_bin).ceil() as usize).min(bins);
        if end > start {
            let power = |v: &[Complex<f32>]| {
                v[start..end]
                    .iter()
                    .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
                    .sum::<f64>()
            };
            let px = power(&x);
            let py = power(&y);
            if px >= x_energy * MIN_REFERENCE_BAND_SHARE && py >= y_energy * MIN_CAPTURE_BAND_SHARE
            {
                let count = (end - start) as f64;
                // A shared -1/8 power weight produces half-whitening of the
                // cross spectrum, preserving amplitude evidence in weak bands.
                let weight = ((px / count) * (py / count)).powf(-0.125);
                bands.push((low, high, weight));
            }
        }
        low = high;
    }
    if bands.is_empty() {
        return Err("Sweep lag: no jointly supported frequency bands".into());
    }
    let min_weight = bands.iter().map(|b| b.2).fold(f64::INFINITY, f64::min);
    for band in &mut bands {
        band.2 = (band.2 / min_weight).min(MAX_WEIGHT_RATIO);
    }
    let mut band_index = 0;
    let mut cross = vec![Complex::new(0.0_f32, 0.0); fft_size];
    let mut weighted_x_energy = 0.0_f64;
    let mut weighted_y_energy = 0.0_f64;
    let mut spectral_mass = 0.0_f64;
    let mut frequency_sum = 0.0_f64;
    let mut frequency_square_sum = 0.0_f64;
    for bin in 1..bins {
        let hz = bin as f64 * hz_per_bin;
        while band_index + 1 < bands.len() && hz >= bands[band_index].1 {
            band_index += 1;
        }
        let (low, high, mut weight) = bands[band_index];
        if hz < low || hz >= high {
            continue;
        }
        // Interpolate neighboring octave-center weights on log frequency.
        // Leave unsupported gaps at zero rather than interpolating noise into them.
        let center = (low * high).sqrt();
        if hz >= center
            && band_index + 1 < bands.len()
            && (bands[band_index + 1].0 - high).abs() < hz_per_bin
        {
            let next = bands[band_index + 1];
            let fraction = (hz / center).ln() / ((next.0 * next.1).sqrt() / center).ln();
            weight = weight.powf(1.0 - fraction) * next.2.powf(fraction);
        } else if hz < center
            && band_index > 0
            && (bands[band_index - 1].1 - low).abs() < hz_per_bin
        {
            let prev = bands[band_index - 1];
            let fraction =
                (hz / (prev.0 * prev.1).sqrt()).ln() / (center / (prev.0 * prev.1).sqrt()).ln();
            weight = prev.2.powf(1.0 - fraction) * weight.powf(fraction);
        }
        let weight_sq = weight * weight;
        let factor = if bin == fft_size / 2 { 1.0 } else { 2.0 };
        weighted_x_energy +=
            factor * weight_sq * (f64::from(x[bin].re).powi(2) + f64::from(x[bin].im).powi(2));
        weighted_y_energy +=
            factor * weight_sq * (f64::from(y[bin].re).powi(2) + f64::from(y[bin].im).powi(2));
        // Positive frequencies doubled and negative frequencies zeroed gives
        // the analytic correlation directly, saving a separate Hilbert FFT.
        cross[bin] = x[bin].conj() * y[bin] * (weight_sq * factor) as f32;
        let mass = f64::from(cross[bin].norm());
        spectral_mass += mass;
        frequency_sum += mass * hz;
        frequency_square_sum += mass * hz * hz;
    }
    let denominator = (weighted_x_energy * weighted_y_energy).sqrt() / fft_size as f64;
    if !denominator.is_finite()
        || denominator <= 0.0
        || !spectral_mass.is_finite()
        || spectral_mass <= 0.0
    {
        return Err("Sweep lag: no finite weighted signal energy".into());
    }
    plan_fft_inverse(fft_size).process(&mut cross);
    let scale = 1.0 / (fft_size as f64 * denominator);
    let score = |lag: isize| {
        let index = if lag < 0 {
            (fft_size as isize + lag) as usize
        } else {
            lag as usize
        };
        (f64::from(cross[index].norm()) * scale).min(1.0) as f32
    };
    // Complete active-stimulus coverage plus 3/4 buffer overlap protects against
    // a loud truncated copy at the tail. Normalization always uses full energy.
    let min_overlap = (reference.len().min(recorded.len()) * 3).div_ceil(4);
    let min_lag = (-(active_start as isize)).max(-((reference.len() - min_overlap) as isize));
    let max_lag = (recorded.len() as isize - active_end as isize)
        .min((recorded.len() - min_overlap) as isize);
    if max_lag < min_lag {
        return Err("Sweep lag: insufficient active-stimulus overlap".into());
    }
    let lag_samples = (min_lag..=max_lag)
        .max_by(|a, b| score(*a).total_cmp(&score(*b)))
        .unwrap();
    let normalized_peak = score(lag_samples);
    let mean_frequency = frequency_sum / spectral_mass;
    let rms_bandwidth = (frequency_square_sum / spectral_mass - mean_frequency * mean_frequency)
        .max(hz_per_bin * hz_per_bin)
        .sqrt();
    // Use actual weighted spectral spread, not nominal sweep bounds. A steep
    // crossover can leave tiny supported HF tails that do not sharpen its main
    // lobe. Two inverse RMS-bandwidth periods cover that lobe; a 2 ms floor
    // avoids treating nearby broadband oscillations as distinct arrivals.
    let guard = ((sample_rate as f64 * 0.002).max(2.0 * sample_rate as f64 / rms_bandwidth)).ceil()
        as isize;
    let sidelobe = (min_lag..=max_lag)
        .filter(|lag| (*lag - lag_samples).abs() > guard)
        .map(score)
        .fold(0.0_f32, f32::max);
    let peak_to_sidelobe_db = if sidelobe > 0.0 {
        20.0 * (normalized_peak / sidelobe).log10()
    } else {
        f32::INFINITY
    };
    let ratio = if normalized_peak > 0.0 {
        (sidelobe / normalized_peak).min(1.0)
    } else {
        1.0
    };
    let confidence = normalized_peak * (1.0 - ratio * ratio);
    if !confidence.is_finite() || confidence < MIN_SWEEP_LAG_CONFIDENCE {
        return Err(format!(
            "Sweep lag: insufficient or ambiguous signal evidence (envelope peak {normalized_peak:.3}, confidence {confidence:.3} < {MIN_SWEEP_LAG_CONFIDENCE:.3}, PSR {peak_to_sidelobe_db:.1} dB, supported band {:.0}–{:.0} Hz, main-lobe guard {:.1} ms)",
            bands[0].0,
            bands.last().unwrap().1,
            guard as f64 * 1000.0 / sample_rate as f64,
        ));
    }
    Ok(LagEstimate {
        lag_samples,
        normalized_peak,
        peak_to_sidelobe_db,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::{gen_log_sweep, prepare_signal_for_playback};

    fn sweep(rate: u32, low: f32, high: f32) -> Vec<f32> {
        prepare_signal_for_playback(gen_log_sweep(low, high, 0.5, rate, 2.0), rate, 20.0, 250.0)
    }

    fn delayed(signal: &[f32], delay: usize, rate: u32) -> Vec<f32> {
        let mut result = vec![0.0; delay + signal.len() + rate as usize / 2];
        result[delay..delay + signal.len()].copy_from_slice(signal);
        result
    }

    fn noise(len: usize) -> Vec<f32> {
        let mut state = 0xABCD1234_u64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 32) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn sweep_lock_delays_polarity_and_rates() {
        for rate in [8000, 44100, 48000, 96000] {
            let reference = sweep(rate, 20.0, (rate as f32 * 0.4).min(20000.0));
            for delay in [0, rate as usize / 3] {
                let mut take = delayed(&reference, delay, rate);
                for sample in &mut take {
                    *sample *= -0.03;
                }
                let lag = estimate_sweep_lag_with_confidence(&reference, &take, rate).unwrap();
                assert!(
                    (lag.lag_samples - delay as isize).abs() <= 1,
                    "{rate}: {lag:?}"
                );
                assert!(lag.confidence > 0.9, "{lag:?}");
            }
        }
    }

    #[test]
    fn sweep_lock_bass_resonance_retains_broadband_timing() {
        let rate = 48000;
        let reference = sweep(rate, 20.0, 20000.0);
        let delay = 12000;
        let mut take = delayed(&reference, delay, rate);
        let radius = (-1.0_f64 / (rate as f64 * 0.2)).exp();
        let coefficient = 2.0 * radius * (2.0 * std::f64::consts::PI * 40.0 / rate as f64).cos();
        let mut previous = [0.0_f64; 2];
        for sample in &mut take[delay..] {
            let resonance =
                f64::from(*sample) + coefficient * previous[0] - radius * radius * previous[1];
            previous = [resonance, previous[0]];
            *sample += (resonance * 0.0002) as f32;
        }
        let raw = super::super::estimate::estimate_lag_with_confidence(&reference, &take).unwrap();
        assert!(
            raw.confidence < MIN_SWEEP_LAG_CONFIDENCE,
            "old capture gate would reject: {raw:?}"
        );
        let lag = estimate_sweep_lag_with_confidence(&reference, &take, rate).unwrap();
        assert!((lag.lag_samples - delay as isize).abs() <= 4, "{lag:?}");
        assert!(lag.confidence >= MIN_SWEEP_LAG_CONFIDENCE, "{lag:?}");
    }

    #[test]
    fn sweep_lock_quadrature_phase_and_stronger_reflection() {
        let rate = 8000;
        let reference = sweep(rate, 20.0, 3000.0);
        let delay = 1000;
        let take = delayed(&reference, delay, rate);
        let quadrature: Vec<f32> = crate::instantaneous_frequency::analytic_signal(&take)
            .iter()
            .map(|sample| sample.im)
            .collect();
        let lag = estimate_sweep_lag_with_confidence(&reference, &quadrature, rate).unwrap();
        assert!((lag.lag_samples - delay as isize).abs() <= 1, "{lag:?}");
        let echo = 800;
        let mut reflected = delayed(&reference, delay + echo, rate);
        for (index, &sample) in reference.iter().enumerate() {
            reflected[delay + index] += sample * 0.3;
        }
        let lag = estimate_sweep_lag_with_confidence(&reference, &reflected, rate).unwrap();
        // Arrival semantics are explicit: select the strongest resolvable peak.
        assert_eq!(lag.lag_samples, (delay + echo) as isize);
    }

    #[test]
    fn sweep_lock_subwoofer_requires_no_high_frequency_support() {
        let rate = 48000;
        let reference = sweep(rate, 20.0, 200.0);
        let delay = 10000;
        let mut take = delayed(&reference, delay, rate);
        let alpha = 1.0 - (-2.0 * std::f64::consts::PI * 100.0 / rate as f64).exp();
        let mut state = 0.0;
        for sample in &mut take {
            state += alpha * (f64::from(*sample) - state);
            *sample = state as f32;
        }
        let lag = estimate_sweep_lag_with_confidence(&reference, &take, rate).unwrap();
        // A low-pass response has physical group delay; the detector must not
        // claim sample precision for a 180 Hz measurement bandwidth.
        assert!(
            (lag.lag_samples - delay as isize).abs() < rate as isize / 100,
            "{lag:?}"
        );
    }

    #[test]
    fn sweep_lock_fullband_reference_through_subwoofer_crossover() {
        let rate = 8000;
        let reference = sweep(rate, 20.0, 3000.0);
        let delay = 3000;
        let mut take = delayed(&reference, delay, rate);
        let alpha = 1.0 - (-2.0 * std::f64::consts::PI * 100.0 / rate as f64).exp();
        let mut states = [0.0; 4];
        for sample in &mut take {
            let mut value = f64::from(*sample);
            for state in &mut states {
                *state += alpha * (value - *state);
                value = *state;
            }
            *sample = value as f32;
        }
        let lag = estimate_sweep_lag_with_confidence(&reference, &take, rate).unwrap();
        assert!(
            (lag.lag_samples - delay as isize).abs() < rate as isize / 50,
            "{lag:?}"
        );
    }

    #[test]
    fn sweep_lock_rejects_unrelated_duration_and_tone() {
        let rate = 8000;
        let reference = sweep(rate, 20.0, 3000.0);
        let other = prepare_signal_for_playback(
            gen_log_sweep(20.0, 3000.0, 0.5, rate, 1.5),
            rate,
            20.0,
            250.0,
        );
        let unrelated = delayed(&other, 1000, rate);
        assert!(estimate_sweep_lag_with_confidence(&reference, &unrelated, rate).is_err());
        let tone = crate::signals::gen_tone(40.0, 0.5, rate, 3.0);
        assert!(estimate_sweep_lag_with_confidence(&reference, &tone, rate).is_err());
    }

    #[test]
    fn sweep_lock_rejects_noise_reversed_sweep_and_truncation() {
        let rate = 8000;
        let reference = sweep(rate, 20.0, 3000.0);
        let white = noise(reference.len() + 4000);
        let mut colored = white.clone();
        let mut state = 0.0;
        for sample in &mut colored {
            state = 0.98 * state + 0.02 * *sample;
            *sample = state;
        }
        let reversed: Vec<_> = reference.iter().copied().rev().collect();
        for take in [
            vec![0.0; white.len()],
            white,
            colored,
            reversed,
            reference[..reference.len() / 2].to_vec(),
        ] {
            assert!(estimate_sweep_lag_with_confidence(&reference, &take, rate).is_err());
        }
        let mut invalid = reference.clone();
        invalid[100] = f32::NAN;
        assert!(estimate_sweep_lag_with_confidence(&reference, &invalid, rate).is_err());
    }

    #[test]
    fn sweep_lock_rejects_two_equally_strong_arrivals() {
        let rate = 8000;
        let reference = sweep(rate, 20.0, 3000.0);
        let delay = 1000;
        let echo = 800;
        let mut take = delayed(&reference, delay + echo, rate);
        for (index, &sample) in reference.iter().enumerate() {
            take[delay + index] += sample;
        }
        assert!(estimate_sweep_lag_with_confidence(&reference, &take, rate).is_err());
    }

    #[test]
    fn sweep_lock_ignores_a_loud_partial_tail_copy() {
        let rate = 8000;
        let reference = sweep(rate, 20.0, 3000.0);
        let delay = 1000;
        let copy_start = delay + reference.len() + 2000;
        let mut take = delayed(&reference, delay, rate);
        take.resize(copy_start + reference.len() / 4, 0.0);
        for index in 0..reference.len() / 4 {
            take[copy_start + index] = reference[index] * 1.2;
        }
        let lag = estimate_sweep_lag_with_confidence(&reference, &take, rate).unwrap();
        assert_eq!(lag.lag_samples, delay as isize);
    }
}

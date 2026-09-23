use criterion::{Criterion, criterion_group, criterion_main};
use math_audio_dsp::analysis::{FiniteWindowFirConfig, FiniteWindowFirObjective};
use std::hint::black_box;

fn fixture() -> (Vec<f64>, FiniteWindowFirConfig, Vec<f64>) {
    let impulse: Vec<f64> = (0..16_384)
        .map(|i| {
            if i < 100 {
                0.0
            } else {
                (-((i - 100) as f64) / 1800.0).exp() * (i as f64 * 0.13).cos()
            }
        })
        .collect();
    let config = FiniteWindowFirConfig {
        capture_rate_hz: 48_000,
        filter_rate_hz: 48_000,
        tap_count: 512,
        anchor_sample: 100,
        frequencies_hz: vec![35.0, 45.0, 60.0, 80.0, 100.0],
        frequency_weights: vec![1.0; 5],
        window_seconds: 0.064,
        starts_seconds: vec![0.0, 0.08, 0.14],
    };
    let mut taps = vec![0.0; 512];
    taps[0] = 1.0;
    taps[100] = -0.12;
    (impulse, config, taps)
}

fn benchmark(c: &mut Criterion) {
    let (impulse, config, taps) = fixture();
    c.bench_function("finite_window_fir_prepare", |b| {
        b.iter(|| FiniteWindowFirObjective::new(black_box(&impulse), black_box(&config)).unwrap())
    });
    let objective = FiniteWindowFirObjective::new(&impulse, &config).unwrap();
    c.bench_function("finite_window_fir_energy", |b| {
        b.iter(|| objective.energy(black_box(&taps)).unwrap())
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

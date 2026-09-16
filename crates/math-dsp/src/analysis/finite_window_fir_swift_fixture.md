# Swift comparison fixture

`finite_window_fir_swift_fixture.json` is synthetic public test data generated with Open Room Calibration's `FiniteWindowFIRObjective.swift` (Apache-2.0; see `../../LICENSE-APACHE-2.0`). No room measurements are included. The Swift implementation selected sample 7 as the peak; Rust supplies that same anchor explicitly. Swift `Float` impulse and taps were serialized as exact `Double` values, and frequency weights are all one because Swift has no weighting input.

- Swift source SHA-256: `0b79250f99c37167d65f68e94827cafe1c81163d55aa824e44b45ae8e17cb0b6`
- Fixture JSON SHA-256: `0c5c626b6f35c2b02da88d8a3101a30e3f62755f18153888e09ec11218d37b37`
- Comparison tolerance: energy absolute error at most `1e-10 * max(expected, 1)`; dB absolute error below `1e-9`.

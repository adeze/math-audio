//! Near-field source multilateration from microphone ranges.
//!
//! Spherical-wave solver: complements the far-field plane-wave direction
//! finding in [`crate::capture_array`]. Given surveyed microphone positions and
//! per-microphone ranges (distance plus sigma, e.g. from chirp arrival times),
//! [`solve_position`] estimates the 3D source position with per-axis
//! uncertainty, residual diagnostics, and dilution of precision.
//!
//! Deterministic: fixed iteration order, no randomness, `f64` throughout.
//! Total: every input maps to `Some` estimate or `None` (refusal); no panics.
//!
//! Three microphones admit a mirror pair across the microphone plane; neither
//! solution is published ([`solve_position`] returns `None`) and the caller is
//! expected to flag the case via [`geometry_rank`]. Four or more coplanar
//! microphones keep the mirror ambiguity but a position IS published with
//! [`PositionEstimate::mirror_ambiguous`] set; the published mirror is
//! deterministic for a given build yet carries no physical preference.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};

/// Coplanarity/collinearity threshold: a centred-scatter singular value at or
/// below 1 mm counts as a collapsed dimension.
pub const GEOMETRY_SINGULAR_THRESHOLD_M: f64 = 1.0e-3;
/// Minimum usable ranges for a 3D fix. Three ranges admit an unpublished
/// mirror pair; fewer constrain nothing.
pub const MIN_RANGES_FOR_FIX: usize = 4;
/// Huber transition in per-cell sigma multiples (IRLS downweighting).
pub const HUBER_SIGMA_MULTIPLE: f64 = 3.0;
/// Post-fit exclusion in per-cell sigma multiples; the solve repeats once
/// without excluded cells.
pub const OUTLIER_SIGMA_MULTIPLE: f64 = 3.5;
/// Fixed Levenberg damping added to the Gauss-Newton normal matrix. Also
/// regularises the reported covariance: in an unobservable direction the
/// variance saturates at ~1/`LEVENBERG_DAMPING` (sigma ~3e4 m), which reads as
/// "unconstrained" rather than a physical value.
pub const LEVENBERG_DAMPING: f64 = 1.0e-9;
/// Gauss-Newton step-norm convergence threshold (m).
pub const CONVERGENCE_STEP_M: f64 = 1.0e-9;
/// Gauss-Newton iteration cap per seed; exhaustion fails the seed.
pub const MAX_REFINEMENT_ITERATIONS: usize = 100;
/// A modelled source-to-microphone distance at or below this collapses the
/// Jacobian direction; refinement touching it fails the seed.
const MIN_MODEL_DISTANCE_M: f64 = 1.0e-12;

/// Rank classification of a microphone point set.
///
/// Classify the microphones that actually carry usable ranges, not the full
/// plan: only used microphones constrain the solve. A three-microphone set is
/// always `coplanar` (three points define a plane); the flag uniformly means
/// "elevation sign unobservable".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryRank {
    /// Centred-scatter singular values above
    /// [`GEOMETRY_SINGULAR_THRESHOLD_M`] (0..=3).
    pub rank: usize,
    /// `rank == 2`: microphones coplanar, mirror ambiguity across the plane.
    pub coplanar: bool,
    /// `rank <= 1`: collinear or duplicated, no position solvable.
    pub collinear: bool,
}

/// Leave-one-out cross-validation of a solved position.
///
/// For every used microphone, the position is re-solved without that
/// microphone's range; the held-out residual measures how well the
/// remaining geometry predicts the dropped range, and the shift measures
/// how much the solution depended on it. A large shift flags leverage
/// (or a mirror jump in ambiguous geometry), even when residuals look
/// small.
#[derive(Debug, Clone)]
pub struct LooDiagnostics {
    /// Largest held-out residual in metres (`|model - measured|`).
    pub max_residual_m: f64,
    /// Input index of the microphone with the largest residual.
    pub worst_index: usize,
    /// Largest solution shift in metres over the re-solves.
    pub max_shift_m: f64,
}

/// Leave-one-out diagnostics for a solved position.
///
/// Re-solves once per used cell of `full` with that cell dropped;
/// depleting drops (fewer than [`MIN_RANGES_FOR_FIX`] left) are
/// skipped. Returns `None` when no re-solve succeeds — notably with
/// exactly four used ranges, where every drop depletes. Total on all
/// inputs; deterministic like the solver.
pub fn loo_diagnostics(
    microphones_m: &[[f64; 3]],
    ranges_m: &[Option<(f64, f64)>],
    full: &PositionEstimate,
) -> Option<LooDiagnostics> {
    if microphones_m.len() != ranges_m.len()
        || microphones_m.is_empty()
        || full.used.len() != ranges_m.len()
    {
        return None;
    }
    let mut ranges = ranges_m.to_vec();
    let mut worst: Option<(usize, f64)> = None;
    let mut max_shift_m = 0.0_f64;
    for (index, used) in full.used.iter().enumerate() {
        if !used {
            continue;
        }
        let held = ranges[index].take();
        let re_solved = solve_position(microphones_m, &ranges);
        ranges[index] = held;
        let Some(re_fit) = re_solved else {
            continue;
        };
        let Some((distance, _)) = ranges_m[index] else {
            continue;
        };
        let dx = re_fit.position_m[0] - microphones_m[index][0];
        let dy = re_fit.position_m[1] - microphones_m[index][1];
        let dz = re_fit.position_m[2] - microphones_m[index][2];
        let residual = ((dx * dx + dy * dy + dz * dz).sqrt() - distance).abs();
        if !residual.is_finite() {
            continue;
        }
        let ex = re_fit.position_m[0] - full.position_m[0];
        let ey = re_fit.position_m[1] - full.position_m[1];
        let ez = re_fit.position_m[2] - full.position_m[2];
        let shift = (ex * ex + ey * ey + ez * ez).sqrt();
        if !shift.is_finite() {
            continue;
        }
        if shift > max_shift_m {
            max_shift_m = shift;
        }
        let more_residual = match worst {
            None => true,
            Some((_, held_residual)) => residual > held_residual,
        };
        if more_residual {
            worst = Some((index, residual));
        }
    }
    worst.map(|(worst_index, max_residual_m)| LooDiagnostics {
        max_residual_m,
        worst_index,
        max_shift_m,
    })
}

/// Solved source position with uncertainty and diagnostics.
#[derive(Debug, Clone)]
pub struct PositionEstimate {
    /// Source position in metres (plan frame).
    pub position_m: [f64; 3],
    /// Per-axis 1-sigma uncertainty in metres, from the damped normal-matrix
    /// inverse at the fixpoint. Unobservable directions saturate at ~3e4 m
    /// (see [`LEVENBERG_DAMPING`]); treat sigma above 1 m as unconstrained.
    pub sigma_m: [f64; 3],
    /// Unweighted RMS residual over used cells (m).
    pub rms_residual_m: f64,
    /// `sqrt(trace(covariance)) / median(used sigma)`; dimensionless.
    /// Near-degenerate geometry inflates this by orders of magnitude.
    pub gdop: f64,
    /// True when the microphone geometry cannot resolve the elevation sign
    /// (coplanar microphones): the mirror across the microphone plane fits
    /// equally well.
    pub mirror_ambiguous: bool,
    /// Aligned with the input ranges: false for missing, invalid, or
    /// outlier-excluded cells.
    pub used: Vec<bool>,
}

/// Classify microphone geometry via the centred-scatter eigenstructure.
///
/// Non-finite positions are ignored. Empty (or fully non-finite) input yields
/// rank 0 with `collinear` set.
pub fn geometry_rank(positions_m: &[[f64; 3]]) -> GeometryRank {
    classify(positions_m).rank
}

/// Solve the 3D source position from microphone ranges.
///
/// `microphones_m[i]` is the surveyed microphone position in metres;
/// `ranges_m[i]` is `Some((distance_m, sigma_m))` or `None` when the range was
/// refused upstream. Cells with non-finite microphone positions,
/// negative/non-finite distances, or non-positive/non-finite sigmas are
/// excluded (`used[i] == false`).
///
/// Returns `None` when the inputs disagree in length or are empty, fewer than
/// [`MIN_RANGES_FOR_FIX`] ranges are usable, the usable microphones are
/// collinear, every seed diverges, outlier exclusion depletes the cell set, or
/// the final covariance is not invertible. Coplanar (non-collinear) geometry
/// DOES solve; the published mirror is deterministic but arbitrary and
/// `mirror_ambiguous` is set.
pub fn solve_position(
    microphones_m: &[[f64; 3]],
    ranges_m: &[Option<(f64, f64)>],
) -> Option<PositionEstimate> {
    if microphones_m.len() != ranges_m.len() || microphones_m.is_empty() {
        return None;
    }
    let mut used = vec![true; ranges_m.len()];
    for (index, range) in ranges_m.iter().enumerate() {
        let microphone_finite = microphones_m[index].iter().all(|c| c.is_finite());
        used[index] = microphone_finite
            && matches!(range, Some((distance, sigma))
            if distance.is_finite() && *distance >= 0.0 && sigma.is_finite() && *sigma > 0.0);
    }
    solve_cells(microphones_m, ranges_m, &mut used, true)
}

/// Usable cells gathered for one solve attempt.
struct Cells {
    /// Original input indices, parallel to the vectors below.
    index: Vec<usize>,
    mics: Vec<[f64; 3]>,
    dist: Vec<f64>,
    sig: Vec<f64>,
}

impl Cells {
    /// Gather cells flagged in `used` with valid range payloads. Total: cells
    /// whose payload disagrees with the flag are skipped, never unwrapped.
    fn gather(microphones_m: &[[f64; 3]], ranges_m: &[Option<(f64, f64)>], used: &[bool]) -> Self {
        let mut cells = Cells {
            index: Vec::new(),
            mics: Vec::new(),
            dist: Vec::new(),
            sig: Vec::new(),
        };
        for (index, range) in ranges_m.iter().enumerate() {
            if !used[index] {
                continue;
            }
            if let Some((distance, sigma)) = range {
                cells.index.push(index);
                cells.mics.push(microphones_m[index]);
                cells.dist.push(*distance);
                cells.sig.push(*sigma);
            }
        }
        cells
    }

    fn len(&self) -> usize {
        self.dist.len()
    }
}

/// Geometry classification with centroid and least-significant axis.
struct Classification {
    rank: GeometryRank,
    centroid: Vector3<f64>,
    /// Unit eigenvector of the smallest scatter eigenvalue: the plane normal
    /// when coplanar. Zero vector when no finite position exists (unused: the
    /// rank is collinear there and the solve refuses before seeding).
    min_axis: Vector3<f64>,
}

/// Centroid, scatter eigenstructure, and rank. Total on all inputs.
fn classify(positions_m: &[[f64; 3]]) -> Classification {
    let degenerate = Classification {
        rank: GeometryRank {
            rank: 0,
            coplanar: false,
            collinear: true,
        },
        centroid: Vector3::zeros(),
        min_axis: Vector3::zeros(),
    };
    let mut count = 0_usize;
    let mut centroid = Vector3::zeros();
    for position in positions_m
        .iter()
        .filter(|p| p.iter().all(|c| c.is_finite()))
    {
        count += 1;
        centroid += Vector3::new(position[0], position[1], position[2]);
    }
    if count == 0 {
        return degenerate;
    }
    centroid /= count as f64;
    let mut scatter = Matrix3::zeros();
    for position in positions_m
        .iter()
        .filter(|p| p.iter().all(|c| c.is_finite()))
    {
        let delta = Vector3::new(
            position[0] - centroid[0],
            position[1] - centroid[1],
            position[2] - centroid[2],
        );
        scatter += delta * delta.transpose();
    }
    let eigen = SymmetricEigen::new(scatter);
    let mut rank_count = 0_usize;
    let mut min_index = 0_usize;
    for (index, lambda) in eigen.eigenvalues.iter().enumerate() {
        if lambda.max(0.0).sqrt() > GEOMETRY_SINGULAR_THRESHOLD_M {
            rank_count += 1;
        }
        if *lambda < eigen.eigenvalues[min_index] {
            min_index = index;
        }
    }
    Classification {
        rank: GeometryRank {
            rank: rank_count,
            coplanar: rank_count == 2,
            collinear: rank_count <= 1,
        },
        centroid,
        min_axis: Vector3::new(
            eigen.eigenvectors[(0, min_index)],
            eigen.eigenvectors[(1, min_index)],
            eigen.eigenvectors[(2, min_index)],
        ),
    }
}

/// Solve over the flagged cells; on outlier exclusion recurse once without the
/// excluded cells (`allow_exclusion` false on the second pass).
fn solve_cells(
    microphones_m: &[[f64; 3]],
    ranges_m: &[Option<(f64, f64)>],
    used: &mut [bool],
    allow_exclusion: bool,
) -> Option<PositionEstimate> {
    let cells = Cells::gather(microphones_m, ranges_m, used);
    if cells.len() < MIN_RANGES_FOR_FIX {
        return None;
    }
    let class = classify(&cells.mics);
    if class.rank.collinear {
        return None;
    }
    let mut seeds = Vec::with_capacity(2);
    if class.rank.coplanar {
        let mean_distance = cells.dist.iter().sum::<f64>() / cells.len() as f64;
        seeds.push(class.centroid + class.min_axis * mean_distance);
        seeds.push(class.centroid - class.min_axis * mean_distance);
    } else {
        if let Some(linear) = linear_init(&cells) {
            seeds.push(linear);
        }
        seeds.push(centroid_init(&cells));
    }
    let mut best: Option<(Vector3<f64>, f64)> = None;
    for seed in seeds {
        if let Some((fixpoint, cost)) = refine(seed, &cells) {
            let better = match &best {
                None => true,
                Some((_, best_cost)) => cost < *best_cost,
            };
            if better {
                best = Some((fixpoint, cost));
            }
        }
    }
    let (position, _) = best?;
    if allow_exclusion {
        let residuals = residuals_at(&position, &cells)?;
        let mut excluded = false;
        for (cell, residual) in residuals.iter().enumerate() {
            if residual.abs() > OUTLIER_SIGMA_MULTIPLE * cells.sig[cell] {
                used[cells.index[cell]] = false;
                excluded = true;
            }
        }
        if excluded {
            return solve_cells(microphones_m, ranges_m, used, false);
        }
    }
    finish(position, &cells, class.rank.coplanar, used)
}

/// Closed-form seed from differenced squared ranges (exact on noiseless data).
/// Weighted least squares against the first cell; `None` when singular.
fn linear_init(cells: &Cells) -> Option<Vector3<f64>> {
    let reference = cells.mics[0];
    let reference_norm_sq =
        reference[0] * reference[0] + reference[1] * reference[1] + reference[2] * reference[2];
    let reference_dist_sq = cells.dist[0] * cells.dist[0];
    let mut normal = Matrix3::zeros();
    let mut target = Vector3::zeros();
    for cell in 1..cells.len() {
        let mic = cells.mics[cell];
        let row = Vector3::new(
            2.0 * (mic[0] - reference[0]),
            2.0 * (mic[1] - reference[1]),
            2.0 * (mic[2] - reference[2]),
        );
        let rhs = (mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2])
            - reference_norm_sq
            - (cells.dist[cell] * cells.dist[cell] - reference_dist_sq);
        let weight = 1.0 / (cells.sig[cell] * cells.sig[cell]);
        normal += (row * row.transpose()) * weight;
        target += row * (rhs * weight);
    }
    normal.try_inverse().map(|inverse| inverse * target)
}

/// Fallback seed: uncertainty-weighted microphone centroid pulled outward
/// along the mic-0-to-centroid axis by the reference range. Total.
fn centroid_init(cells: &Cells) -> Vector3<f64> {
    let mut weight_sum = 0.0_f64;
    let mut centroid = Vector3::zeros();
    for cell in 0..cells.len() {
        let weight = 1.0 / (cells.sig[cell] * cells.sig[cell]);
        weight_sum += weight;
        centroid += Vector3::new(
            cells.mics[cell][0] * weight,
            cells.mics[cell][1] * weight,
            cells.mics[cell][2] * weight,
        );
    }
    if weight_sum.is_finite() && weight_sum > 0.0 {
        centroid /= weight_sum;
    } else {
        centroid = Vector3::zeros();
        for mic in &cells.mics {
            centroid += Vector3::new(mic[0], mic[1], mic[2]);
        }
        centroid /= cells.len() as f64;
    }
    let origin = Vector3::new(cells.mics[0][0], cells.mics[0][1], cells.mics[0][2]);
    let axis = centroid - origin;
    let axis_norm = axis.norm();
    if axis_norm > 1.0e-9 {
        origin + axis * (cells.dist[0] / axis_norm)
    } else {
        origin + Vector3::new(cells.dist[0], 0.0, 0.0)
    }
}

/// Huber-IRLS Gauss-Newton from one seed. Returns the fixpoint and its robust
/// cost, or `None` when a modelled distance collapses, the step diverges, the
/// normal matrix stays singular under damping, or the iteration cap exhausts.
fn refine(seed: Vector3<f64>, cells: &Cells) -> Option<(Vector3<f64>, f64)> {
    if !seed.iter().all(|c| c.is_finite()) {
        return None;
    }
    let mut position = seed;
    let mut converged = false;
    for _ in 0..MAX_REFINEMENT_ITERATIONS {
        let (normal, gradient) = weighted_normal_system(&position, cells)?;
        let damped = normal + Matrix3::identity() * LEVENBERG_DAMPING;
        let step = damped.try_inverse().map(|inverse| inverse * (-gradient))?;
        position += step;
        if !position.iter().all(|c| c.is_finite()) || position.norm() > 1.0e9 {
            return None;
        }
        if step.norm() < CONVERGENCE_STEP_M {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }
    robust_cost(&position, cells).map(|cost| (position, cost))
}

/// Weighted normal system `(J'WJ, J'Wr)` with Huber IRLS weights. `None` when
/// a modelled source-to-microphone distance collapses.
fn weighted_normal_system(
    position: &Vector3<f64>,
    cells: &Cells,
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let mut normal = Matrix3::zeros();
    let mut gradient = Vector3::zeros();
    for cell in 0..cells.len() {
        let jacobian = unit_jacobian(position, &cells.mics[cell])?;
        let residual = model_distance(position, &cells.mics[cell]) - cells.dist[cell];
        let sigma = cells.sig[cell];
        let huber = (HUBER_SIGMA_MULTIPLE * sigma / residual.abs()).min(1.0);
        let weight = huber / (sigma * sigma);
        normal += (jacobian * jacobian.transpose()) * weight;
        gradient += jacobian * (residual * weight);
    }
    Some((normal, gradient))
}

/// Huber robust cost, sigma-normalised. `None` on collapsed model distance.
fn robust_cost(position: &Vector3<f64>, cells: &Cells) -> Option<f64> {
    let mut cost = 0.0_f64;
    for cell in 0..cells.len() {
        let residual = (model_distance(position, &cells.mics[cell]) - cells.dist[cell]).abs();
        if !residual.is_finite() {
            return None;
        }
        let delta = HUBER_SIGMA_MULTIPLE * cells.sig[cell];
        let rho = if residual <= delta {
            0.5 * residual * residual
        } else {
            delta * (residual - 0.5 * delta)
        };
        cost += rho / (cells.sig[cell] * cells.sig[cell]);
    }
    Some(cost)
}

/// Residuals (`model - measured`) over all cells. `None` on collapse.
fn residuals_at(position: &Vector3<f64>, cells: &Cells) -> Option<Vec<f64>> {
    let mut residuals = Vec::with_capacity(cells.len());
    for cell in 0..cells.len() {
        let distance = model_distance(position, &cells.mics[cell]);
        if distance.is_nan() || distance <= MIN_MODEL_DISTANCE_M {
            return None;
        }
        residuals.push(distance - cells.dist[cell]);
    }
    Some(residuals)
}

/// Euclidean model distance; callers treat values at/below
/// [`MIN_MODEL_DISTANCE_M`] (and NaN) as collapsed.
fn model_distance(position: &Vector3<f64>, mic: &[f64; 3]) -> f64 {
    let dx = position[0] - mic[0];
    let dy = position[1] - mic[1];
    let dz = position[2] - mic[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Unit Jacobian row (source-to-microphone direction). `None` on collapse.
fn unit_jacobian(position: &Vector3<f64>, mic: &[f64; 3]) -> Option<Vector3<f64>> {
    let distance = model_distance(position, mic);
    if distance.is_nan() || distance <= MIN_MODEL_DISTANCE_M {
        return None;
    }
    Some(Vector3::new(
        (position[0] - mic[0]) / distance,
        (position[1] - mic[1]) / distance,
        (position[2] - mic[2]) / distance,
    ))
}

/// Assemble the estimate: damped covariance, per-axis sigma, GDOP, RMS.
fn finish(
    position: Vector3<f64>,
    cells: &Cells,
    mirror_ambiguous: bool,
    used: &[bool],
) -> Option<PositionEstimate> {
    let (normal, _) = weighted_normal_system(&position, cells)?;
    let damped = normal + Matrix3::identity() * LEVENBERG_DAMPING;
    let covariance = damped.try_inverse()?;
    let mut sigma_m = [0.0_f64; 3];
    for axis in 0..3 {
        sigma_m[axis] = covariance[(axis, axis)].max(0.0).sqrt();
    }
    let trace = covariance[(0, 0)] + covariance[(1, 1)] + covariance[(2, 2)];
    let mut sorted_sigmas = cells.sig.clone();
    sorted_sigmas.sort_by(|a, b| a.total_cmp(b));
    let count = sorted_sigmas.len();
    let median_sigma = if count % 2 == 1 {
        sorted_sigmas[count / 2]
    } else {
        0.5 * (sorted_sigmas[count / 2 - 1] + sorted_sigmas[count / 2])
    };
    let residuals = residuals_at(&position, cells)?;
    let rms_residual_m = (residuals.iter().map(|r| r * r).sum::<f64>() / count as f64).sqrt();
    Some(PositionEstimate {
        position_m: [position[0], position[1], position[2]],
        sigma_m,
        rms_residual_m,
        gdop: trace.max(0.0).sqrt() / median_sigma,
        mirror_ambiguous,
        used: used.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn exact_ranges(mics: &[[f64; 3]], source: &[f64; 3], sigma: f64) -> Vec<Option<(f64, f64)>> {
        mics.iter()
            .map(|mic| Some((distance(mic, source), sigma)))
            .collect()
    }

    /// Unit-cube corners: full-rank 8-microphone array.
    fn cube_mics() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    }

    #[test]
    fn tetrahedron_recovers_exact_position() {
        let mics = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let source = [2.3, -1.1, 0.7];
        let ranges = exact_ranges(&mics, &source, 0.001);
        let estimate = solve_position(&mics, &ranges).expect("tetrahedron solves");
        for axis in 0..3 {
            assert!(
                (estimate.position_m[axis] - source[axis]).abs() < 1.0e-9,
                "axis {axis}: {} vs {}",
                estimate.position_m[axis],
                source[axis]
            );
        }
        assert!(!estimate.mirror_ambiguous);
        assert!(estimate.used.iter().all(|u| *u));
        assert!(estimate.rms_residual_m < 1.0e-9);
        assert!(estimate.gdop.is_finite() && estimate.gdop > 0.0);
        assert!(estimate.sigma_m.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn noisy_ranges_recover_within_sigma() {
        let mics = cube_mics();
        let source = [2.3, -1.1, 0.7];
        // Deterministic sub-sigma offsets (mm); no RNG in tests.
        let offsets_mm = [1.2, -0.8, 0.5, -1.5, 0.9, -0.3, 1.0, -0.6];
        let sigma = 0.002;
        let ranges: Vec<Option<(f64, f64)>> = mics
            .iter()
            .zip(offsets_mm.iter())
            .map(|(mic, offset)| Some((distance(mic, &source) + offset / 1000.0, sigma)))
            .collect();
        let estimate = solve_position(&mics, &ranges).expect("noisy cube solves");
        let error = distance(&estimate.position_m, &source);
        assert!(error < 3.0 * sigma, "error {error}");
        assert!(estimate.rms_residual_m < 3.0 * sigma);
        for axis in 0..3 {
            assert!(
                estimate.sigma_m[axis] > 0.1 * sigma && estimate.sigma_m[axis] < 10.0 * sigma,
                "sigma[{}] = {}",
                axis,
                estimate.sigma_m[axis]
            );
        }
    }

    #[test]
    fn three_mics_return_none_with_mirror_rank() {
        let mics = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let source = [0.3, 0.4, 1.2];
        let ranges = exact_ranges(&mics, &source, 0.001);
        assert!(solve_position(&mics, &ranges).is_none());
        let rank = geometry_rank(&mics);
        assert_eq!(rank.rank, 2);
        assert!(rank.coplanar);
        assert!(!rank.collinear);
    }

    #[test]
    fn coplanar_mics_publish_with_mirror_flag() {
        let mics = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        let source = [1.0, 0.5, 1.5];
        let ranges = exact_ranges(&mics, &source, 0.001);
        let estimate = solve_position(&mics, &ranges).expect("coplanar solves");
        assert!(estimate.mirror_ambiguous);
        assert!((estimate.position_m[0] - source[0]).abs() < 1.0e-6);
        assert!((estimate.position_m[1] - source[1]).abs() < 1.0e-6);
        assert!((estimate.position_m[2].abs() - source[2]).abs() < 1.0e-6);
        assert!(estimate.rms_residual_m < 1.0e-6);
    }

    #[test]
    fn collinear_mics_return_none() {
        let mics = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let source = [0.5, 1.0, 0.0];
        let ranges = exact_ranges(&mics, &source, 0.001);
        assert!(solve_position(&mics, &ranges).is_none());
        let rank = geometry_rank(&mics);
        assert!(rank.collinear);
        assert!(!rank.coplanar);
    }

    #[test]
    fn outlier_is_excluded_and_solve_repeats() {
        let mics = cube_mics();
        let source = [2.3, -1.1, 0.7];
        let sigma = 0.001;
        let mut ranges = exact_ranges(&mics, &source, sigma);
        ranges[5] = Some((ranges[5].expect("exact range").0 + 0.05, sigma));
        let estimate = solve_position(&mics, &ranges).expect("outlier excluded");
        assert!(!estimate.used[5]);
        assert_eq!(estimate.used.iter().filter(|u| **u).count(), 7);
        let error = distance(&estimate.position_m, &source);
        assert!(error < 1.0e-6, "error {error}");
    }

    #[test]
    fn outlier_depletion_returns_none() {
        let mics = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let source = [2.3, -1.1, 0.7];
        let sigma = 0.001;
        let mut ranges = exact_ranges(&mics, &source, sigma);
        ranges[2] = Some((ranges[2].expect("exact range").0 + 0.05, sigma));
        assert!(solve_position(&mics, &ranges).is_none());
    }

    #[test]
    fn invalid_cells_are_excluded() {
        let source = [2.3, -1.1, 0.7];
        let mut ranges = exact_ranges(&cube_mics(), &source, 0.001);
        let mut mics = cube_mics();
        mics[0] = [f64::NAN, 0.0, 0.0];
        ranges[1] = None;
        ranges[2] = Some((f64::NAN, 0.001));
        ranges[3] = Some((-1.0, 0.001));
        let estimate = solve_position(&mics, &ranges).expect("four valid cells solve");
        assert!(!estimate.used[0]);
        assert!(!estimate.used[1]);
        assert!(!estimate.used[2]);
        assert!(!estimate.used[3]);
        let error = distance(&estimate.position_m, &source);
        assert!(error < 1.0e-9, "error {error}");

        let mut bad_sigma = exact_ranges(&cube_mics(), &source, 0.001);
        bad_sigma[0] = Some((distance(&cube_mics()[0], &source), 0.0));
        bad_sigma[1] = Some((distance(&cube_mics()[1], &source), -0.001));
        bad_sigma[2] = Some((distance(&cube_mics()[2], &source), f64::NAN));
        let estimate = solve_position(&cube_mics(), &bad_sigma).expect("five valid solve");
        assert!(!estimate.used[0]);
        assert!(!estimate.used[1]);
        assert!(!estimate.used[2]);
    }

    #[test]
    fn length_mismatch_and_empty_return_none() {
        let mics = vec![[0.0, 0.0, 0.0]];
        let ranges = vec![Some((1.0, 0.001)), Some((2.0, 0.001))];
        assert!(solve_position(&mics, &ranges).is_none());
        let empty: Vec<[f64; 3]> = Vec::new();
        let empty_ranges: Vec<Option<(f64, f64)>> = Vec::new();
        assert!(solve_position(&empty, &empty_ranges).is_none());
    }

    #[test]
    fn repeated_calls_are_bit_identical() {
        let mics = cube_mics();
        let source = [2.3, -1.1, 0.7];
        let ranges = exact_ranges(&mics, &source, 0.001);
        let first = solve_position(&mics, &ranges).expect("solves");
        let second = solve_position(&mics, &ranges).expect("solves");
        assert_eq!(first.position_m, second.position_m);
        assert_eq!(first.sigma_m, second.sigma_m);
        assert_eq!(first.rms_residual_m, second.rms_residual_m);
        assert_eq!(first.gdop, second.gdop);
    }

    /// Fixed-seed xorshift64 for the Monte-Carlo test; deterministic with
    /// no RNG dependency.
    struct FixedRng(u64);

    impl FixedRng {
        fn next_unit(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            // Low 32 bits scaled into (0, 1); never exactly 0 or 1.
            f64::from((x >> 11) as u32) / f64::from(u32::MAX) * 0.999_999 + 0.000_000_5
        }

        fn next_gaussian(&mut self) -> f64 {
            use std::f64::consts::PI;
            let (u1, u2) = (self.next_unit(), self.next_unit());
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }

    #[test]
    fn monte_carlo_errors_match_predicted_sigma() {
        // §13.1: 5 mm Gaussian range noise, 200 fixed-seed trials. The
        // empirical 3σ coverage must stay near nominal (joint 3-axis
        // ≈ 99%) and the mean predicted σ within 2× of empirical RMS:
        // the covariance has to be honest, not merely large.
        let mics = cube_mics();
        let source = [2.3, -1.1, 0.7];
        let sigma = 0.005;
        let mut rng = FixedRng(0x1_2345_6789_ABCD_EF);
        let mut covered = 0_usize;
        let mut sum_sq_err = [0.0_f64; 3];
        let mut sum_sigma = [0.0_f64; 3];
        for _ in 0..200 {
            let ranges: Vec<Option<(f64, f64)>> = mics
                .iter()
                .map(|mic| Some((distance(mic, &source) + rng.next_gaussian() * sigma, sigma)))
                .collect();
            let estimate = solve_position(&mics, &ranges).expect("noisy trial solves");
            let mut inside = true;
            for axis in 0..3 {
                let error = estimate.position_m[axis] - source[axis];
                sum_sq_err[axis] += error * error;
                sum_sigma[axis] += estimate.sigma_m[axis];
                inside &= error.abs() <= 3.0 * estimate.sigma_m[axis];
            }
            covered += usize::from(inside);
        }
        assert!(
            covered >= 195,
            "3σ coverage {covered}/200 below the 195 bound"
        );
        for axis in 0..3 {
            let rms = (sum_sq_err[axis] / 200.0).sqrt();
            let mean_sigma = sum_sigma[axis] / 200.0;
            let ratio = rms / mean_sigma;
            assert!(
                (0.5..=2.0).contains(&ratio),
                "axis {axis}: rms {rms} vs mean σ {mean_sigma} (ratio {ratio})"
            );
        }
    }

    #[test]
    fn loo_flags_influential_range() {
        let mics = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let source = [2.3, -1.1, 0.7];
        let sigma = 0.001;
        let mut ranges = exact_ranges(&mics, &source, sigma);
        ranges[4] = Some((ranges[4].expect("exact range").0 + 3.0 * sigma, sigma));
        let full = solve_position(&mics, &ranges).expect("six ranges solve");
        assert!(full.used.iter().all(|used| *used));
        let loo = loo_diagnostics(&mics, &ranges, &full).expect("spare cells diagnose");
        assert_eq!(loo.worst_index, 4);
        assert!(
            loo.max_residual_m > 2.0 * sigma,
            "residual {}",
            loo.max_residual_m
        );
        assert!(loo.max_shift_m.is_finite() && loo.max_shift_m > 0.0);
    }

    #[test]
    fn loo_empty_without_spare_cells() {
        let mics = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let source = [2.3, -1.1, 0.7];
        let ranges = exact_ranges(&mics, &source, 0.001);
        let full = solve_position(&mics, &ranges).expect("tetra solves");
        assert!(loo_diagnostics(&mics, &ranges, &full).is_none());
    }

    #[test]
    fn loo_rejects_mismatched_inputs() {
        let mics = cube_mics();
        let source = [2.3, -1.1, 0.7];
        let ranges = exact_ranges(&mics, &source, 0.001);
        let full = solve_position(&mics, &ranges).expect("cube solves");
        assert!(loo_diagnostics(&mics[..4], &ranges, &full).is_none());
        let short = exact_ranges(&mics[..4], &source, 0.001);
        assert!(loo_diagnostics(&mics, &short, &full).is_none());
        let mut short_used = full.clone();
        short_used.used.pop();
        assert!(loo_diagnostics(&mics, &ranges, &short_used).is_none());
    }

    #[test]
    fn geometry_rank_handles_degenerate_sets() {
        let empty: Vec<[f64; 3]> = Vec::new();
        assert!(geometry_rank(&empty).collinear);
        let duplicated = vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let rank = geometry_rank(&duplicated);
        assert_eq!(rank.rank, 0);
        assert!(rank.collinear);
        let pair = vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let rank = geometry_rank(&pair);
        assert_eq!(rank.rank, 1);
        assert!(rank.collinear);
        let rank = geometry_rank(&cube_mics());
        assert_eq!(rank.rank, 3);
        assert!(!rank.coplanar);
        assert!(!rank.collinear);
    }

    #[test]
    fn near_plane_source_inflates_normal_sigma() {
        // Coplanar microphones: the mirror ambiguity is global (two wells),
        // so a well-off-plane source keeps a tight local covariance while a
        // near-plane source flattens the normal curvature.
        let mics = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ];
        let sigma = 0.001;
        let high = [1.0, 0.5, 1.5];
        let high_est =
            solve_position(&mics, &exact_ranges(&mics, &high, sigma)).expect("off-plane solves");
        assert!(high_est.mirror_ambiguous);
        assert!(
            high_est.sigma_m[2] < 10.0 * sigma,
            "{}",
            high_est.sigma_m[2]
        );
        let low = [1.0, 0.5, 0.002];
        let low_est =
            solve_position(&mics, &exact_ranges(&mics, &low, sigma)).expect("near-plane solves");
        assert!(low_est.mirror_ambiguous);
        assert!(
            low_est.sigma_m[2] > 100.0 * low_est.sigma_m[0],
            "normal sigma {} vs in-plane {}",
            low_est.sigma_m[2],
            low_est.sigma_m[0]
        );
    }
}

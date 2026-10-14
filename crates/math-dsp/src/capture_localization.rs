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

use nalgebra::{DMatrix, DVector, Matrix3, SymmetricEigen, Vector3};

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

/// Maximum joint iterations before the calibration refuses.
pub const MAX_JOINT_ITERATIONS: usize = 100;
/// Joint convergence: max accepted step over all unknowns (m).
pub const JOINT_CONVERGENCE_STEP_M: f64 = 1.0e-9;
/// Damping retries per joint iteration before the calibration refuses.
pub const MAX_JOINT_LAMBDA_BUMPS: usize = 20;
/// Minimum surveyed anchors fixing the joint frame.
pub const MIN_JOINT_ANCHORS: usize = 3;

/// One jointly calibrated point (unsurveyed source or microphone).
#[derive(Debug, Clone)]
pub struct JointPoint {
    /// Calibrated position in metres (anchor frame).
    pub position_m: [f64; 3],
    /// Marginal per-axis sigma from the joint covariance: every unknown
    /// stays free, so this is never conditioned on estimated anchors.
    pub sigma_m: [f64; 3],
    /// RMS residual over this point's used cross cells (m).
    pub rms_residual_m: f64,
    /// Median sigma over this point's used cross cells (m); the Q7 input.
    pub median_sigma_m: f64,
}

/// Joint self-calibration of unsurveyed sources and microphones.
#[derive(Debug, Clone)]
pub struct JointEstimate {
    /// One entry per source seed; `None` for surveyed anchors.
    pub source_points: Vec<Option<JointPoint>>,
    /// One entry per microphone seed; `None` for surveyed anchors.
    pub mic_points: Vec<Option<JointPoint>>,
    /// RMS residual over all jointly used cells (m).
    pub rms_residual_m: f64,
    /// Row-major source-by-microphone usage after polish exclusions.
    pub used_links: Vec<bool>,
    /// True unless the anchors span rank 3: a 3-anchor (or any
    /// coplanar-anchor) frame leaves a global mirror across the anchor
    /// plane with identical residuals.
    pub mirror_ambiguous: bool,
    /// Joint iterations run before convergence.
    pub iterations: usize,
}

/// Jointly self-calibrate unsurveyed sources and microphones.
///
/// `source_seeds[s]` / `mic_seeds[m]` are `Some` for surveyed anchors
/// (held fixed) and `None` for unknowns; `ranges_sm[s * mics + m]` is
/// the measured cross range or `None` when refused upstream. Scale is
/// fixed by the absolute ranges; the anchors fix the rigid frame.
///
/// A trilateration warm start (exact on noiseless data up to the
/// anchor-plane mirror) seeds joint Levenberg-Marquardt over all
/// unknowns at once — per-point alternation cannot bootstrap here,
/// because sweep-one sub-solves face mostly-unconverged anchors and
/// diverge, while the joint gradient descends from the same start.
/// Then a final polish sweep re-solves every unknown with exclusion
/// against the converged frame, and the joint normal matrix over all
/// unknowns is inverted once (Cholesky) for honest marginal sigmas.
///
/// Returns `None` when the inputs disagree in shape, any surveyed seed
/// is non-finite, fewer than [`MIN_JOINT_ANCHORS`] anchors exist, the
/// anchors are collinear, the counting rules fail (fewer usable cells
/// than 3× unknowns, or any unknown with fewer than
/// [`MIN_RANGES_FOR_FIX`] links), the iteration or damping caps
/// exhaust, or the joint normal matrix is singular. Total on all
/// inputs; deterministic like the solver.
pub fn joint_self_calibrate(
    source_seeds: &[Option<[f64; 3]>],
    mic_seeds: &[Option<[f64; 3]>],
    ranges_sm: &[Option<(f64, f64)>],
) -> Option<JointEstimate> {
    let sources = source_seeds.len();
    let mics = mic_seeds.len();
    if sources == 0 || mics == 0 || ranges_sm.len() != sources * mics {
        return None;
    }
    for seed in source_seeds.iter().chain(mic_seeds.iter()).flatten() {
        if !seed.iter().all(|c| c.is_finite()) {
            return None;
        }
    }
    let anchors: Vec<[f64; 3]> = source_seeds
        .iter()
        .chain(mic_seeds.iter())
        .filter_map(|seed| *seed)
        .collect();
    if anchors.len() < MIN_JOINT_ANCHORS || geometry_rank(&anchors).collinear {
        return None;
    }
    let mut usable = vec![false; sources * mics];
    let mut usable_count = 0_usize;
    for (cell, range) in ranges_sm.iter().enumerate() {
        if matches!(range, Some((distance, sigma))
            if distance.is_finite() && *distance >= 0.0 && sigma.is_finite() && *sigma > 0.0)
        {
            usable[cell] = true;
            usable_count += 1;
        }
    }
    let unknown_sources: Vec<usize> = source_seeds
        .iter()
        .enumerate()
        .filter(|(_, seed)| seed.is_none())
        .map(|(index, _)| index)
        .collect();
    let unknown_mics: Vec<usize> = mic_seeds
        .iter()
        .enumerate()
        .filter(|(_, seed)| seed.is_none())
        .map(|(index, _)| index)
        .collect();
    let unknowns = unknown_sources.len() + unknown_mics.len();
    if unknowns == 0 || usable_count < 3 * unknowns {
        return None;
    }
    for source in &unknown_sources {
        let links = (0..mics).filter(|mic| usable[source * mics + mic]).count();
        if links < MIN_RANGES_FOR_FIX {
            return None;
        }
    }
    for mic in &unknown_mics {
        let links = (0..sources)
            .filter(|source| usable[source * mics + mic])
            .count();
        if links < MIN_RANGES_FOR_FIX {
            return None;
        }
    }
    // Warm start: closed-form trilateration from each unknown's three
    // best-linked anchors. Alternation cannot bootstrap from centroid
    // seeds — sweep-one sub-solves face mostly-wrong anchors and
    // diverge — but trilateration is exact on noiseless data up to the
    // anchor-plane mirror, and the mirror side is unobservable anyway:
    // every unknown starts on the +normal side (an arbitrary global
    // choice the mirror gauge absorbs) and alternation refines from
    // there. Unknowns without three linked anchors fall back to the
    // anchor centroid plus a deterministic per-index spread.
    let mut centroid = [0.0_f64; 3];
    for anchor in &anchors {
        for axis in 0..3 {
            centroid[axis] += anchor[axis];
        }
    }
    for axis in 0..3 {
        centroid[axis] /= anchors.len() as f64;
    }
    // Anchor catalogue: position plus its (is_source, index) identity
    // for link lookup. Fixed order: sources then microphones.
    let mut anchor_table: Vec<([f64; 3], bool, usize)> = Vec::new();
    for (index, seed) in source_seeds.iter().enumerate() {
        if let Some(position) = seed {
            anchor_table.push((*position, true, index));
        }
    }
    for (index, seed) in mic_seeds.iter().enumerate() {
        if let Some(position) = seed {
            anchor_table.push((*position, false, index));
        }
    }
    let mut source_positions: Vec<[f64; 3]> = source_seeds
        .iter()
        .map(|seed| seed.unwrap_or(centroid))
        .collect();
    let mut mic_positions: Vec<[f64; 3]> = mic_seeds
        .iter()
        .map(|seed| seed.unwrap_or(centroid))
        .collect();
    const SPREAD: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut spread_index = 0_usize;
    for source in &unknown_sources {
        let links: Vec<([f64; 3], f64)> = anchor_table
            .iter()
            .filter_map(|(position, is_source, index)| {
                let cell = if *is_source {
                    // Sources never range to sources; anchors on the
                    // source side link only to unknown microphones.
                    return None;
                } else {
                    source * mics + index
                };
                if !usable[cell] {
                    return None;
                }
                ranges_sm[cell].map(|(distance, _)| (*position, distance))
            })
            .collect();
        source_positions[*source] =
            warm_start(&links, &centroid, spread_index, &SPREAD);
        spread_index += 1;
    }
    for mic in &unknown_mics {
        let links: Vec<([f64; 3], f64)> = anchor_table
            .iter()
            .filter_map(|(position, is_source, index)| {
                let cell = if *is_source {
                    index * mics + mic
                } else {
                    return None;
                };
                if !usable[cell] {
                    return None;
                }
                ranges_sm[cell].map(|(distance, _)| (*position, distance))
            })
            .collect();
        mic_positions[*mic] = warm_start(&links, &centroid, spread_index, &SPREAD);
        spread_index += 1;
    }
    // Joint Levenberg-Marquardt over all unknowns. Per-point
    // alternation cannot bootstrap here: sweep-one sub-solves face
    // mostly-unconverged anchors and diverge, while the joint gradient
    // descends from the same warm start because every residual pulls at
    // once. Adaptive damping retries each iteration up to a bounded
    // number of lambda bumps; anything unbounded refuses.
    let debug_joint = std::env::var("SOTF_DEBUG_JOINT").is_ok();
    let mut lambda = 1.0e-3_f64;
    let mut cost = joint_robust_cost(
        &source_positions,
        &mic_positions,
        ranges_sm,
        &usable,
        sources,
        mics,
    )?;
    let mut iterations = 0_usize;
    let mut converged = false;
    while !converged {
        if iterations >= MAX_JOINT_ITERATIONS {
            return None;
        }
        let (normal, gradient) = joint_normal_system(
            &unknown_sources,
            &unknown_mics,
            &source_positions,
            &mic_positions,
            ranges_sm,
            &usable,
            sources,
            mics,
        )?;
        let mut accepted = false;
        for _ in 0..MAX_JOINT_LAMBDA_BUMPS {
            let mut damped = normal.clone();
            for axis in 0..damped.nrows() {
                damped[(axis, axis)] += lambda;
            }
            let Some(decomposition) = damped.cholesky() else {
                lambda *= 10.0;
                continue;
            };
            // The bump loop may retry, so the gradient is cloned per
            // attempt rather than moved.
            let step = decomposition.solve(&(-gradient.clone()));
            let mut trial_sources = source_positions.clone();
            let mut trial_mics = mic_positions.clone();
            for (slot, source) in unknown_sources.iter().enumerate() {
                for axis in 0..3 {
                    trial_sources[*source][axis] += step[slot * 3 + axis];
                }
            }
            for (slot, mic) in unknown_mics.iter().enumerate() {
                let base = (unknown_sources.len() + slot) * 3;
                for axis in 0..3 {
                    trial_mics[*mic][axis] += step[base + axis];
                }
            }
            if !trial_sources.iter().chain(trial_mics.iter()).all(|point| {
                point.iter().all(|c| c.is_finite())
                    && model_distance_vec(point, &centroid) < 1.0e6
            }) {
                lambda *= 10.0;
                continue;
            }
            let Some(trial_cost) = joint_robust_cost(
                &trial_sources,
                &trial_mics,
                ranges_sm,
                &usable,
                sources,
                mics,
            ) else {
                lambda *= 10.0;
                continue;
            };
            if trial_cost < cost {
                let mut max_step = 0.0_f64;
                for value in step.iter() {
                    max_step = max_step.max(value.abs());
                }
                source_positions = trial_sources;
                mic_positions = trial_mics;
                cost = trial_cost;
                lambda = (lambda / 10.0).max(1.0e-12);
                accepted = true;
                if debug_joint {
                    eprintln!("JOINT iter {iterations} cost {cost:.6} max_step {max_step:.3e}");
                }
                converged = max_step < JOINT_CONVERGENCE_STEP_M;
                break;
            }
            lambda *= 10.0;
            if lambda > 1.0e12 {
                return None;
            }
        }
        if !accepted {
            return None;
        }
        iterations += 1;
    }
    // Final polish: every unknown re-solves with exclusion against the
    // converged frame, so true outliers drop without cascade risk.
    let mut source_polish: Vec<Option<PositionEstimate>> = vec![None; sources];
    for source in &unknown_sources {
        let row: Vec<Option<(f64, f64)>> = (0..mics)
            .map(|mic| {
                let cell = source * mics + mic;
                if usable[cell] {
                    ranges_sm[cell]
                } else {
                    None
                }
            })
            .collect();
        source_polish[*source] = Some(solve_position(&mic_positions, &row)?);
    }
    let mut mic_polish: Vec<Option<PositionEstimate>> = vec![None; mics];
    for mic in &unknown_mics {
        let column: Vec<Option<(f64, f64)>> = (0..sources)
            .map(|source| {
                let cell = source * mics + mic;
                if usable[cell] {
                    ranges_sm[cell]
                } else {
                    None
                }
            })
            .collect();
        mic_polish[*mic] = Some(solve_position(&source_positions, &column)?);
    }
    for source in &unknown_sources {
        let Some(polish) = &source_polish[*source] else {
            return None;
        };
        source_positions[*source] = polish.position_m;
    }
    for mic in &unknown_mics {
        let Some(polish) = &mic_polish[*mic] else {
            return None;
        };
        mic_positions[*mic] = polish.position_m;
    }
    // A link survives only when usable and accepted by both endpoints'
    // polish (surveyed endpoints accept everything usable).
    let mut used_links = vec![false; sources * mics];
    for source in 0..sources {
        for mic in 0..mics {
            let cell = source * mics + mic;
            if !usable[cell] {
                continue;
            }
            let source_used = match &source_polish[source] {
                Some(polish) => polish.used.get(mic).copied().unwrap_or(false),
                None => true,
            };
            let mic_used = match &mic_polish[mic] {
                Some(polish) => polish.used.get(source).copied().unwrap_or(false),
                None => true,
            };
            used_links[cell] = source_used && mic_used;
        }
    }
    let sigmas = joint_marginal_sigmas(
        &unknown_sources,
        &unknown_mics,
        &source_positions,
        &mic_positions,
        ranges_sm,
        &used_links,
        sources,
        mics,
    )?;
    let mut rms_sum = 0.0_f64;
    let mut rms_count = 0_usize;
    for source in 0..sources {
        for mic in 0..mics {
            let cell = source * mics + mic;
            if !used_links[cell] {
                continue;
            }
            let Some((distance, _)) = ranges_sm[cell] else {
                continue;
            };
            let residual = model_distance_vec(&source_positions[source], &mic_positions[mic]) - distance;
            if !residual.is_finite() {
                return None;
            }
            rms_sum += residual * residual;
            rms_count += 1;
        }
    }
    if rms_count == 0 {
        return None;
    }
    let mut source_points: Vec<Option<JointPoint>> = vec![None; sources];
    for (slot, source) in unknown_sources.iter().enumerate() {
        source_points[*source] = Some(point_summary(
            *source,
            true,
            &source_positions,
            &mic_positions,
            ranges_sm,
            &used_links,
            sources,
            mics,
            &sigmas[slot * 3..slot * 3 + 3],
        )?);
    }
    let mut mic_points: Vec<Option<JointPoint>> = vec![None; mics];
    for (slot, mic) in unknown_mics.iter().enumerate() {
        let base = (unknown_sources.len() + slot) * 3;
        mic_points[*mic] = Some(point_summary(
            *mic,
            false,
            &source_positions,
            &mic_positions,
            ranges_sm,
            &used_links,
            sources,
            mics,
            &sigmas[base..base + 3],
        )?);
    }
    Some(JointEstimate {
        source_points,
        mic_points,
        rms_residual_m: (rms_sum / rms_count as f64).sqrt(),
        used_links,
        mirror_ambiguous: geometry_rank(&anchors).rank < 3,
        iterations,
    })
}

/// Euclidean distance between two slice points.
fn model_distance_vec(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Warm start for one unknown from its linked anchors.
///
/// Closed-form trilateration against the three best-spread linked
/// anchors, always on the +normal side — the side is unobservable, so
/// the choice is an arbitrary global gauge the frame mirror absorbs.
/// Falls back to the anchor centroid plus a deterministic per-index
/// spread when fewer than three anchors link or the triple
/// degenerates.
fn warm_start(
    links: &[([f64; 3], f64)],
    centroid: &[f64; 3],
    spread_index: usize,
    spread_axes: &[[f64; 3]; 6],
) -> [f64; 3] {
    if let Some(triple) = best_anchor_triple(links) {
        if let Some(point) = trilaterate(triple) {
            return point;
        }
    }
    let radius = 2.0 * f64::from(1 + (spread_index / spread_axes.len()) as u32);
    let axis = spread_axes[spread_index % spread_axes.len()];
    [
        centroid[0] + radius * axis[0],
        centroid[1] + radius * axis[1],
        centroid[2] + radius * axis[2],
    ]
}

/// Best-spread anchor triple: the combination maximizing the minimum
/// pairwise anchor distance. Deterministic (first maximum wins).
fn best_anchor_triple(links: &[([f64; 3], f64)]) -> Option<[([f64; 3], f64); 3]> {
    if links.len() < 3 {
        return None;
    }
    let mut best: Option<([([f64; 3], f64); 3], f64)> = None;
    for i in 0..links.len() {
        for j in (i + 1)..links.len() {
            for k in (j + 1)..links.len() {
                let (a, b, c) = (&links[i], &links[j], &links[k]);
                let spread = model_distance_vec(&a.0, &b.0)
                    .min(model_distance_vec(&a.0, &c.0))
                    .min(model_distance_vec(&b.0, &c.0));
                let better = match &best {
                    None => true,
                    Some((_, held)) => spread > *held,
                };
                if better {
                    best = Some(([*a, *b, *c], spread));
                }
            }
        }
    }
    best.map(|(triple, _)| triple)
}

/// Closed-form trilateration from three anchor ranges.
///
/// Returns the +normal-side solution; `None` when the triple
/// degenerates (coincident anchors or a collinear frame). A negative
/// squared height (noisy ranges missing intersection) clamps to the
/// anchor plane — a start, not a solution; the joint iterations refine
/// from there.
fn trilaterate(triple: [([f64; 3], f64); 3]) -> Option<[f64; 3]> {
    let [(a, ra), (b, rb), (c, rc)] = triple;
    let va = Vector3::new(a[0], a[1], a[2]);
    let ab = Vector3::new(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    let distance = ab.norm();
    if !distance.is_finite() || distance <= 1.0e-9 {
        return None;
    }
    let ex = ab / distance;
    let ac = Vector3::new(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
    let along = ex.dot(&ac);
    let temp = ac - ex * along;
    if !temp.norm().is_finite() || temp.norm() <= 1.0e-9 {
        return None;
    }
    let ey = temp.normalize();
    let ez = ex.cross(&ey);
    let cross = ey.dot(&ac);
    if !cross.is_finite() || cross.abs() <= 1.0e-9 {
        return None;
    }
    for range in [ra, rb, rc] {
        if !range.is_finite() || range < 0.0 {
            return None;
        }
    }
    let x = (ra * ra - rb * rb + distance * distance) / (2.0 * distance);
    let y = (ra * ra - rc * rc + along * along + cross * cross) / (2.0 * cross)
        - (along / cross) * x;
    let height_sq = ra * ra - x * x - y * y;
    if !height_sq.is_finite() {
        return None;
    }
    let point = va + ex * x + ey * y + ez * height_sq.max(0.0).sqrt();
    if point.iter().all(|c| c.is_finite()) {
        Some([point[0], point[1], point[2]])
    } else {
        None
    }
}

/// Weighted joint normal system `(J'WJ, J'Wr)` over all unknowns.
///
/// `link_mask` selects the cells (the usable mask while iterating, the
/// polish-survivor mask for the final covariance). Jacobian rows carry
/// +unit at the source block and -unit at the mic block; surveyed
/// endpoints contribute no columns. `None` on collapse.
fn joint_normal_system(
    unknown_sources: &[usize],
    unknown_mics: &[usize],
    source_positions: &[[f64; 3]],
    mic_positions: &[[f64; 3]],
    ranges_sm: &[Option<(f64, f64)>],
    link_mask: &[bool],
    sources: usize,
    mics: usize,
) -> Option<(DMatrix<f64>, DVector<f64>)> {
    let unknowns = unknown_sources.len() + unknown_mics.len();
    let mut unknown_id = vec![usize::MAX; sources + mics];
    for (slot, source) in unknown_sources.iter().enumerate() {
        unknown_id[*source] = slot;
    }
    for (slot, mic) in unknown_mics.iter().enumerate() {
        unknown_id[sources + mic] = unknown_sources.len() + slot;
    }
    let dim = 3 * unknowns;
    let mut normal = DMatrix::<f64>::zeros(dim, dim);
    let mut gradient = DVector::<f64>::zeros(dim);
    for source in 0..sources {
        for mic in 0..mics {
            let cell = source * mics + mic;
            if !link_mask.get(cell).copied().unwrap_or(false) {
                continue;
            }
            let Some((distance, sigma)) = ranges_sm.get(cell).copied().flatten() else {
                continue;
            };
            let from = &source_positions[source];
            let to = &mic_positions[mic];
            let model = model_distance_vec(from, to);
            if !model.is_finite() || model <= MIN_MODEL_DISTANCE_M {
                return None;
            }
            let unit = [
                (from[0] - to[0]) / model,
                (from[1] - to[1]) / model,
                (from[2] - to[2]) / model,
            ];
            let residual = model - distance;
            if !residual.is_finite() {
                return None;
            }
            let huber = (HUBER_SIGMA_MULTIPLE * sigma / residual.abs()).min(1.0);
            if !huber.is_finite() {
                return None;
            }
            let weight = huber / (sigma * sigma);
            let mut blocks: [(usize, f64); 2] = [(usize::MAX, 0.0); 2];
            let mut count = 0_usize;
            let source_unknown = unknown_id[source];
            if source_unknown != usize::MAX {
                blocks[count] = (source_unknown, 1.0);
                count += 1;
            }
            let mic_unknown = unknown_id[sources + mic];
            if mic_unknown != usize::MAX {
                blocks[count] = (mic_unknown, -1.0);
                count += 1;
            }
            for a in 0..count {
                for axis_a in 0..3 {
                    gradient[3 * blocks[a].0 + axis_a] +=
                        weight * blocks[a].1 * unit[axis_a] * residual;
                }
                for b in 0..count {
                    for axis_a in 0..3 {
                        for axis_b in 0..3 {
                            normal[(3 * blocks[a].0 + axis_a, 3 * blocks[b].0 + axis_b)] += weight
                                * blocks[a].1
                                * blocks[b].1
                                * unit[axis_a]
                                * unit[axis_b];
                        }
                    }
                }
            }
        }
    }
    Some((normal, gradient))
}

/// Huber robust cost over masked links, sigma-normalised (mirrors
/// [`robust_cost`]). `None` on collapse or a non-finite residual.
fn joint_robust_cost(
    source_positions: &[[f64; 3]],
    mic_positions: &[[f64; 3]],
    ranges_sm: &[Option<(f64, f64)>],
    link_mask: &[bool],
    sources: usize,
    mics: usize,
) -> Option<f64> {
    let mut cost = 0.0_f64;
    for source in 0..sources {
        for mic in 0..mics {
            let cell = source * mics + mic;
            if !link_mask.get(cell).copied().unwrap_or(false) {
                continue;
            }
            let Some((distance, sigma)) = ranges_sm.get(cell).copied().flatten() else {
                continue;
            };
            let model = model_distance_vec(&source_positions[source], &mic_positions[mic]);
            let residual = (model - distance).abs();
            if !residual.is_finite() {
                return None;
            }
            let delta = HUBER_SIGMA_MULTIPLE * sigma;
            let rho = if residual <= delta {
                0.5 * residual * residual
            } else {
                delta * (residual - 0.5 * delta)
            };
            cost += rho / (sigma * sigma);
        }
    }
    Some(cost)
}

/// Marginal per-axis sigmas for every unknown from the joint covariance.
///
/// Inverts the weighted joint normal matrix once via Cholesky (with
/// [`LEVENBERG_DAMPING`] on the diagonal) and returns the 3×U diagonal
/// in unknown order (sources then microphones). `None` on a singular
/// matrix — numerically unidentifiable refuses, never publishes.
fn joint_marginal_sigmas(
    unknown_sources: &[usize],
    unknown_mics: &[usize],
    source_positions: &[[f64; 3]],
    mic_positions: &[[f64; 3]],
    ranges_sm: &[Option<(f64, f64)>],
    used_links: &[bool],
    sources: usize,
    mics: usize,
) -> Option<Vec<f64>> {
    let (mut normal, _) = joint_normal_system(
        unknown_sources,
        unknown_mics,
        source_positions,
        mic_positions,
        ranges_sm,
        used_links,
        sources,
        mics,
    )?;
    for axis in 0..normal.nrows() {
        normal[(axis, axis)] += LEVENBERG_DAMPING;
    }
    let covariance = normal.cholesky()?.inverse();
    let mut sigmas = Vec::with_capacity(covariance.nrows());
    for axis in 0..covariance.nrows() {
        let sigma = covariance[(axis, axis)].max(0.0).sqrt();
        if !sigma.is_finite() {
            return None;
        }
        sigmas.push(sigma);
    }
    Some(sigmas)
}

/// Per-point summary: own-cell RMS, median sigma, and joint marginals.
fn point_summary(
    index: usize,
    is_source: bool,
    source_positions: &[[f64; 3]],
    mic_positions: &[[f64; 3]],
    ranges_sm: &[Option<(f64, f64)>],
    used_links: &[bool],
    sources: usize,
    mics: usize,
    sigma3: &[f64],
) -> Option<JointPoint> {
    let position_m = if is_source {
        source_positions.get(index).copied()?
    } else {
        mic_positions.get(index).copied()?
    };
    let mut sum_sq = 0.0_f64;
    let mut count = 0_usize;
    let mut sigmas = Vec::new();
    if is_source {
        for mic in 0..mics {
            let cell = index * mics + mic;
            if !used_links.get(cell).copied().unwrap_or(false) {
                continue;
            }
            let Some((distance, sigma)) = ranges_sm.get(cell).copied().flatten() else {
                continue;
            };
            let residual = model_distance_vec(&position_m, &mic_positions[mic]) - distance;
            if !residual.is_finite() {
                return None;
            }
            sum_sq += residual * residual;
            count += 1;
            sigmas.push(sigma);
        }
    } else {
        for source in 0..sources {
            let cell = source * mics + index;
            if !used_links.get(cell).copied().unwrap_or(false) {
                continue;
            }
            let Some((distance, sigma)) = ranges_sm.get(cell).copied().flatten() else {
                continue;
            };
            let residual = model_distance_vec(&source_positions[source], &position_m) - distance;
            if !residual.is_finite() {
                return None;
            }
            sum_sq += residual * residual;
            count += 1;
            sigmas.push(sigma);
        }
    }
    if count == 0 {
        return None;
    }
    sigmas.sort_by(f64::total_cmp);
    let median_sigma_m = if count % 2 == 1 {
        sigmas[count / 2]
    } else {
        0.5 * (sigmas[count / 2 - 1] + sigmas[count / 2])
    };
    Some(JointPoint {
        position_m,
        sigma_m: [sigma3.first().copied()?, sigma3.get(1).copied()?, sigma3.get(2).copied()?],
        rms_residual_m: (sum_sq / count as f64).sqrt(),
        median_sigma_m,
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

    /// Rich joint geometry: 5 non-coplanar sources.
    fn joint_sources() -> Vec<[f64; 3]> {
        vec![
            [3.5, 1.0, 1.4],
            [3.5, 2.0, 1.0],
            [1.0, -1.0, 1.6],
            [1.0, 4.0, 0.9],
            [1.0, 1.5, 2.4],
        ]
    }

    /// Rich joint geometry: 6 non-coplanar microphones.
    fn joint_mics() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.2],
            [0.0, 2.0, 0.9],
            [2.0, 2.0, 1.5],
            [1.0, 1.0, 0.8],
            [1.0, 3.0, 1.1],
        ]
    }

    /// Flat source-major cross ranges with one sigma everywhere.
    fn joint_ranges(
        sources: &[[f64; 3]],
        mics: &[[f64; 3]],
        sigma: f64,
    ) -> Vec<Option<(f64, f64)>> {
        let mut ranges = Vec::with_capacity(sources.len() * mics.len());
        for source in sources {
            for mic in mics {
                ranges.push(Some((distance(source, mic), sigma)));
            }
        }
        ranges
    }

    fn joint_max_error(estimate: &JointEstimate, sources: &[[f64; 3]], mics: &[[f64; 3]]) -> f64 {
        let mut worst = 0.0_f64;
        for (point, truth) in estimate
            .source_points
            .iter()
            .zip(sources.iter())
        {
            if let Some(point) = point {
                worst = worst.max(distance(&point.position_m, truth));
            }
        }
        for (point, truth) in estimate.mic_points.iter().zip(mics.iter()) {
            if let Some(point) = point {
                worst = worst.max(distance(&point.position_m, truth));
            }
        }
        worst
    }

    #[test]
    fn joint_recovers_exact_frame_from_three_anchors() {
        let sources = joint_sources();
        let mics = joint_mics();
        let ranges = joint_ranges(&sources, &mics, 0.001);
        let source_seeds: Vec<Option<[f64; 3]>> = vec![
            Some(sources[0]),
            Some(sources[1]),
            None,
            None,
            None,
        ];
        let mic_seeds: Vec<Option<[f64; 3]>> =
            vec![Some(mics[0]), None, None, None, None, None];
        let estimate = joint_self_calibrate(&source_seeds, &mic_seeds, &ranges)
            .expect("exact joint frame calibrates");
        assert!(estimate.mirror_ambiguous, "3 anchors leave the global mirror");
        assert!(estimate.iterations < MAX_JOINT_ITERATIONS);
        assert!(estimate.used_links.iter().all(|used| *used));
        assert!(estimate.rms_residual_m < 1.0e-9);
        let error = joint_max_error(&estimate, &sources, &mics);
        assert!(error < 1.0e-6, "joint max error {error}");
        // Anchors are held, never estimated.
        assert!(estimate.source_points[0].is_none());
        assert!(estimate.mic_points[0].is_none());
        assert_eq!(estimate.source_points.len(), sources.len());
        assert_eq!(estimate.mic_points.len(), mics.len());
    }

    #[test]
    fn joint_noisy_errors_match_marginal_sigmas() {
        // Fixed-seed honesty check mirroring the single-point
        // Monte-Carlo test: every unknown's error must sit inside its
        // joint 3σ marginal on every trial.
        let sources = joint_sources();
        let mics = joint_mics();
        let sigma = 0.005;
        let source_seeds: Vec<Option<[f64; 3]>> = vec![
            Some(sources[0]),
            Some(sources[1]),
            None,
            None,
            None,
        ];
        let mic_seeds: Vec<Option<[f64; 3]>> =
            vec![Some(mics[0]), None, None, None, None, None];
        let mut rng = FixedRng(0x5EED_5EED_5EED_5EED);
        for trial in 0..30 {
            let ranges: Vec<Option<(f64, f64)>> = joint_ranges(&sources, &mics, sigma)
                .into_iter()
                .map(|range| {
                    range.map(|(distance, sigma)| (distance + rng.next_gaussian() * sigma, sigma))
                })
                .collect();
            let estimate = joint_self_calibrate(&source_seeds, &mic_seeds, &ranges)
                .expect("noisy trial calibrates");
            for (point, truth) in estimate
                .source_points
                .iter()
                .zip(sources.iter())
                .filter_map(|(point, truth)| point.as_ref().map(|point| (point, truth)))
                .chain(estimate.mic_points.iter().zip(mics.iter()).filter_map(
                    |(point, truth)| point.as_ref().map(|point| (point, truth)),
                ))
            {
                for axis in 0..3 {
                    let error = (point.position_m[axis] - truth[axis]).abs();
                    assert!(
                        error <= 3.0 * point.sigma_m[axis],
                        "trial {trial} axis {axis}: error {error} sigma {}",
                        point.sigma_m[axis]
                    );
                }
            }
        }
    }

    #[test]
    fn joint_refuses_underdetermined_frames() {
        // Stereo sessions can never joint-solve: each microphone sees
        // at most 2 sources, below the 4-link floor (and below the
        // global count too).
        let sources = vec![[3.5, 1.0, 1.4], [3.5, 2.0, 1.0]];
        let mics = vec![
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.2],
            [0.0, 2.0, 0.9],
            [2.0, 2.0, 1.5],
        ];
        let ranges = joint_ranges(&sources, &mics, 0.001);
        let source_seeds = vec![Some(sources[0]), Some(sources[1])];
        let mic_seeds: Vec<Option<[f64; 3]>> =
            vec![Some(mics[0]), None, None, None];
        assert!(joint_self_calibrate(&source_seeds, &mic_seeds, &ranges).is_none());
    }

    #[test]
    fn joint_refuses_collinear_anchors() {
        let sources = joint_sources();
        let mics = joint_mics();
        let ranges = joint_ranges(&sources, &mics, 0.001);
        // Three anchors on a line: rotation about the line unobservable.
        let source_seeds: Vec<Option<[f64; 3]>> =
            vec![Some([0.0, 0.0, 0.0]), Some([1.0, 0.0, 0.0]), None, None, None];
        let mic_seeds: Vec<Option<[f64; 3]>> =
            vec![Some([2.0, 0.0, 0.0]), None, None, None, None, None];
        assert!(joint_self_calibrate(&source_seeds, &mic_seeds, &ranges).is_none());
        // Fewer than three anchors: the frame floats.
        let mic_seeds: Vec<Option<[f64; 3]>> =
            vec![Some(mics[0]), None, None, None, None, None];
        let source_seeds: Vec<Option<[f64; 3]>> = vec![None; 5];
        assert!(joint_self_calibrate(&source_seeds, &mic_seeds, &ranges).is_none());
    }

    #[test]
    fn joint_clears_mirror_with_rank3_anchors() {
        let sources = joint_sources();
        let mics = joint_mics();
        let ranges = joint_ranges(&sources, &mics, 0.001);
        // Four non-coplanar anchors break the anchor-plane mirror.
        let source_seeds: Vec<Option<[f64; 3]>> = vec![
            Some(sources[0]),
            Some(sources[1]),
            None,
            None,
            None,
        ];
        let mic_seeds: Vec<Option<[f64; 3]>> =
            vec![Some(mics[0]), Some(mics[4]), None, None, None, None];
        let estimate = joint_self_calibrate(&source_seeds, &mic_seeds, &ranges)
            .expect("rank-3 anchors calibrate");
        assert!(!estimate.mirror_ambiguous);
        let error = joint_max_error(&estimate, &sources, &mics);
        assert!(error < 1.0e-6, "joint max error {error}");
    }

    #[test]
    fn joint_polish_excludes_a_wild_link() {
        // One range corrupted by +15 cm: the polish drops the link and
        // the frame still recovers.
        let sources = joint_sources();
        let mics = joint_mics();
        let mut ranges = joint_ranges(&sources, &mics, 0.001);
        let wild = 2 * mics.len() + 3;
        ranges[wild] = Some((ranges[wild].expect("exact range").0 + 0.15, 0.001));
        let source_seeds: Vec<Option<[f64; 3]>> = vec![
            Some(sources[0]),
            Some(sources[1]),
            None,
            None,
            None,
        ];
        let mic_seeds: Vec<Option<[f64; 3]>> =
            vec![Some(mics[0]), None, None, None, None, None];
        let estimate = joint_self_calibrate(&source_seeds, &mic_seeds, &ranges)
            .expect("wild link drops on polish");
        assert!(!estimate.used_links[wild]);
        let error = joint_max_error(&estimate, &sources, &mics);
        assert!(error < 0.02, "joint max error {error}");
    }

    #[test]
    fn joint_is_deterministic() {
        let sources = joint_sources();
        let mics = joint_mics();
        let ranges = joint_ranges(&sources, &mics, 0.002);
        let source_seeds: Vec<Option<[f64; 3]>> = vec![
            Some(sources[0]),
            Some(sources[1]),
            None,
            None,
            None,
        ];
        let mic_seeds: Vec<Option<[f64; 3]>> =
            vec![Some(mics[0]), None, None, None, None, None];
        let first = joint_self_calibrate(&source_seeds, &mic_seeds, &ranges)
            .expect("first run calibrates");
        let second = joint_self_calibrate(&source_seeds, &mic_seeds, &ranges)
            .expect("second run calibrates");
        assert_eq!(first.iterations, second.iterations);
        assert_eq!(first.used_links, second.used_links);
        for (a, b) in first
            .source_points
            .iter()
            .zip(second.source_points.iter())
        {
            match (a, b) {
                (Some(a), Some(b)) => assert_eq!(a.position_m, b.position_m),
                (None, None) => {}
                _ => panic!("anchor pattern differs between runs"),
            }
        }
        for (a, b) in first.mic_points.iter().zip(second.mic_points.iter()) {
            match (a, b) {
                (Some(a), Some(b)) => assert_eq!(a.position_m, b.position_m),
                (None, None) => {}
                _ => panic!("anchor pattern differs between runs"),
            }
        }
    }

    #[test]
    fn joint_matches_inverse_when_sources_surveyed() {
        // All sources surveyed: the joint alternation collapses to one
        // inverse sweep per microphone, and the polish replays the
        // standard solve bit-for-bit.
        let sources = joint_sources();
        let mics = joint_mics();
        let ranges = joint_ranges(&sources, &mics, 0.001);
        let source_seeds: Vec<Option<[f64; 3]>> =
            sources.iter().map(|source| Some(*source)).collect();
        let mic_seeds: Vec<Option<[f64; 3]>> = vec![None; mics.len()];
        let estimate = joint_self_calibrate(&source_seeds, &mic_seeds, &ranges)
            .expect("surveyed sources calibrate");
        for (index, point) in estimate.mic_points.iter().enumerate() {
            let column: Vec<Option<(f64, f64)>> = (0..sources.len())
                .map(|source| ranges[source * mics.len() + index])
                .collect();
            let direct = solve_position(&sources, &column).expect("column solves");
            let joint = point.as_ref().expect("joint mic solves");
            assert_eq!(joint.position_m, direct.position_m);
            for axis in 0..3 {
                let ratio = joint.sigma_m[axis] / direct.sigma_m[axis];
                assert!(
                    (ratio - 1.0).abs() < 1.0e-9,
                    "axis {axis}: joint {} vs direct {}",
                    joint.sigma_m[axis],
                    direct.sigma_m[axis]
                );
            }
        }
    }

    #[test]
    fn joint_refuses_without_unknowns() {
        let sources = joint_sources();
        let mics = joint_mics();
        let ranges = joint_ranges(&sources, &mics, 0.001);
        let source_seeds: Vec<Option<[f64; 3]>> =
            sources.iter().map(|source| Some(*source)).collect();
        let mic_seeds: Vec<Option<[f64; 3]>> =
            mics.iter().map(|mic| Some(*mic)).collect();
        assert!(joint_self_calibrate(&source_seeds, &mic_seeds, &ranges).is_none());
    }
}

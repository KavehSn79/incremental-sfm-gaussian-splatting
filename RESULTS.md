# Reconstruction Results

## Dataset

Eight overlapping photographs of a static desk scene, captured using the
same iPhone 16 Pro main lens at 1× zoom.

- Oriented image dimensions: 4284 × 5712 pixels.
- Approximate focal length: fx = fy = 3960.6 pixels.
- Approximate principal point: (2142, 2856).
- Shared, fixed pinhole intrinsics.
- Lens distortion is not explicitly modeled.
- Reconstruction scale is arbitrary; coordinates are not measured in meters.

The intrinsics were estimated from image dimensions and EXIF information,
rather than obtained through camera calibration.

## Implementation

The project implements:

- Pinhole projection and camera-pose transformations.
- Descriptor matching with NumPy and Lowe's ratio test.
- Point normalization and the normalized eight-point algorithm.
- Fundamental-matrix RANSAC with Sampson-error scoring.
- Fundamental-to-essential matrix conversion.
- Essential-matrix decomposition and cheirality-based pose selection.
- Linear triangulation using SVD.
- Reprojection-error and triangulation-angle filtering.
- Feature-track ownership and incremental map expansion.
- Sparse bundle-adjustment residuals, parameterization, and Jacobian sparsity.
- Similarity alignment for comparison with COLMAP.

External components include:

- OpenCV SIFT feature extraction.
- OpenCV direct essential-matrix estimation for the final reconstruction.
- OpenCV PnP-RANSAC and pose refinement.
- SciPy nonlinear least-squares optimization.
- Open3D visualization and point-cloud export.

The final reconstruction uses directly estimated essential geometry.
The custom fundamental-matrix pipeline remains an experimental baseline.

## Eight-View Reconstruction

Before global bundle adjustment:

| Metric | Value |
|---|---:|
| Registered images | 8 |
| 3D points | 1,419 |
| Feature observations | 4,027 |
| RMS reprojection error | 1.004 px |

Global bundle adjustment:

- Optimized camera poses and 3D point positions.
- Kept camera 1 fixed to define the world coordinate system.
- Fixed one translation component of camera 2 to anchor scale.
- Kept intrinsics fixed.
- Used a robust soft-L1 objective and sparse numerical derivatives.
- Satisfied the solver's relative-cost termination criterion after
  25 function evaluations.

| Metric | Before BA | After BA |
|---|---:|---:|
| RMS reprojection error | 1.004 px | 0.775 px |
| Robust objective | 1469.224 | 939.042 |

These errors use identical observations before final filtering.
The RMS reprojection error decreased by approximately 23%.

After filtering invalid or high-error observations and weak tracks:

| Metric | Value |
|---|---:|
| Registered images | 8 |
| Retained 3D points | 1,413 |
| Retained feature observations | 4,013 |
| Mean track length | 2.84 |

The 0.775 px RMS value describes the pre-filtering comparison.
It should not be presented as a separately measured post-filtering RMS.

## COLMAP Comparison

COLMAP reconstructed the same eight images using shared, fixed PINHOLE
intrinsics.

For memory control, COLMAP feature extraction used a maximum image
dimension of 3200 pixels, first octave 0, and a feature budget of 2000.
Our OpenCV SIFT pipeline used full-resolution images. Feature sets,
matching behavior, and observation sets therefore differ.

| Metric | Our pipeline | COLMAP |
|---|---:|---:|
| Registered images | 8 | 8 |
| 3D points | 1,413 | 2,813 |
| Feature observations | 4,013 | 10,060 |
| Mean track length | 2.84 | 3.58 |

COLMAP produced approximately twice as many sparse points and more
observations per point.

COLMAP's reported mean reprojection error was 1.206 px. This is not
directly comparable to our 0.775 px pre-filtering RMS: the aggregation
and underlying observations differ.

## Camera-Pose Agreement

Camera centers were matched by image filename. A seven-parameter
similarity transform aligned COLMAP's centers to our coordinate system.

| Metric | Result |
|---|---:|
| Shared cameras | 8 |
| Aligned center RMS difference / initial baseline | 3.25% |
| Leave-one-camera-out center RMS / initial baseline | 5.54% |
| Median orientation difference | 3.32° |
| Maximum orientation difference | 4.12° |

The orientation comparison uses the alignment fitted from camera centers.
Orientations were not independently aligned.

These values measure agreement between two reconstructions.
COLMAP is a reference system, not ground truth.

![Camera-pose comparison](outputs/camera_pose_comparison.png)

## Validation and Limitations

### Synthetic geometry

The noiseless tests recovered known camera poses and 3D points to
numerical precision.

A noisy synthetic bundle-adjustment experiment exposed a limitation:
reprojection error decreased while error against the known 3D geometry
increased. Dense and accurately solved sparse optimization reached
essentially the same solution.

Consequently, low reprojection error alone is not treated as evidence of
accurate 3D geometry. Noise sensitivity and geometric conditioning remain
open evaluation topics.

### Custom fundamental-matrix baseline

The custom eight-point and RANSAC implementation passed its noiseless
synthetic checks. On the real dataset, converting its estimated F to a
constrained E produced substantially worse pose reprojection errors than
direct essential-matrix estimation.

The cause has not been conclusively isolated.

### Reconstruction coverage

This is a sparse feature-based reconstruction, not a dense surface or
mesh. Smooth, reflective, and weakly textured regions have limited
coverage. Repeated patterns can produce ambiguous matches.

Track ownership prevents reuse of the same feature in multiple tracks,
but uncertain track merges and multiple descriptors per landmark are
not yet implemented.

### Optimization

Solver convergence refers to numerical termination of a local objective.
It does not establish a global optimum or metric scene accuracy.

### Resource use

An isolated full-resolution OpenCV SIFT extraction reached approximately
5.42 GiB peak resident memory. Repeated WSL service failures occurred
during the initial incremental run.

Feature extraction was subsequently isolated in subprocesses and cached
on disk. The remaining images registered successfully after this change.
Memory pressure is a suspected contributor, not a confirmed crash cause.

[Evaluation results and limitations](RESULTS.md)
Eight-view sparse reconstruction after global bundle adjustment. Camera wireframes show estimated poses; 1,384 of 1,413 points are displayed, with 29 distant points hidden for presentation only.

## Interview Description

I built an incremental sparse Structure-from-Motion pipeline that
registered eight images and reconstructed 1,413 points. I implemented
camera geometry, descriptor matching, normalized eight-point estimation,
RANSAC, essential-matrix decomposition, triangulation, feature-track
management, and the bundle-adjustment objective and sparsity structure.
I used OpenCV for SIFT, direct essential estimation, and PnP, and SciPy
for nonlinear optimization. Global bundle adjustment reduced RMS
reprojection error by approximately 23%. I also compared the result with
COLMAP after similarity alignment and documented cases where low pixel
error did not imply accurate geometry.
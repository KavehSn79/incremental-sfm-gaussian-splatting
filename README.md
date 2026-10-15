# Incremental 3D Reconstruction and Gaussian Splatting

An incremental Structure-from-Motion pipeline that reconstructs a static scene
from photographs, refines camera poses and sparse geometry with bundle
adjustment, and exports the reconstruction for Gaussian Splatting.

The project combines NumPy implementations of multiview geometry with OpenCV
feature detection and robust pose estimation, SciPy optimization, and the
external Graphdeco Gaussian Splatting implementation.

![Sparse reconstruction and estimated cameras](assets/reconstruction.png)

[Watch the rendered camera-path video](assets/camera_path.mp4)

## Pipeline

1. Detect SIFT keypoints and descriptors.
2. Match descriptors using Euclidean distance and Lowe's ratio test.
3. Estimate two-view geometry and reject inconsistent matches.
4. Recover relative camera pose and triangulate initial map points.
5. Register additional cameras using 2D–3D matching and PnP-RANSAC.
6. Extend feature tracks and triangulate new points.
7. Jointly refine camera poses and points using robust bundle adjustment.
8. Filter observations and points using reprojection error, depth, and geometry.
9. Compare camera poses with an independently generated COLMAP reconstruction.
10. Export our map and camera poses for Gaussian Splatting.
11. Train Gaussians and render an interpolated camera-path video.

## Implementation scope

Implemented in this project:

- Camera transformations and perspective projection.
- NumPy descriptor matching and ratio testing.
- Point normalization and the normalized eight-point algorithm.
- Fundamental-matrix RANSAC with Sampson-error scoring.
- Fundamental-to-essential conversion and essential-matrix decomposition.
- Linear triangulation and pose selection by cheirality.
- Reprojection and triangulation-angle filtering.
- Track construction, incremental map management, and feature caching.
- Bundle-adjustment parameterization, residuals, and Jacobian sparsity.
- Similarity alignment for comparison with COLMAP.
- Export to COLMAP's model format and camera-path rendering integration.

External components:

- OpenCV: SIFT, direct essential-matrix RANSAC, PnP, and pose refinement.
- SciPy: nonlinear least squares and rotation interpolation.
- Open3D: point-cloud and camera visualization.
- COLMAP: independent reconstruction baseline.
- Graphdeco Gaussian Splatting: Gaussian optimization and CUDA rendering.

The practical reconstruction uses OpenCV's direct essential-matrix estimation.
The custom fundamental-matrix route is retained for learning and diagnostics:
it passes the noiseless synthetic tests but produced substantially worse
real-image pose estimates in these experiments.

## Results

### Sparse reconstruction

Eight original iPhone photographs of a static desk scene were used.

| Measurement | Result |
|---|---:|
| Registered cameras | 8 / 8 |
| Final retained points | 1,413 |
| Final retained observations | 4,013 |
| RMS reprojection error before global BA | 1.004 px |
| RMS reprojection error after global BA | 0.775 px |

The before/after RMS values use the same 4,027 observations, before final
filtering.

### Comparison with COLMAP

Camera centers were aligned using a fitted similarity transformation.

| Measurement | Result |
|---|---:|
| Shared cameras | 8 |
| Aligned camera-center RMS / initial baseline | 3.249% |
| Median orientation difference | 3.317° |
| Maximum orientation difference | 4.123° |
| Leave-one-camera-out center RMS | 0.055191 map units |

COLMAP is a comparison baseline, not ground truth. The aligned RMS is measured
on the same cameras used to fit the alignment.

![Aligned camera comparison](assets/camera_pose_comparison.png)

### Gaussian Splatting

Both experiments used seven appearance-training images and one held-out image,
3,000 iterations, resolution factor 8, and spherical-harmonic degree 1.

| Initialization | Initial points | Test PSNR ↑ | Test L1 ↓ |
|---|---:|---:|---:|
| COLMAP | 2,813 | 21.632 dB | 0.038443 |
| Our SfM | 1,413 | 19.788 dB | 0.052646 |

Metrics were calculated from saved PNG renders against an identical reference
image at 536 × 714 pixels. Our initialization achieved 1.844 dB lower test PSNR
in this single-image comparison.

Camera poses, initial point locations, colors, and coverage differ between
the initializations. This is not an isolated test of point count.

The held-out image was excluded from Gaussian appearance optimization.
Both SfM reconstructions used all eight photographs, so this is an
appearance holdout with known estimated cameras.

| Reference | Our SfM initialization | COLMAP initialization |
|---|---|---|
| ![](assets/test_reference.png) | ![](assets/our_sfm_test_render.png) | ![](assets/colmap_test_render.png) |

## Source guide

| File | Purpose |
|---|---|
| `src/camera.py` | Coordinate transformations and projection |
| `src/sift_features.py` | SIFT extraction |
| `src/cache_features.py` | Cached feature extraction in a separate process |
| `src/epipolar.py` | Two-view geometry, triangulation, and error functions |
| `src/demo_single_match.py` | Initial two-view reconstruction |
| `src/demo_pnp.py` | Third-camera registration |
| `src/expand_reconstruction.py` | Initial map expansion |
| `src/build_tracks.py` | Consistent observation tracks |
| `src/demo_bundle_adjustment.py` | Three-camera bundle adjustment |
| `src/incremental_sfm.py` | Additional camera registration and map expansion |
| `src/bundle_adjust_all.py` | Global bundle adjustment and filtering |
| `src/run_colmap.py` | Independent COLMAP baseline |
| `src/compare_colmap.py` | Camera alignment and comparison |
| `src/export_our_sfm.py` | Export our geometry for Gaussian Splatting |
| `src/portfolio_viewer.py` | Reconstruction visualization |
| `src/demo_fundamental.py` | Synthetic geometry validation |
| `src/test_bundle_adjustment.py` | Synthetic optimizer diagnostics |
| `scripts/render_camera_path.py` | Video renderer; run inside the Gaussian Splatting repository |

## Camera convention

Rotations use `R_cw` for world-to-camera and `R_wc` for camera-to-world:

```text
P_camera = R_cw @ P_world + t_cw
R_wc = R_cw.T
Camera center = -R_cw.T @ t_cw
```

Reconstruction scale is arbitrary. Bundle adjustment fixes camera 1 and one
translation component of camera 2 to remove the similarity gauge freedom.

## Running the reconstruction

The tested setup uses Python 3.12 on Ubuntu under WSL2. Reconstruction and
Gaussian Splatting use separate virtual environments.

Place the original photos at `data/images/image1.jpg` through `image8.jpg`.
The current demonstration expects oriented grayscale dimensions
`(5712, 4284)` and uses approximate, fixed shared intrinsics:

```python
K = [
    [3960.6, 0.0, 2142.0],
    [0.0, 3960.6, 2856.0],
    [0.0, 0.0, 1.0],
]
```

These values are specific to this dataset. A different camera, resolution,
crop, or orientation requires corresponding intrinsics.

Run the stages from the project root:

```bash
source reconstruction_env/bin/activate

python src/demo_fundamental.py
python src/demo_single_match.py
python src/demo_pnp.py
python src/expand_reconstruction.py
python src/build_tracks.py
python src/demo_bundle_adjustment.py

python -u src/incremental_sfm.py image4.jpg
python -u src/incremental_sfm.py --resume image5.jpg
python -u src/incremental_sfm.py --resume image6.jpg
python -u src/incremental_sfm.py --resume image7.jpg
python -u src/incremental_sfm.py --resume image8.jpg

python -u src/bundle_adjust_all.py
python src/export_our_sfm.py
```

Generated maps, caches, and diagnostics are stored in `outputs/`.

## Gaussian Splatting integration

Use the external implementation:
https://github.com/graphdeco-inria/gaussian-splatting

Our exported dataset is written to `data/gs_our_sfm`.
Its COLMAP-compatible format does not mean COLMAP estimated this geometry.

From the Gaussian Splatting repository, with its environment activated:

```bash
python -u train.py \
  -s ~/projects/3d-reconstruction/data/gs_our_sfm \
  -m ~/projects/3d-reconstruction/outputs/gs_our_sfm_holdout_3000 \
  -r 8 \
  --eval \
  --iterations 3000 \
  --sh_degree 1 \
  --data_device cpu \
  --densify_from_iter 500 \
  --densify_until_iter 2500 \
  --densification_interval 200 \
  --densify_grad_threshold 0.0005 \
  --test_iterations 3000 \
  --save_iterations 3000 \
  --checkpoint_iterations 3000 \
  --disable_viewer
```

Copy `scripts/render_camera_path.py` into that repository's root, then run:

```bash
python render_camera_path.py \
  -m ~/projects/3d-reconstruction/outputs/gs_our_sfm_holdout_3000 \
  --iteration 3000 \
  --seconds 8 \
  --fps 24
```

The tested GPU was an NVIDIA RTX 3060 Laptop GPU with 6 GB VRAM.
Gaussian Splatting was built using PyTorch 2.9.0+cu130 and CUDA Toolkit 13.0.
Its CUDA extensions require a compatible build environment; installing the
reconstruction requirements alone does not install them.

## Validation and limitations

- Noiseless synthetic tests recover the expected geometry to numerical precision.
- Noisy synthetic tests show that lower reprojection error can coexist with
  worse 3D geometry.
- Real-world geometric accuracy has not been measured against ground truth.
- Intrinsics are approximate and fixed; residual lens distortion is not modeled.
- The scene contains reflective and weakly textured surfaces.
- Gaussian renders retain blur, streaks, and doubled edges.
- The appearance evaluation uses one held-out image from one scene.
- The camera-path video demonstrates rendering; it is not an accuracy test.
- Full-resolution SIFT is memory intensive. Isolated extraction and caching
  reduce memory pressure during incremental reconstruction.

## Acknowledgments

This project uses OpenCV, NumPy, SciPy, Open3D, COLMAP, PyTorch, and the
Graphdeco Gaussian Splatting implementation. Third-party code retains its
respective license terms.
import numpy as np


def normalize_points(points):
    """
    points: shape (N, 2)

    Returns:
        normalized_points: shape (N, 2)
        T: shape (3, 3)
    """
    npPoints = np.array(points, dtype=float)
    centroid = np.mean(npPoints, axis=0)
    diff = npPoints - centroid
    dist = np.linalg.norm(diff, axis=1)

    mean_distance = np.mean(dist)
    if mean_distance < 1e-8:
        raise ValueError("Points have insufficient spread for normalization.")

    s = np.sqrt(2) / mean_distance
    normalized_points = s * diff

    T = np.array([
    [s, 0, -s * centroid[0]],
    [0, s, -s * centroid[1]],
    [0, 0, 1],
    ], dtype=float)

    return normalized_points, T

def build_design_matrix(points1, points2):
    """
    points1, points2: corresponding normalized points, shape (N, 2)
    returns: design matrix A, shape (N, 9)
    """
    u1 = points1[:, :1]
    v1 = points1[:, 1:2]

    u2 = points2[:, :1]
    v2 = points2[:, 1:2]

    A = np.column_stack((u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1 , np.ones(len(points1))
                         ))

    return A

def estimate_fundamental_matrix(points1, points2):
    """
    Estimate F from at least eight pixel correspondences.
    """

    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)

    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError("points1 must have shape (N, 2).")

    if points2.ndim != 2 or points2.shape[1] != 2:
        raise ValueError("points2 must have shape (N, 2).")

    if len(points1) != len(points2):
        raise ValueError("Both point arrays must have the same length.")

    if len(points1) < 8:
        raise ValueError("At least eight correspondences are required.")

    if not np.all(np.isfinite(points1)) or not np.all(np.isfinite(points2)):
        raise ValueError("Point coordinates must not contain NaN or infinity.")

    
    normalized_points1, T1 = normalize_points(points1)
    normalized_points2, T2 = normalize_points(points2)

    A = build_design_matrix(normalized_points1, normalized_points2)

    U, singular_values, Vt = np.linalg.svd(A, full_matrices=True)

    f = Vt[-1]
    F_initial = f.reshape(3, 3)

    U_F, s_F, Vt_F = np.linalg.svd(F_initial)
    s_F[-1] = 0

    F_rank2 = U_F @ np.diag(s_F) @ Vt_F

    F = T2.T @ F_rank2 @ T1

    F = F / np.linalg.norm(F)

    return F

def sampson_error(F, points1, points2):
    """
    Compute squared Sampson errors.

    points1, points2: pixel coordinates, shape (N, 2)
    returns: errors in squared pixels, shape (N,)
    """
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)
    F = np.asarray(F, dtype=float)

    # Normalize F so the denominator tolerance is scale-independent.
    F_norm = np.linalg.norm(F)

    if not np.isfinite(F_norm) or F_norm == 0:
        raise ValueError("F must be finite and nonzero.")

    F = F / F_norm

    # Convert pixel coordinates to homogeneous coordinates.
    points1_h = np.column_stack((
        points1, np.ones(len(points1))
    ))
    points2_h = np.column_stack((
        points2, np.ones(len(points2))
    ))

    # Compute epipolar lines in both images.
    lines2 = (F @ points1_h.T).T
    lines1 = (F.T @ points2_h.T).T

    # One epipolar residual per correspondence.
    residuals = np.sum(points2_h * lines2, axis=1)

    denominator = (
        np.sum(lines2[:, :2] ** 2, axis=1)
        + np.sum(lines1[:, :2] ** 2, axis=1)
    )

    # Undefined errors remain infinity.
    errors = np.full(len(points1), np.inf)

    np.divide(
        residuals ** 2,
        denominator,
        out=errors,
        where=denominator > 1e-15,
    )

    return errors

def ransac_fundamental_matrix(
    points1,
    points2,
    threshold_px=1.5,
    iterations=2000,
    seed=42,
):
    """
    Estimate F robustly using eight-point RANSAC.

    Returns:
        F: shape (3, 3)
        inlier_mask: shape (N,), dtype bool
    """
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)

    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError("points1 must have shape (N, 2).")

    if points2.ndim != 2 or points2.shape[1] != 2:
        raise ValueError("points2 must have shape (N, 2).")

    if points1.shape != points2.shape:
        raise ValueError("Point arrays must have matching shapes.")

    if len(points1) < 8:
        raise ValueError("At least eight correspondences are required.")

    if not (
        np.all(np.isfinite(points1))
        and np.all(np.isfinite(points2))
    ):
        raise ValueError("Point coordinates must be finite.")

    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError("threshold_px must be finite and positive.")

    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")

    rng = np.random.default_rng(seed)
    threshold_squared = threshold_px ** 2

    best_F = None
    best_mask = None
    best_count = 0
    best_median_error = np.inf

    for _ in range(iterations):
        # Sample eight pairs without repeating an index.
        indices = rng.choice(len(points1), size=8, replace=False)
        sample1 = points1[indices]
        sample2 = points2[indices]

        try:
            # Reject samples that do not provide eight
            # independent linear constraints.
            normalized1, _ = normalize_points(sample1)
            normalized2, _ = normalize_points(sample2)

            A = build_design_matrix(normalized1, normalized2)

            if np.linalg.matrix_rank(A) < 8:
                continue

            candidate_F = estimate_fundamental_matrix(
                sample1, sample2
            )

        except (ValueError, np.linalg.LinAlgError):
            continue

        if not np.all(np.isfinite(candidate_F)):
            continue

        # Evaluate the candidate on all correspondences.
        errors = sampson_error(candidate_F, points1, points2)
        mask = errors < threshold_squared
        count = int(np.count_nonzero(mask))

        if count < 8:
            continue

        median_error = float(np.median(errors[mask]))

        # Prefer more inliers; break ties using median error.
        if (
            count > best_count
            or (
                count == best_count
                and median_error < best_median_error
            )
        ):
            best_F = candidate_F
            best_mask = mask
            best_count = count
            best_median_error = median_error

    if best_F is None:
        raise RuntimeError(
            "RANSAC could not find a model with at least eight inliers."
        )

    # Refit using the winning consensus set.
    # Keep the refinement only if its consensus score improves.
    try:
        refined_F = estimate_fundamental_matrix(
            points1[best_mask],
            points2[best_mask],
        )

        if np.all(np.isfinite(refined_F)):
            errors = sampson_error(refined_F, points1, points2)
            refined_mask = errors < threshold_squared
            refined_count = int(np.count_nonzero(refined_mask))

            if refined_count >= 8:
                refined_median = float(
                    np.median(errors[refined_mask])
                )

                if (
                    refined_count > best_count
                    or (
                        refined_count == best_count
                        and refined_median <= best_median_error
                    )
                ):
                    best_F = refined_F
                    best_mask = refined_mask

    except (ValueError, np.linalg.LinAlgError):
        pass

    print("\nRANSAC refinement diagnostic")
    print("Original inlier count:", best_count)
    print("Refitted inlier count:", refined_count)

    # Compare both models on the same original consensus set.
    original_errors = sampson_error(best_F, points1, points2)

    print(
        "Original median Sampson error:",
        np.median(original_errors[best_mask]),
    )
    print(
        "Refitted median Sampson error:",
        np.median(errors[best_mask]),
    )

    # Check compatibility with the approximate camera intrinsics.
    K = np.array([
        [3960.6, 0.0, 2142.0],
        [0.0, 3960.6, 2856.0],
        [0.0, 0.0, 1.0],
    ])

    for label, model in (
        ("Original", best_F),
        ("Refitted", refined_F),
    ):
        values = np.linalg.svd(K.T @ model @ K, compute_uv=False)
        print(f"{label} E singular-value ratio:", values[1] / values[0])
        
    return best_F, best_mask

def fundamental_to_essential(F, K1, K2):
    """
    Convert F to E and enforce the essential-matrix constraints.
    """
    E_initial = K2.T @ F @ K1
    U, singular_values, Vt = np.linalg.svd(E_initial, full_matrices=True)
    s = (singular_values[0] + singular_values[1]) / 2
    singular_values = [s, s, 0]
    E = U @ np.diag(singular_values) @ Vt
    E_normalized = E / np.linalg.norm(E)
    return E_normalized

def decompose_essential_matrix(E):
    """
    Return four (R, t) candidates mapping camera 1 to camera 2.
    """
    U, _, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1
    
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    Ra = U @ W @ Vt
    Rb = U @ W.T @ Vt

    t = U[:, 2]

    return (Ra, t), (Ra, -t), (Rb, t), (Rb, -t)


def triangulate_points(points1, points2, P1, P2):
    """
    Triangulate corresponding pixels using linear DLT.

    Args:
        points1, points2: shape (N, 2)
        P1, P2: projection matrices, shape (3, 4)

    Returns:
        points3d: shape (N, 3); invalid rows contain NaN
        valid: shape (N,), indicating finite triangulations

    Coordinates are expressed in the frame used by P1 and P2.
    Positive depth is checked separately.
    """
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)

    if points1.ndim != 2 or points1.shape[1] != 2:
        raise ValueError("points1 must have shape (N, 2).")

    if points2.shape != points1.shape:
        raise ValueError("points2 must have the same shape as points1.")

    if P1.shape != (3, 4) or P2.shape != (3, 4):
        raise ValueError("Projection matrices must have shape (3, 4).")

    if not all(
        np.all(np.isfinite(array))
        for array in (points1, points2, P1, P2)
    ):
        raise ValueError("All inputs must contain finite values.")

    points3d = np.full((len(points1), 3), np.nan)
    valid = np.zeros(len(points1), dtype=bool)

    for index, (point1, point2) in enumerate(zip(points1, points2)):
        u1, v1 = point1
        u2, v2 = point2

        # Each camera contributes two linear constraints.
        A = np.vstack((
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ))

        _, _, Vt = np.linalg.svd(A)
        point_h = Vt[-1]

        # Avoid converting points at or near infinity.
        if abs(point_h[3]) < 1e-10:
            continue

        point3d = point_h[:3] / point_h[3]

        if np.all(np.isfinite(point3d)):
            points3d[index] = point3d
            valid[index] = True

    return points3d, valid

def select_pose_by_cheirality(points1, points2, K1, K2, candidates):
    """
    Choose the pose with the most points in front of both cameras.

    Camera 1 defines the world frame:
        P1 = K1 @ [I | 0]
        P2 = K2 @ [R | t]

    Returns:
        R, t: transformation from camera 1 to camera 2
        points3d: all triangulated points, shape (N, 3)
        positive_depth_mask: shape (N,)
        counts: positive-depth count for each candidate
    """
    P1 = K1 @ np.column_stack((np.eye(3), np.zeros(3)))

    best_result = None
    best_count = -1
    counts = []

    for R, t in candidates:
        P2 = K2 @ np.column_stack((R, t))

        points3d, valid = triangulate_points(
            points1, points2, P1, P2
        )

        # Express reconstructed points in camera 2's frame.
        points_camera2 = (R @ points3d.T).T + t

        positive_depth_mask = (
            valid
            & (points3d[:, 2] > 1e-8)
            & (points_camera2[:, 2] > 1e-8)
        )

        count = int(np.count_nonzero(positive_depth_mask))
        counts.append(count)

        if count > best_count:
            best_count = count
            best_result = (
                R, t, points3d, positive_depth_mask
            )

    if best_result is None or best_count == 0:
        raise RuntimeError(
            "No candidate produced points in front of both cameras."
        )

    return (*best_result, counts)

def reprojection_errors(points3d, observed_pixels, P):
    """
    Project 3D points and measure Euclidean pixel errors.

    Args:
        points3d: shape (N, 3)
        observed_pixels: shape (N, 2)
        P: projection matrix, shape (3, 4)

    Returns:
        errors: shape (N,), in pixels.
                Invalid projections receive infinity.
    """
    points3d = np.asarray(points3d, dtype=float)
    observed_pixels = np.asarray(observed_pixels, dtype=float)
    P = np.asarray(P, dtype=float)

    if points3d.ndim != 2 or points3d.shape[1] != 3:
        raise ValueError("points3d must have shape (N, 3).")

    if observed_pixels.shape != (len(points3d), 2):
        raise ValueError("observed_pixels must have shape (N, 2).")

    if P.shape != (3, 4) or not np.all(np.isfinite(P)):
        raise ValueError("P must be a finite (3, 4) matrix.")

    points_h = np.column_stack((
        points3d, np.ones(len(points3d))
    ))

    projected_h = (P @ points_h.T).T

    valid = (
        np.all(np.isfinite(projected_h), axis=1)
        & np.all(np.isfinite(observed_pixels), axis=1)
        & (np.abs(projected_h[:, 2]) > 1e-10)
    )

    errors = np.full(len(points3d), np.inf)

    projected_pixels = (
        projected_h[valid, :2]
        / projected_h[valid, 2:3]
    )

    errors[valid] = np.linalg.norm(
        projected_pixels - observed_pixels[valid],
        axis=1,
    )

    return errors




if __name__ == "__main__":
    F = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    ])

    points1 = np.array([
        [200.0, 100.0],
        [200.0, 100.0],
    ])

    points2 = np.array([
        [350.0, 100.0],
        [350.0, 102.0],
    ])

    print("Sampson errors:", sampson_error(F, points1, points2))

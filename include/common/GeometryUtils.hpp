#pragma once

/**
 * @file GeometryUtils.hpp
 * @brief Header-only geometry utility functions.
 *
 * Replaces Python numpy/scipy.spatial.transform operations with Eigen equivalents.
 * All functions are inline to allow header-only usage.
 */

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gpnp {

// ============================================================================
// Camera Geometry
// ============================================================================

/**
 * @brief Convert pixel coordinates to a normalized camera ray.
 *
 * Equivalent to Python:
 *   ray = K_inv @ [u, v, 1]
 *   return ray / norm(ray)
 *
 * @param u        Pixel x-coordinate
 * @param v        Pixel y-coordinate
 * @param K_inv    3×3 inverse camera intrinsic matrix
 * @return Unit direction vector in camera frame
 */
inline Eigen::Vector3d pixelToCameraRay(double u, double v, const Eigen::Matrix3d& K_inv) {
    Eigen::Vector3d ray = K_inv * Eigen::Vector3d(u, v, 1.0);
    return ray.normalized();
}

/**
 * @brief Batch pixel coordinates → normalized camera ray directions.
 *
 * Vectorized equivalent of Python 0626 _batch_pixel_to_camera_ray:
 *   homo = [pixels, ones(N,1)]
 *   rays = homo @ K_inv.T
 *   return rays / norm(rays, axis=1)
 *
 * @param pixels  N×2 matrix of (u,v) pixel coordinates
 * @param K_inv   3×3 inverse camera intrinsic matrix
 * @param rays_out [out] N×3 matrix of unit direction vectors
 */
inline void batchPixelToCameraRay(const Eigen::MatrixX2d& pixels,
                                   const Eigen::Matrix3d& K_inv,
                                   Eigen::MatrixX3d& rays_out) {
    const int N = static_cast<int>(pixels.rows());
    // Build homogeneous: [pixels | ones(N,1)]
    Eigen::MatrixXd homo(N, 3);
    homo.leftCols<2>() = pixels;
    homo.col(2).setOnes();
    // rays = homo @ K_inv.T  (N×3)
    rays_out = homo * K_inv.transpose();
    // Normalize each row
    for (int i = 0; i < N; ++i) {
        double n = rays_out.row(i).norm();
        if (n > 1e-12) rays_out.row(i) /= n;
    }
}

/**
 * @brief Project a 3D camera-frame point to image coordinates.
 *
 * Equivalent to Python:
 *   uv = K @ P_cam
 *   return uv[:2] / uv[2]
 *
 * @param P_cam  3D point in camera frame
 * @param K      3×3 camera intrinsic matrix
 * @return (u, v) pixel coordinates
 */
inline Eigen::Vector2d projectToImage(const Eigen::Vector3d& P_cam, const Eigen::Matrix3d& K) {
    Eigen::Vector3d uv = K * P_cam;
    return Eigen::Vector2d(uv.x() / uv.z(), uv.y() / uv.z());
}

// ============================================================================
// Quaternion / Rotation Utilities
// ============================================================================

/**
 * @brief Convert a rotation matrix to quaternion [x, y, z, w] (scipy convention).
 *
 * @param R  3×3 rotation matrix
 * @return [qx, qy, qz, qw] in scipy-compatible order
 */
inline Eigen::Vector4d rotationMatrixToQuat(const Eigen::Matrix3d& R) {
    Eigen::Quaterniond q(R);
    // Eigen stores [x, y, z, w] internally; coeffs() returns [x, y, z, w]
    return q.coeffs(); // [x, y, z, w] — matches scipy convention
}

/**
 * @brief Convert quaternion [x, y, z, w] to rotation matrix.
 *
 * @param q  [qx, qy, qz, qw] in scipy-compatible order
 * @return 3×3 rotation matrix
 */
inline Eigen::Matrix3d quatToRotationMatrix(const Eigen::Vector4d& q_vec) {
    Eigen::Quaterniond q(q_vec(3), q_vec(0), q_vec(1), q_vec(2)); // w, x, y, z
    return q.toRotationMatrix();
}

/**
 * @brief Normalize a quaternion and ensure scalar part is positive.
 *
 * Equivalent to Python:
 *   q = q / np.linalg.norm(q)
 *   if q[3] < 0: q = -q
 *
 * @param q_vec  [qx, qy, qz, qw] in scipy-compatible order
 * @return Normalized quaternion with qw >= 0
 */
inline Eigen::Vector4d normalizeQuaternion(const Eigen::Vector4d& q_vec) {
    double norm = q_vec.norm();
    Eigen::Vector4d qn = q_vec / norm;
    if (qn(3) < 0.0) { // qw < 0
        qn = -qn;
    }
    return qn;
}

// ============================================================================
// Statistics (matching numpy behavior)
// ============================================================================

/**
 * @brief Compute median of a vector (matches numpy.median for both odd/even length).
 *
 * For even-length vectors, returns the average of the two middle elements.
 *
 * @param values  Input values (copied, not modified in-place)
 * @return Median value, or 0.0 if empty
 */
inline double computeMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;

    size_t n = values.size();
    size_t mid = n / 2;

    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
    double mid_val = values[mid];

    if (n % 2 == 0) {
        // Average of two middle elements (matches numpy behavior)
        std::nth_element(values.begin(), values.begin() + static_cast<long>(mid - 1), values.end());
        return (values[mid - 1] + mid_val) / 2.0;
    }
    return mid_val;
}

/**
 * @brief Compute Median Absolute Deviation: median(|values - median(values)|).
 *
 * @param values  Input values (copied)
 * @return MAD value, or 0.0 if empty
 */
inline double computeMAD(std::vector<double> values) {
    if (values.empty()) return 0.0;

    double med = computeMedian(values);
    for (auto& v : values) {
        v = std::abs(v - med);
    }
    return computeMedian(std::move(values));
}

/**
 * @brief Compute element-wise absolute deviation from median.
 *
 * @param values  Input values
 * @return Vector of |values[i] - median(values)|
 */
inline std::vector<double> absDeviation(const std::vector<double>& values) {
    if (values.empty()) return {};
    double med = computeMedian(values);
    std::vector<double> result;
    result.reserve(values.size());
    for (double v : values) {
        result.push_back(std::abs(v - med));
    }
    return result;
}

} // namespace gpnp

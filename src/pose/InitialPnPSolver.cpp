#include "pose/InitialPnPSolver.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <iostream>

namespace gpnp {

InitialPnPSolver::InitialPnPSolver(int gpnp_min_pts)
    : gpnp_min_pts_(gpnp_min_pts)
{
}

// ============================================================================
// Pose Estimation
// ============================================================================

PoseEstimate InitialPnPSolver::solve(
    const MatchResult& match_result,
    const std::vector<Eigen::Vector3d>& pts_3d_T,
    const Eigen::Matrix3d& K)
{
    PoseEstimate result;
    result.success = false;

    const auto& good_matches = match_result.good_matches;
    const auto& pts_left_match = match_result.pts_left_match;

    // Hardcoded fallback (matching Python default)
    const Eigen::Vector3d t_hardcoded(0.0, 0.0, 500.0); // 500mm depth

    if (static_cast<int>(good_matches.size()) < gpnp_min_pts_) {
        std::cout << "  [InitialPnP] Skip | good_matches=" << good_matches.size()
                  << " < " << gpnp_min_pts_ << std::endl;
        result.R = Eigen::Matrix3d::Identity();
        result.t = t_hardcoded;
        return result;
    }

    // Extract 3D-2D correspondences
    std::vector<cv::Point3d> object_pts;
    std::vector<cv::Point2d> image_pts;
    object_pts.reserve(good_matches.size());
    image_pts.reserve(good_matches.size());

    for (size_t i = 0; i < good_matches.size(); ++i) {
        int train_idx = good_matches[i].trainIdx;
        if (train_idx < 0 || train_idx >= static_cast<int>(pts_3d_T.size())) continue;
        if (i >= pts_left_match.size()) continue;

        const auto& p3d = pts_3d_T[train_idx];
        object_pts.emplace_back(p3d.x(), p3d.y(), p3d.z());
        image_pts.emplace_back(pts_left_match[i].x, pts_left_match[i].y);
    }

    int n_pts = static_cast<int>(object_pts.size());
    if (n_pts < gpnp_min_pts_) {
        std::cout << "  [InitialPnP] Skip | n_pts=" << n_pts
                  << " < " << gpnp_min_pts_ << std::endl;
        result.R = Eigen::Matrix3d::Identity();
        result.t = t_hardcoded;
        return result;
    }

    std::cout << "  [InitialPnP] " << n_pts << " 3D↔2D correspondences" << std::endl;

    // Convert K to cv::Mat
    cv::Mat K_cv = (cv::Mat_<double>(3, 3) <<
        K(0,0), K(0,1), K(0,2),
        K(1,0), K(1,1), K(1,2),
        K(2,0), K(2,1), K(2,2));

    try {
        // ---- Step 1: RANSAC PnP ----
        // Pre-initialize rvec/tvec to zero — OpenCV 4.5.4's solvePnPGeneric
        // (called internally by solvePnPRansac) requires non-empty initial
        // guess when using SOLVEPNP_ITERATIVE, unlike the Python wrapper
        // which pre-allocates these internally.
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat inliers;
        bool ransac_ok = cv::solvePnPRansac(
            object_pts, image_pts, K_cv, cv::noArray(),
            rvec, tvec, true,  // useExtrinsicGuess=true (required for ITERATIVE)
            300,               // iterationsCount
            8.0,               // reprojectionError
            0.99,              // confidence
            inliers,
            cv::SOLVEPNP_ITERATIVE);

        if (!ransac_ok || rvec.empty() || tvec.empty()) {
            throw std::runtime_error("RANSAC returned failure");
        }

        int n_inliers = inliers.rows;
        if (n_inliers < gpnp_min_pts_) {
            throw std::runtime_error(
                "RANSAC inliers insufficient: " + std::to_string(n_inliers) +
                "/" + std::to_string(n_pts) + " < " + std::to_string(gpnp_min_pts_));
        }

        // ---- Step 2: Refinement with inlier subset ----
        std::vector<cv::Point3d> obj_inliers;
        std::vector<cv::Point2d> img_inliers;
        obj_inliers.reserve(n_inliers);
        img_inliers.reserve(n_inliers);

        for (int i = 0; i < n_inliers; ++i) {
            int idx = inliers.at<int>(i);
            obj_inliers.push_back(object_pts[idx]);
            img_inliers.push_back(image_pts[idx]);
        }

        // Refine using RANSAC result as initial guess
        cv::Mat rvec_ref = rvec.clone();
        cv::Mat tvec_ref = tvec.clone();
        bool ref_ok = cv::solvePnP(
            obj_inliers, img_inliers, K_cv, cv::noArray(),
            rvec_ref, tvec_ref,
            true, // useExtrinsicGuess (warm-start from RANSAC)
            cv::SOLVEPNP_ITERATIVE);

        if (!ref_ok) {
            throw std::runtime_error("Refinement solvePnP failed");
        }

        cv::Mat t_candidate = tvec_ref.reshape(1, 3);

        // ---- Step 3: Validity checks ----
        double tz = t_candidate.at<double>(2);

        // Check 1: Camera must be in front of template plane
        if (tz < 0.0) {
            throw std::runtime_error("t[2]=" + std::to_string(tz) + " < 0 (camera behind plane)");
        }

        // Check 2: Translation magnitude range
        double t_norm = cv::norm(t_candidate);
        if (t_norm < 10.0 || t_norm > 20000.0) {
            throw std::runtime_error("t range anomaly |t|=" + std::to_string(t_norm));
        }

        // Check 3: Finite values
        if (!std::isfinite(t_norm)) {
            throw std::runtime_error("t contains NaN/Inf");
        }

        // Check 4: Rotation matrix validity
        cv::Mat R_cv;
        cv::Rodrigues(rvec_ref, R_cv);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                if (!std::isfinite(R_cv.at<double>(r, c))) {
                    throw std::runtime_error("R contains NaN/Inf");
                }
            }
        }

        // ---- All checks passed ----
        // Convert to Eigen
        result.R << R_cv.at<double>(0,0), R_cv.at<double>(0,1), R_cv.at<double>(0,2),
                    R_cv.at<double>(1,0), R_cv.at<double>(1,1), R_cv.at<double>(1,2),
                    R_cv.at<double>(2,0), R_cv.at<double>(2,1), R_cv.at<double>(2,2);

        result.t << t_candidate.at<double>(0),
                    t_candidate.at<double>(1),
                    t_candidate.at<double>(2);

        result.success = true;
        result.num_points = n_inliers;

        std::cout << "  [InitialPnP] Success | n=" << n_pts
                  << " | inliers=" << n_inliers
                  << " | t=[" << result.t(0) << "," << result.t(1) << "," << result.t(2) << "]"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  [InitialPnP] Failed (" << e.what()
                  << ") → using hardcoded initial t=["
                  << t_hardcoded(0) << "," << t_hardcoded(1) << "," << t_hardcoded(2) << "]"
                  << std::endl;
        result.R = Eigen::Matrix3d::Identity();
        result.t = t_hardcoded;
        result.success = false;
    }

    return result;
}

} // namespace gpnp

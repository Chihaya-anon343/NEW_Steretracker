#pragma once

/**
 * @file OpticalFlowTracker.hpp
 * @brief Lucas-Kanade optical flow tracking with Forward-Backward check
 *        and MAD-based outlier filtering.
 *
 * Module: feature
 * Function: Track features from left to right stereo image via LK optical flow,
 *           validate with FB check, and filter outliers using Median Absolute
 *           Deviation (MAD) on disparity components.
 * Input:   left_gray, right_gray (CV_8UC1), pts_left (vector<Point2f>)
 * Output:  TrackResult {pts_left_good, pts_right_good, idx_from_filtered, disparity}
 * Dependencies: OpenCV video/tracking, GeometryUtils.hpp
 * Relations: Used by StereoTracker; output feeds StereoProjector and GPnPSolver
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

namespace gpnp {

class OpticalFlowTracker {
public:
    /**
     * @brief Construct optical flow tracker.
     * @param params  Lucas-Kanade parameters
     */
    explicit OpticalFlowTracker(const LKParams& params = LKParams{});

    ~OpticalFlowTracker() = default;

    // ========================================================================
    // Main Tracking API
    // ========================================================================

    /**
     * @brief Track features from left to right image.
     *
     * Pipeline:
     *   1. LK optical flow L→R
     *   2. LK optical flow R→L (backward)
     *   3. Forward-Backward consistency check (error < 1.0 px)
     *   4. MAD outlier filter on dx (positive disparity only) and dy
     *   5. Degrade gracefully if filtered points < 3
     *
     * @param left_gray   Left camera grayscale image
     * @param right_gray  Right camera grayscale image
     * @param pts_left    Left image feature points (N×1×2 or N×2 layout)
     * @return TrackResult with valid stereo correspondences
     */
    TrackResult track(const cv::Mat& left_gray,
                      const cv::Mat& right_gray,
                      const std::vector<cv::Point2f>& pts_left);

    // ========================================================================
    // Accessors
    // ========================================================================

    const LKParams& params() const { return params_; }

private:
    LKParams params_;

    /// Compute median of a vector<double> (used internally for MAD).
    static double median(std::vector<double> values);

    /// Compute MAD: median(|x - median(x)|).
    static double mad(std::vector<double> values);
};

} // namespace gpnp

#pragma once

/**
 * @file MadDisparityFilter.hpp
 * @brief MAD-based disparity outlier filter (Python 0626 algorithm).
 *
 * Module: feature
 * Function: Filter stereo-matched point pairs by disparity consistency.
 *           dx: Median Absolute Deviation (MAD) filter
 *           dy: 2-sigma filter on dx-valid subset only
 * Input:   pts_left, pts_right (N×2), idx_from_good (N ints)
 * Output:  MadFilterResult with filtered points, disparity, mask, etc.
 * Dependencies: Types.hpp, GeometryUtils.hpp
 * Relations: Used by StereoTracker at process() level (full-image coords)
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace gpnp {

class MadDisparityFilter {
public:
    /**
     * @brief Construct MAD disparity filter.
     * @param min_pts_threshold  Minimum points to apply filter (below → skip)
     * @param mad_factor         MAD multiplier for dx threshold (default 3.0)
     */
    explicit MadDisparityFilter(int min_pts_threshold = 3, double mad_factor = 3.0);

    ~MadDisparityFilter() = default;

    // ========================================================================
    // Filtering
    // ========================================================================

    /**
     * @brief Apply MAD disparity filtering to stereo-matched point pairs.
     *
     * Algorithm (matching Python 0626 MadDisparityFilter.filter):
     *   1. Compute dx = left.x - right.x, dy = left.y - right.y
     *   2. dx filter: MAD (|dx - median(dx)| > mad_factor * max(mad(dx), 1.0)) AND dx > 0
     *   3. dy filter: 2-sigma (|dy - mean(dy[dx_valid])| > 2.5 * max(std(dy[dx_valid]), 0.5))
     *   4. Combine masks: dx_valid AND dy_valid
     *   5. Degrade gracefully if filtered points < min_pts_threshold
     *
     * @param pts_left       Left image matched points (N×2, full-image coords)
     * @param pts_right      Right image matched points (N×2, full-image coords)
     * @param idx_from_good  Indices mapping to original AKAZE keypoints
     * @return MadFilterResult with filtered points, disparity, mask, idx
     */
    MadFilterResult filter(const std::vector<cv::Point2f>& pts_left,
                           const std::vector<cv::Point2f>& pts_right,
                           const std::vector<int>& idx_from_good);

private:
    int min_pts_threshold_;
    double mad_factor_;

    /// Compute mean of a vector.
    static double mean(const std::vector<double>& v);

    /// Compute standard deviation (population std, matching numpy default).
    static double stdev(const std::vector<double>& v, double mean_val);
};

} // namespace gpnp

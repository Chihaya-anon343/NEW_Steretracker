#include "feature/MadDisparityFilter.hpp"
#include "common/GeometryUtils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace gpnp {

MadDisparityFilter::MadDisparityFilter(int min_pts_threshold, double mad_factor)
    : min_pts_threshold_(min_pts_threshold)
    , mad_factor_(mad_factor)
{
}

// ============================================================================
// Statistics helpers
// ============================================================================

double MadDisparityFilter::mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double MadDisparityFilter::stdev(const std::vector<double>& v, double mean_val) {
    if (v.size() <= 1) return 0.0;
    double sum_sq = 0.0;
    for (double x : v) sum_sq += (x - mean_val) * (x - mean_val);
    return std::sqrt(sum_sq / static_cast<double>(v.size()));
}

// ============================================================================
// Filter
// ============================================================================

MadFilterResult MadDisparityFilter::filter(
    const std::vector<cv::Point2f>& pts_left,
    const std::vector<cv::Point2f>& pts_right,
    const std::vector<int>& idx_from_good)
{
    MadFilterResult result;
    const int N = static_cast<int>(pts_left.size());
    if (N == 0) return result;

    // ---- Step 1: Compute dx, dy ----
    std::vector<double> dx(N), dy(N);
    for (int i = 0; i < N; ++i) {
        dx[i] = static_cast<double>(pts_left[i].x - pts_right[i].x);
        dy[i] = static_cast<double>(pts_left[i].y - pts_right[i].y);
    }

    std::vector<bool> dx_valid(N), dy_valid(N);
    std::vector<double> dx_filt, dy_filt;
    std::vector<cv::Point2f> pts_left_filt, pts_right_filt;
    std::vector<int> idx_filt;

    if (N >= min_pts_threshold_) {
        // ---- Step 2: dx MAD filter ----
        double dx_median = computeMedian(dx);
        std::vector<double> dx_absdev(N);
        for (int i = 0; i < N; ++i) dx_absdev[i] = std::abs(dx[i] - dx_median);
        double dx_mad = computeMedian(dx_absdev);
        double dx_thresh = mad_factor_ * std::max(dx_mad, 1.0);

        for (int i = 0; i < N; ++i) {
            dx_valid[i] = (dx[i] > 0.0) && (std::abs(dx[i] - dx_median) < dx_thresh);
        }

        // ---- Step 3: dy 2-sigma filter on dx-valid subset ----
        std::vector<double> dy_dx_valid;
        for (int i = 0; i < N; ++i) {
            if (dx_valid[i]) dy_dx_valid.push_back(dy[i]);
        }

        double dy_mean_val;
        double dy_std_val;
        if (!dy_dx_valid.empty()) {
            dy_mean_val = mean(dy_dx_valid);
            dy_std_val = stdev(dy_dx_valid, dy_mean_val);
        } else {
            dy_mean_val = mean(dy);
            dy_std_val = stdev(dy, dy_mean_val);
        }
        double dy_thresh = 2.5 * std::max(dy_std_val, 0.5);

        for (int i = 0; i < N; ++i) {
            dy_valid[i] = (std::abs(dy[i] - dy_mean_val) < dy_thresh);
        }

        // ---- Step 4: Combine masks & apply ----
        result.filter_mask.resize(N);
        for (int i = 0; i < N; ++i) {
            result.filter_mask[i] = dx_valid[i] && dy_valid[i];
            if (result.filter_mask[i]) {
                pts_left_filt.push_back(pts_left[i]);
                pts_right_filt.push_back(pts_right[i]);
                dx_filt.push_back(dx[i]);
                dy_filt.push_back(dy[i]);
                idx_filt.push_back(idx_from_good[i]);
            }
        }
    } else {
        // Too few points: skip filter, keep all
        result.filter_mask.assign(N, true);
        pts_left_filt = pts_left;
        pts_right_filt = pts_right;
        dx_filt = dx;
        dy_filt = dy;
        idx_filt = idx_from_good;
    }

    // ---- Step 5: Degrade gracefully if too few survive ----
    result.downgraded = false;
    if (static_cast<int>(pts_left_filt.size()) < min_pts_threshold_) {
        pts_left_filt = pts_left;
        pts_right_filt = pts_right;
        dx_filt = dx;
        dy_filt = dy;
        idx_filt = idx_from_good;
        result.downgraded = true;
        result.filter_mask.assign(N, true);
    }

    // ---- Step 6: Build output ----
    result.pts_left_filtered = std::move(pts_left_filt);
    result.pts_right_filtered = std::move(pts_right_filt);
    result.dx_filtered = std::move(dx_filt);
    result.dy_filtered = std::move(dy_filt);

    // disparity = -dx_filtered
    result.disparity.reserve(result.dx_filtered.size());
    for (double d : result.dx_filtered) {
        result.disparity.push_back(-d);
    }

    // idx_from_filtered (matching Python logic)
    if (result.downgraded) {
        result.idx_from_filtered = idx_from_good;
    } else if (N >= min_pts_threshold_) {
        bool all_pass = true;
        for (bool m : result.filter_mask) { if (!m) { all_pass = false; break; } }
        if (all_pass) {
            result.idx_from_filtered = idx_from_good;
        } else {
            result.idx_from_filtered = std::move(idx_filt);
        }
    } else {
        result.idx_from_filtered = idx_from_good;
    }

    return result;
}

} // namespace gpnp

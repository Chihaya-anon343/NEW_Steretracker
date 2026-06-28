#pragma once

/**
 * @file RoiGenerator.hpp
 * @brief Converts YOLO Detection results to RoiRect for the stereo tracker.
 *
 * Handles class filtering, confidence-based selection, ROI expansion,
 * boundary clamping, and minimum size enforcement.
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace gpnp {

// ============================================================================
// ROI Generator
// ============================================================================

class RoiGenerator {
public:
    struct Config {
        int target_class_id{0};         ///< Only consider detections of this class
        float roi_expand_ratio{0.1f};   ///< Expand ROI by this fraction on each side
        int roi_min_size{100};          ///< Minimum ROI width and height in pixels
    };

    RoiGenerator();
    explicit RoiGenerator(const Config& cfg);

    /// Generate a single ROI from detections.
    /// Returns valid RoiRect on success, or invalid (width=0) RoiRect if no detection found.
    RoiRect generate(const std::vector<Detection>& detections,
                     const cv::Size& image_size) const;

    /// Generate left/right ROI pair from separate detection sets.
    /// If right detections are empty, left ROI is copied to right.
    std::pair<RoiRect, RoiRect> generateStereo(
        const std::vector<Detection>& detections_left,
        const std::vector<Detection>& detections_right,
        const cv::Size& left_img_size,
        const cv::Size& right_img_size) const;

private:
    Config config_;

    /// Convert a single detection to an expanded, clamped RoiRect.
    static RoiRect detectionToRoi(const Detection& det, const cv::Size& img_size,
                                   float expand_ratio, int min_size);
};

} // namespace gpnp

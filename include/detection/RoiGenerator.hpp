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

    /// Generate a single ROI from detections for the specified class.
    /// @param class_id  Target class to filter; -1 uses config_.target_class_id.
    /// Returns valid RoiRect on success, or invalid (width=0) RoiRect if no detection found.
    RoiRect generate(const std::vector<Detection>& detections,
                     const cv::Size& image_size,
                     int class_id = -1) const;

    /// Generate at most two ROIs: class 0 (always), class 1 (only if class 0 area > 700*700).
    /// Returns RoiGroup with is_dual=true when secondary is present.
    RoiGroup generateGroup(const std::vector<Detection>& detections,
                           const cv::Size& image_size) const;

    /// Generate left/right ROI pair from separate detection sets.
    /// If right detections are empty, left ROI is copied to right.
    std::pair<RoiRect, RoiRect> generateStereo(
        const std::vector<Detection>& detections_left,
        const std::vector<Detection>& detections_right,
        const cv::Size& left_img_size,
        const cv::Size& right_img_size) const;

    /// Generate left/right RoiGroup pair from separate detection sets.
    /// Supports dual-ROI mode when both sides have class 0 area > 700*700.
    std::pair<RoiGroup, RoiGroup> generateStereoGroup(
        const std::vector<Detection>& detections_left,
        const std::vector<Detection>& detections_right,
        const cv::Size& left_img_size,
        const cv::Size& right_img_size) const;

private:
    Config config_;

    /// Convert a single detection to an expanded, clamped RoiRect.
    static RoiRect detectionToRoi(const Detection& det, const cv::Size& img_size,
                                   float expand_ratio, int min_size);

    /// Normalize a stereo ROI pair: right-anchored resize, clamp, size sync.
    static void normalizeStereoPair(RoiRect& left, RoiRect& right,
                                    const Detection* right_det,
                                    const cv::Size& left_img_size,
                                    const cv::Size& right_img_size);
};

} // namespace gpnp

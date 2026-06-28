/**
 * @file RoiGenerator.cpp
 * @brief Implementation of YOLO detection → RoiRect conversion.
 */

#include "detection/RoiGenerator.hpp"

#include <algorithm>
#include <cmath>

namespace gpnp {

// ============================================================================
// Construction
// ============================================================================

RoiGenerator::RoiGenerator()
    : config_{}
{}

RoiGenerator::RoiGenerator(const Config& cfg)
    : config_(cfg)
{}

// ============================================================================
// Single ROI Generation
// ============================================================================

RoiRect RoiGenerator::generate(const std::vector<Detection>& detections,
                                const cv::Size& image_size) const {
    if (detections.empty()) {
        return RoiRect{};
    }

    // Find the highest-confidence detection matching the target class
    const Detection* best = nullptr;
    float best_conf = -1.0f;

    for (const auto& det : detections) {
        if (det.class_id == config_.target_class_id || config_.target_class_id < 0) {
            if (det.confidence > best_conf) {
                best_conf = det.confidence;
                best = &det;
            }
        }
    }

    if (best == nullptr) {
        return RoiRect{};
    }

    return detectionToRoi(*best, image_size,
                          config_.roi_expand_ratio, config_.roi_min_size);
}

// ============================================================================
// Stereo ROI Pair Generation
// ============================================================================

std::pair<RoiRect, RoiRect> RoiGenerator::generateStereo(
    const std::vector<Detection>& detections_left,
    const std::vector<Detection>& detections_right,
    const cv::Size& left_img_size,
    const cv::Size& right_img_size) const {

    // 1. Generate left ROI from left detections
    RoiRect left_roi = generate(detections_left, left_img_size);
    if (!left_roi.valid()) {
        return {};
    }

    // 2. Generate right ROI from right detections
    RoiRect right_roi = generate(detections_right, right_img_size);
    if (!right_roi.valid()) {
        return {};
    }

    // 3. If sizes differ, resize right ROI to match left size, anchored at right center
    if (left_roi.width != right_roi.width || left_roi.height != right_roi.height) {
        // Find right detection center
        const Detection* best_right = nullptr;
        float best_conf = -1.0f;
        for (const auto& det : detections_right) {
            if (det.class_id == config_.target_class_id || config_.target_class_id < 0) {
                if (det.confidence > best_conf) {
                    best_conf = det.confidence;
                    best_right = &det;
                }
            }
        }
        if (best_right != nullptr) {
            float rx_center = best_right->bbox.x + best_right->bbox.width  / 2.0f;
            float ry_center = best_right->bbox.y + best_right->bbox.height / 2.0f;
            right_roi.x      = static_cast<int>(std::round(rx_center - left_roi.width  / 2.0f));
            right_roi.y      = static_cast<int>(std::round(ry_center - left_roi.height / 2.0f));
            right_roi.width  = left_roi.width;
            right_roi.height = left_roi.height;
        }
    }

    // 4. Clamp to right image bounds
    right_roi.x = std::max(0, right_roi.x);
    right_roi.y = std::max(0, right_roi.y);
    if (right_roi.x + right_roi.width > right_img_size.width) {
        right_roi.width = right_img_size.width - right_roi.x;
    }
    if (right_roi.y + right_roi.height > right_img_size.height) {
        right_roi.height = right_img_size.height - right_roi.y;
    }

    // 5. Sync dimensions: optical flow requires identical sizes
    int w = std::min(left_roi.width, right_roi.width);
    int h = std::min(left_roi.height, right_roi.height);
    left_roi.width  = w;
    left_roi.height = h;
    right_roi.width = w;
    right_roi.height = h;

    return {left_roi, right_roi};
}

// ============================================================================
// Detection → RoiRect Conversion
// ============================================================================

RoiRect RoiGenerator::detectionToRoi(const Detection& det,
                                      const cv::Size& img_size,
                                      float expand_ratio, int min_size) {
    const float bw = det.bbox.width;
    const float bh = det.bbox.height;

    // Expand bounding box
    const float dx = bw * expand_ratio;
    const float dy = bh * expand_ratio;

    float x = det.bbox.x - dx;
    float y = det.bbox.y - dy;
    float w = bw + 2.0f * dx;
    float h = bh + 2.0f * dy;

    // Clamp to image boundaries
    x = std::max(0.0f, x);
    y = std::max(0.0f, y);
    if (x + w > static_cast<float>(img_size.width)) {
        w = static_cast<float>(img_size.width) - x;
    }
    if (y + h > static_cast<float>(img_size.height)) {
        h = static_cast<float>(img_size.height) - y;
    }

    // Enforce minimum size
    w = std::max(w, static_cast<float>(min_size));
    h = std::max(h, static_cast<float>(min_size));

    // Re-clamp after min-size enforcement
    if (x + w > static_cast<float>(img_size.width)) {
        x = std::max(0.0f, static_cast<float>(img_size.width) - w);
    }
    if (y + h > static_cast<float>(img_size.height)) {
        y = std::max(0.0f, static_cast<float>(img_size.height) - h);
    }

    RoiRect roi;
    roi.x = static_cast<int>(std::round(x));
    roi.y = static_cast<int>(std::round(y));
    roi.width = static_cast<int>(std::round(w));
    roi.height = static_cast<int>(std::round(h));

    return roi;
}

} // namespace gpnp

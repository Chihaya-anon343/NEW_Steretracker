/**
 * @file RoiGenerator.cpp
 * @brief Implementation of YOLO detection → RoiRect conversion.
 */

#include "detection/RoiGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace gpnp {

// ============================================================================
// Constants
// ============================================================================

/// Class 0 bounding box area threshold for triggering dual-ROI mode.
/// When class 0 detection area exceeds this, class 1 ROI is also extracted.
static constexpr int kDualTriggerArea = 700 * 700;  // 490000 px²

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
                                const cv::Size& image_size,
                                int class_id) const {
    if (detections.empty()) {
        return RoiRect{};
    }

    // Resolve effective class ID: explicit arg takes precedence over config
    int effective_class_id = (class_id >= 0) ? class_id : config_.target_class_id;

    // Find the highest-confidence detection matching the target class
    const Detection* best = nullptr;
    float best_conf = -1.0f;

    for (const auto& det : detections) {
        if (det.class_id == effective_class_id || effective_class_id < 0) {
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
// Dual-Class ROI Generation
// ============================================================================

RoiGroup RoiGenerator::generateGroup(const std::vector<Detection>& detections,
                                     const cv::Size& image_size) const {
    // 1. Always try class 0 first
    RoiRect class0_roi = generate(detections, image_size, 0);
    if (!class0_roi.valid()) {
        return RoiGroup{};
    }

    // 2. Check if class 0 area exceeds threshold → also extract class 1
    int area = class0_roi.width * class0_roi.height;
    if (area > kDualTriggerArea) {
        RoiRect class1_roi = generate(detections, image_size, 1);
        if (class1_roi.valid()) {
            std::cout << "[RoiGenerator] Dual-ROI mode triggered"
                      << " (class0 area=" << area << " > " << kDualTriggerArea << ")"
                      << "  primary=" << class0_roi.width << "x" << class0_roi.height
                      << "  secondary=" << class1_roi.width << "x" << class1_roi.height
                      << std::endl;
            return RoiGroup{class0_roi, class1_roi, true};
        }
    }

    return RoiGroup{class0_roi, {}, false};
}

// ============================================================================
// Stereo Pair Normalization (shared helper)
// ============================================================================

void RoiGenerator::normalizeStereoPair(RoiRect& left, RoiRect& right,
                                       const Detection* right_det,
                                       const cv::Size& left_img_size,
                                       const cv::Size& right_img_size) {
    // 1. If sizes differ, resize right ROI to match left size, anchored at right center
    if (left.width != right.width || left.height != right.height) {
        if (right_det != nullptr) {
            float rx_center = right_det->bbox.x + right_det->bbox.width  / 2.0f;
            float ry_center = right_det->bbox.y + right_det->bbox.height / 2.0f;
            right.x      = static_cast<int>(std::round(rx_center - left.width  / 2.0f));
            right.y      = static_cast<int>(std::round(ry_center - left.height / 2.0f));
            right.width  = left.width;
            right.height = left.height;
        }
    }

    // 2. Clamp to right image bounds
    right.x = std::max(0, right.x);
    right.y = std::max(0, right.y);
    if (right.x + right.width > right_img_size.width) {
        right.width = right_img_size.width - right.x;
    }
    if (right.y + right.height > right_img_size.height) {
        right.height = right_img_size.height - right.y;
    }

    // 3. Sync dimensions: optical flow requires identical sizes
    int w = std::min(left.width, right.width);
    int h = std::min(left.height, right.height);
    left.width  = w;
    left.height = h;
    right.width = w;
    right.height = h;
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
    RoiRect left_roi = generate(detections_left, left_img_size, config_.target_class_id);
    if (!left_roi.valid()) {
        return {};
    }

    // 2. Generate right ROI from right detections
    RoiRect right_roi = generate(detections_right, right_img_size, config_.target_class_id);
    if (!right_roi.valid()) {
        return {};
    }

    // 3. Find right detection for center-anchoring (needed by normalizeStereoPair)
    const Detection* best_right = nullptr;
    {
        int effective_class = config_.target_class_id;
        float best_conf = -1.0f;
        for (const auto& det : detections_right) {
            if (det.class_id == effective_class || effective_class < 0) {
                if (det.confidence > best_conf) {
                    best_conf = det.confidence;
                    best_right = &det;
                }
            }
        }
    }

    // 4. Normalize the stereo pair
    normalizeStereoPair(left_roi, right_roi, best_right, left_img_size, right_img_size);

    return {left_roi, right_roi};
}

// ============================================================================
// Stereo RoiGroup Pair Generation (dual-class aware)
// ============================================================================

std::pair<RoiGroup, RoiGroup> RoiGenerator::generateStereoGroup(
    const std::vector<Detection>& detections_left,
    const std::vector<Detection>& detections_right,
    const cv::Size& left_img_size,
    const cv::Size& right_img_size) const {

    // 1. Generate RoiGroup for each side independently
    RoiGroup left_group  = generateGroup(detections_left,  left_img_size);
    RoiGroup right_group = generateGroup(detections_right, right_img_size);

    if (!left_group.valid() || !right_group.valid()) {
        return {};
    }

    // 2. Normalize primary stereo pair
    if (left_group.primary.valid() && right_group.primary.valid()) {
        // Find right class-0 detection for center anchoring
        const Detection* best_right0 = nullptr;
        {
            float best_conf = -1.0f;
            for (const auto& det : detections_right) {
                if (det.class_id == 0) {
                    if (det.confidence > best_conf) {
                        best_conf = det.confidence;
                        best_right0 = &det;
                    }
                }
            }
        }
        normalizeStereoPair(left_group.primary, right_group.primary,
                           best_right0, left_img_size, right_img_size);
    }

    // 3. Normalize secondary stereo pair (only when both sides are dual)
    if (left_group.is_dual && right_group.is_dual) {
        if (left_group.secondary.valid() && right_group.secondary.valid()) {
            const Detection* best_right1 = nullptr;
            {
                float best_conf = -1.0f;
                for (const auto& det : detections_right) {
                    if (det.class_id == 1) {
                        if (det.confidence > best_conf) {
                            best_conf = det.confidence;
                            best_right1 = &det;
                        }
                    }
                }
            }
            normalizeStereoPair(left_group.secondary, right_group.secondary,
                               best_right1, left_img_size, right_img_size);
        }
    } else {
        // If only one side is dual, clear secondary on both sides for consistency
        left_group.is_dual = false;
        left_group.secondary = RoiRect{};
        right_group.is_dual = false;
        right_group.secondary = RoiRect{};
    }

    return {left_group, right_group};
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

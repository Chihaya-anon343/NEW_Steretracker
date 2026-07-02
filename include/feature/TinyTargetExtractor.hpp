#pragma once

/**
 * @file TinyTargetExtractor.hpp
 * @brief Feature extraction for tiny rectangular targets (≤800 px area).
 *
 * Uses super-resolution + minAreaRect + cornerSubPix.
 * Selected when ROI area ≤ 800 px.
 *
 * Pipeline:
 *   1. Otsu → template IoU match → best angle
 *   2. Super-resolution (×scale_factor) → Otsu → morph open+close
 *   3. Connected-component scoring (rect_ratio, area, center, aspect)
 *   4. minAreaRect → 4 corners → cornerSubPix refinement
 *   5. Angle alignment → scale back → 4 ordered corners
 *
 * Migrated from legacy/TinyTargetPose.cpp (original Python port).
 */

#include "feature/FeatureExtractor.hpp"
#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace gpnp {

class TinyTargetExtractor : public FeatureExtractor {
public:
    struct Config {
        cv::Size target_size{50, 50};         ///< Template match normalization size
        int scale_factor = 4;                 ///< Super-resolution upscale factor (2~8)
        double square_size_m = 0.20;          ///< Physical target side length (meters)
        int roi_pad_pixels = 0;               ///< Expand ROI by N pixels on all sides
    };

    /// Construct with config and template directory.
    /// Templates are loaded from the NewMuBan directory via PoseUtils.
    TinyTargetExtractor(const Config& config, const std::string& template_dir);

    ~TinyTargetExtractor() override = default;

    // ---- FeatureExtractor interface ----

    std::string name() const override { return "TinyTarget"; }

    StrategyType strategyType() const override { return StrategyType::TinyTarget; }

    void setTemplateData(const std::string& template_dir,
                         double real_width_mm,
                         double real_height_mm) override;

    /// Full pipeline: extract 4 corners from left & right, fill PipelineResult
    /// for stereo GPNP (same pattern as BinaryCornerExtractor).
    PipelineResult extract(const cv::Mat& left_gray,
                           const cv::Mat& right_gray,
                           const cv::Mat& left_color,
                           const cv::Mat& right_color) override;

    const TemplateData& templateData() const override { return template_data_; }

    // ---- Post-extraction state ----

    int lastMatchedAngle() const { return last_best_angle_; }
    double lastMatchOverlap() const { return last_best_overlap_; }

    /// Return the matched template (for visualization), or nullptr.
    const TemplateData* lastMatchedTemplate() const {
        if (last_best_angle_ < 0) return nullptr;
        for (const auto& t : templates_)
            if (t.angle == last_best_angle_) return &t;
        return nullptr;
    }

private:
    /// Core 4-corner extraction from one grayscale ROI.
    /// Returns corners in ROI-local coordinates (TL→TR→BR→BL).
    Status extract4Corners(const cv::Mat& roi_gray,
                            std::vector<cv::Point2f>& out_corners,
                            int& best_angle,
                            double& best_overlap);

    // Template matching
    struct TemplateMatchResult {
        int best_angle = -1;
        double best_overlap = 0.0;
        std::vector<std::pair<int, double>> all_overlaps;
    };
    TemplateMatchResult matchTemplate(const cv::Mat& roi_binary);

    // Component scoring & selection
    int selectBestComponent(const cv::Mat& binary, const cv::Mat& label_map,
                            int num_labels, const cv::Mat& stats,
                            const cv::Mat& centroids);

    // Sub-pixel corner refinement
    std::vector<cv::Point2f> refineCorners(const cv::Mat& image,
                                            const std::vector<cv::Point2f>& corners,
                                            int win_size);

    Config config_;
    std::vector<TemplateData> templates_;       ///< Corner templates from NewMuBan
    TemplateData template_data_;                 ///< Stores pts_3d for GPNP

    // Post-extraction state
    int last_best_angle_ = -1;
    double last_best_overlap_ = 0.0;
};

} // namespace gpnp

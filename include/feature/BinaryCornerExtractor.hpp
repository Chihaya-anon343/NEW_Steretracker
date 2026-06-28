#pragma once

/**
 * @file BinaryCornerExtractor.hpp
 * @brief Feature extraction via binary contour analysis + approxPolyDP corner detection.
 *
 * Selected when ROI area is between 801 and 40000 px.
 *
 * Pipeline:
 *   1. Otsu binarization (internal — caller provides grayscale ROI)
 *   2. Keep largest connected component
 *   3. Fill holes
 *   4. Morphological smoothing (close → open)
 *   5. Template matching (IoU-based, 24 templates 0°~345°)
 *   6. Rotate to upright (if template matched)
 *   7. Extract largest contour
 *   8. approxPolyDP binary search → N corners
 *   9. Rotate corners back (if rotated)
 *  10. Reorder corners by template geometry
 *
 * Migrated from legacy/BinaryCorner.cpp (original Python port).
 */

#include "feature/FeatureExtractor.hpp"
#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace gpnp {

class BinaryCornerExtractor : public FeatureExtractor {
public:
    struct Config {
        int corners = 10;                     ///< Expected number of corners
        int kernel_size = 3;                  ///< Morphological kernel size (forced odd)
        double scale = 1.0;                   ///< Internal upscale for sub-pixel accuracy
        cv::Size target_size{100, 100};       ///< Template match normalization size
        double pixel_to_meter_scale = 0.0;    ///< Scale: 1 template pixel = ? meters
        int roi_pad_pixels = 0;               ///< Expand ROI by N pixels on all sides
        double otsu_ratio = 1.3;              ///< Otsu threshold multiplier (>1 = stricter)
    };

    /// Construct with config and template directory.
    /// Templates are loaded from the NewMuBan directory via PoseUtils.
    BinaryCornerExtractor(const Config& config, const std::string& template_dir);

    ~BinaryCornerExtractor() override = default;

    // ---- FeatureExtractor interface ----

    std::string name() const override { return "BinaryCorner"; }

    void setTemplateData(const std::string& template_dir,
                         double real_width_mm,
                         double real_height_mm) override;

    /// Full pipeline entry point.
    ///
    /// Input:  left_gray (CV_8UC1 grayscale ROI), right_gray/color for passthrough.
    /// Output: PipelineResult with:
    ///   - kp_left: extracted corners as KeyPoints
    ///   - pts_left_match: corner coordinates (for debug/PnP)
    ///   - pts_template_match: matched template corners scaled to ROI size
    ///   - desc_left, disparity, pts_left_good etc.: EMPTY (no stereo/AKAZE descriptors)
    PipelineResult extract(const cv::Mat& left_gray,
                           const cv::Mat& right_gray,
                           const cv::Mat& left_color,
                           const cv::Mat& right_color) override;

    const TemplateData& templateData() const override { return template_data_; }

    // ---- Post-extraction state ----

    const TemplateData* lastMatchedTemplate() const { return last_matched_template_; }
    double lastMatchOverlap() const { return last_match_overlap_; }
    const std::vector<cv::Point2f>& lastCornersBeforeReorder() const {
        return last_corners_before_reorder_;
    }

    /// 返回最近一次提取的二值图像（用于可视化调试）
    const cv::Mat& lastLeftBinary()  const { return last_left_binary_; }
    const cv::Mat& lastRightBinary() const { return last_right_binary_; }

    /// 返回旋转回正后的二值图像（用于可视化调试）
    const cv::Mat& lastUprightBinary() const { return last_upright_binary_; }

    // ---- Diagnostics ----

    const std::vector<std::pair<std::string, std::string>>& processLog() const {
        return process_log_;
    }

    // ---- Static visualization helpers ----

    static cv::Mat drawCorners(const cv::Mat& img,
                                const std::vector<cv::Point2f>& corners);

    static cv::Mat drawMatchedCorners(
        const cv::Mat& input_img,
        const std::vector<cv::Point2f>& input_corners,
        const cv::Mat& template_img,
        const std::vector<cv::Point2f>& template_corners,
        double template_angle = 0.0);

    static std::vector<cv::Mat> debugVisualizeReordering(
        const cv::Mat& binary_img,
        const cv::Mat& template_img,
        const std::vector<cv::Point2f>& binary_corners,
        const std::vector<cv::Point2f>& template_corners,
        double template_angle,
        const cv::Size& coord_size);

private:
    // ========================================================================
    // Internal pipeline (operates on binary image, same as legacy)
    // ========================================================================

    /// Core corner extraction from a binary image (0/255).
    /// @param binary_img   Otsu-binarized ROI
    /// @param gray_roi     Original grayscale ROI（用于先旋转再二值化，提高精度）
    /// @param out_corners  输出的角点（ROI局部坐标）
    Status extractFromBinary(const cv::Mat& binary_img,
                              const cv::Mat& gray_roi,
                              std::vector<cv::Point2f>& out_corners);

    cv::Mat keepLargestRegion(const cv::Mat& binary_img);
    cv::Mat keepRegionFromCenter(const cv::Mat& binary_img);
    cv::Mat fillHoles(const cv::Mat& binary_img);
    cv::Mat smoothBoundary(const cv::Mat& binary_img);
    std::vector<cv::Point> extractLargestContour(const cv::Mat& binary_img);
    std::vector<cv::Point2f> extractCornersFromContour(
        const std::vector<cv::Point>& contour, const cv::Size& img_size);

    // Template matching
    struct BestMatch {
        int template_index = -1;
        double overlap = 0.0;
    };
    BestMatch findBestMatch(const cv::Mat& binary_img);
    std::vector<std::pair<int, int>> matchCorners(
        const std::vector<cv::Point2f>& template_corners,
        const std::vector<cv::Point2f>& binary_corners,
        double template_angle,
        const cv::Size& coord_size);
    static std::vector<int> reorderByGeometry(
        const std::vector<cv::Point2f>& corners,
        const cv::Point2f& center,
        double reference_angle_deg,
        double ref_dist = -1.0);

    // Logging
    void logStep(const std::string& step, const std::string& info);

    // ========================================================================
    // State
    // ========================================================================

    Config config_;
    cv::Mat kernel_;
    std::vector<TemplateData> templates_;       ///< Corner templates from NewMuBan
    TemplateData template_data_;                 ///< Empty placeholder (no AKAZE template)

    // Post-extraction state
    const TemplateData* last_matched_template_ = nullptr;
    double last_match_overlap_ = 0.0;
    std::vector<cv::Point2f> last_corners_before_reorder_;
    std::vector<std::pair<std::string, std::string>> process_log_;

    // 最近一次提取的二值图像（用于可视化）
    cv::Mat last_left_binary_;
    cv::Mat last_right_binary_;
    cv::Mat last_upright_binary_;    // 旋转回正后的二值图
};

} // namespace gpnp

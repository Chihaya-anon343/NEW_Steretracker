#pragma once

/**
 * @file FeatureExtractor.hpp
 * @brief Abstract strategy interface for feature extraction.
 *
 * Strategies:
 *   - AkazeGpnpExtractor   (ROI area > 40000): AKAZE + LK flow + GPNP
 *   - BinaryCornerExtractor (801 ~ 40000):    contour-based corner detection
 *   - TinyTargetExtractor   (≤ 800):           super-resolution minAreaRect
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace gpnp {

/// Strategy type for degradation chain dispatch.
enum class StrategyType {
    Akaze,          ///< AKAZE feature extraction + optical flow + GPNP
    BinaryCorner,   ///< Otsu binary contour corner extraction + GPNP
    TinyTarget      ///< Super-resolution minAreaRect 4-point extraction + solvePnP
};

/// Abstract interface for feature extraction strategies.
///
/// Each strategy implements feature extraction + template matching,
/// producing a PipelineResult consumable by the downstream pipeline
/// (MAD filtering, GPNP pose estimation).
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    /// Human-readable strategy name for logging.
    virtual std::string name() const = 0;

    /// Strategy type for PnP dispatch and degradation chain ordering.
    virtual StrategyType strategyType() const = 0;

    /// Accept and load template data (called once after construction).
    ///
    /// For AKAZE strategies this extracts AKAZE features from a reference
    /// template image. For BinaryCorner/TinyTarget strategies this loads
    /// the PNG+TXT corner templates from a directory.
    ///
    /// @param template_dir     Path to template image or template directory
    /// @param real_width_mm    Physical template width (mm) — AKAZE only
    /// @param real_height_mm   Physical template height (mm) — AKAZE only
    virtual void setTemplateData(const std::string& template_dir,
                                  double real_width_mm,
                                  double real_height_mm) = 0;

    /// Extract features from a stereo ROI pair.
    ///
    /// Produces a PipelineResult fully populated for AKAZE (keypoints,
    /// descriptors, tracked points, stereo projection, template matches).
    /// For stub strategies, returns empty/zeroed result with error status.
    ///
    /// @param left_gray    Left camera grayscale ROI
    /// @param right_gray   Right camera grayscale ROI
    /// @param left_color   Left camera color ROI (for visualization)
    /// @param right_color  Right camera color ROI (for visualization)
    /// @return             PipelineResult with extracted feature data
    virtual PipelineResult extract(const cv::Mat& left_gray,
                                    const cv::Mat& right_gray,
                                    const cv::Mat& left_color,
                                    const cv::Mat& right_color) = 0;

    /// Return the template data used by this strategy.
    virtual const TemplateData& templateData() const = 0;
};

} // namespace gpnp

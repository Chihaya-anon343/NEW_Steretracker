#pragma once

/**
 * @file AkazeExtractor.hpp
 * @brief AKAZE feature detector/descriptor with scale-factor handling.
 *
 * Module: feature
 * Function: AKAZE keypoint detection + descriptor computation with optional
 *           image downscaling for speed, followed by coordinate rescaling.
 * Input:   cv::Mat grayscale image
 * Output:  FeatureSet {keypoints, descriptors, points}
 * Dependencies: OpenCV features2d, Types.hpp
 * Relations: Used by StereoTracker; feeds OpticalFlowTracker and TemplateMatcher
 */

#include "common/Types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <memory>

namespace gpnp {

class AkazeExtractor {
public:
    /**
     * @brief Construct AKAZE extractor.
     *
     * @param scale  Image downscale factor (0~1). Features are extracted on
     *               the scaled image, then keypoint coordinates are divided
     *               by scale to map back to original image coordinates.
     *               scale=1.0 means no scaling.
     */
    explicit AkazeExtractor(double scale = 0.5);

    ~AkazeExtractor();

    // Non-copyable (owns cv::Ptr)
    AkazeExtractor(const AkazeExtractor&) = delete;
    AkazeExtractor& operator=(const AkazeExtractor&) = delete;

    // Movable
    AkazeExtractor(AkazeExtractor&&) noexcept;
    AkazeExtractor& operator=(AkazeExtractor&&) noexcept;

    // ========================================================================
    // Feature Extraction
    // ========================================================================

    /**
     * @brief Extract AKAZE features from a grayscale image.
     *
     * If scale < 1.0, the image is first downscaled, features extracted,
     * then keypoint coordinates are divided by scale.
     *
     * @param gray  Input grayscale image (CV_8UC1)
     * @return FeatureSet with keypoints, descriptors, and point coordinates
     */
    FeatureSet extract(const cv::Mat& gray);

    /**
     * @brief Extract and cache template features (called once at initialization).
     *
     * Extracts features at original scale (no downscaling).
     * Also computes 3D coordinates from physical template dimensions.
     *
     * @param template_path  Path to template image file
     * @param real_width_mm  Physical width of template in mm
     * @param real_height_mm Physical height of template in mm
     * @return TemplateData ready for matching
     */
    TemplateData extractTemplate(const std::string& template_path,
                                 double real_width_mm,
                                 double real_height_mm);

    // ========================================================================
    // Accessors
    // ========================================================================

    double scale() const { return scale_; }

private:
    cv::Ptr<cv::AKAZE> akaze_;   ///< OpenCV AKAZE detector instance
    double scale_;                ///< Image scale factor
};

} // namespace gpnp

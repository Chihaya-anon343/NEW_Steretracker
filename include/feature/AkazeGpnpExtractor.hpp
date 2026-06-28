#pragma once

/**
 * @file AkazeGpnpExtractor.hpp
 * @brief AKAZE + optical flow + template matching strategy.
 *
 * Wraps the existing pipeline: AKAZE detection on left → LK optical flow
 * L→R with FB check → stereo projection → template matching.
 *
 * Selected when ROI area > 40000 px.
 */

#include "feature/FeatureExtractor.hpp"
#include "feature/AkazeExtractor.hpp"
#include "feature/OpticalFlowTracker.hpp"
#include "matching/TemplateMatcher.hpp"
#include "stereo/StereoProjector.hpp"
#include "common/Config.hpp"
#include "common/Types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace gpnp {

class AkazeGpnpExtractor : public FeatureExtractor {
public:
    /// Construct with AKAZE scale and optical flow parameters.
    /// Camera params must be set via initCamera() before extract() is called.
    AkazeGpnpExtractor(double scale, const LKParams& lk_params = LKParams{});

    ~AkazeGpnpExtractor() override = default;

    // Non-copyable (owns unique_ptr)
    AkazeGpnpExtractor(const AkazeGpnpExtractor&) = delete;
    AkazeGpnpExtractor& operator=(const AkazeGpnpExtractor&) = delete;

    // Movable
    AkazeGpnpExtractor(AkazeGpnpExtractor&&) noexcept = default;
    AkazeGpnpExtractor& operator=(AkazeGpnpExtractor&&) noexcept = default;

    // ---- FeatureExtractor interface ----

    std::string name() const override { return "AkazeGpnp"; }

    void setTemplateData(const std::string& template_dir,
                         double real_width_mm,
                         double real_height_mm) override;

    PipelineResult extract(const cv::Mat& left_gray,
                           const cv::Mat& right_gray,
                           const cv::Mat& left_color,
                           const cv::Mat& right_color) override;

    const TemplateData& templateData() const override { return template_; }

    // ---- Camera initialization (required before extract) ----

    /// Set stereo camera parameters. Must be called before extract().
    void initCamera(const StereoCameraParams& camera_params);

private:
    double scale_;
    TemplateData template_;

    AkazeExtractor akaze_extractor_;
    OpticalFlowTracker flow_tracker_;
    TemplateMatcher template_matcher_;
    std::unique_ptr<StereoProjector> projector_;  // initialized via initCamera()
};

} // namespace gpnp

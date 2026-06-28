#include "feature/FeatureExtractorFactory.hpp"

#include <iostream>

namespace gpnp {

std::unique_ptr<FeatureExtractor> createFeatureExtractor(
    int roi_area,
    double akaze_scale,
    const LKParams& lk_params,
    const std::string& template_dir,
    const BinaryCornerExtractor::Config& binary_cfg,
    const TinyTargetExtractor::Config& tiny_cfg) {

    if (roi_area >= FeatureThresholds::AKAZE_MIN_AREA) {
        std::cout << "[Factory] ROI area=" << roi_area
                  << " → AkazeGpnpExtractor (scale=" << akaze_scale << ")"
                  << std::endl;
        auto extractor = std::make_unique<AkazeGpnpExtractor>(akaze_scale, lk_params);
        return extractor;

    } else if (roi_area <= FeatureThresholds::TINY_MAX_AREA && roi_area > 0) {
        std::cout << "[Factory] ROI area=" << roi_area
                  << " → TinyTargetExtractor (stub, scale_factor="
                  << tiny_cfg.scale_factor << ")" << std::endl;
        auto extractor = std::make_unique<TinyTargetExtractor>(tiny_cfg, template_dir);
        return extractor;

    } else {
        std::cout << "[Factory] ROI area=" << roi_area
                  << " → BinaryCornerExtractor (stub, corners="
                  << binary_cfg.corners << ")" << std::endl;
        auto extractor = std::make_unique<BinaryCornerExtractor>(binary_cfg, template_dir);
        return extractor;
    }
}

} // namespace gpnp

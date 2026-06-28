#pragma once

/**
 * @file FeatureExtractorFactory.hpp
 * @brief Factory function for selecting FeatureExtractor strategy by ROI area.
 *
 * Thresholds:
 *   - ROI area >  40000 px  →  AkazeGpnpExtractor
 *   - ROI area <= 800   px  →  TinyTargetExtractor
 *   - otherwise              →  BinaryCornerExtractor
 */

#include "feature/FeatureExtractor.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "common/Config.hpp"
#include "common/Types.hpp"

#include <memory>
#include <string>

namespace gpnp {

/// Thresholds for ROI-area-based strategy selection.
struct FeatureThresholds {
    static constexpr int TINY_MAX_AREA   = 800;    ///< ≤ this → TinyTargetExtractor
    static constexpr int AKAZE_MIN_AREA  = 40001;  ///< ≥ this → AkazeGpnpExtractor
};

/// Select and construct the appropriate FeatureExtractor strategy.
///
/// @param roi_area      ROI bounding-box area (width × height) in pixels
/// @param akaze_scale   AKAZE image scale factor (0~1)
/// @param lk_params     Lucas-Kanade optical flow parameters
/// @param template_dir  Path to template directory (NewMuBan)
/// @param binary_cfg    Configuration for BinaryCornerExtractor (used when selected)
/// @param tiny_cfg      Configuration for TinyTargetExtractor (used when selected)
/// @return              Owning pointer to the selected strategy
std::unique_ptr<FeatureExtractor> createFeatureExtractor(
    int roi_area,
    double akaze_scale,
    const LKParams& lk_params,
    const std::string& template_dir,
    const BinaryCornerExtractor::Config& binary_cfg = {},
    const TinyTargetExtractor::Config& tiny_cfg = {});

} // namespace gpnp

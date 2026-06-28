#pragma once

/**
 * @file YoloRoiProvider.hpp
 * @brief Facade: YOLO detection → stereo ROI pair.
 *
 * Encapsulates YoloDetector + RoiGenerator behind a single interface.
 * main.cpp only depends on this header — no knowledge of ONNX inference
 * or ROI generation internals.
 */

#include "common/Types.hpp"
#include "detection/RoiGenerator.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <utility>

namespace gpnp {

class YoloDetector;

class YoloRoiProvider {
public:
    YoloRoiProvider();
    ~YoloRoiProvider();

    YoloRoiProvider(const YoloRoiProvider&) = delete;
    YoloRoiProvider& operator=(const YoloRoiProvider&) = delete;

    /// Initialize detector and ROI generator. Returns false on failure.
    bool initialize(const YoloConfig& yolo_cfg, const RoiGenerator::Config& roi_cfg);

    /// True if YOLO is ready for inference.
    bool isReady() const;

    /// Run YOLO on both images, return dimension-matched stereo ROI pair.
    /// Returns {invalid, invalid} if detection fails on either side.
    std::pair<RoiRect, RoiRect> detect(const cv::Mat& left_img,
                                        const cv::Mat& right_img);

private:
    std::unique_ptr<YoloDetector> detector_;
    std::unique_ptr<RoiGenerator> roi_gen_;
};

} // namespace gpnp

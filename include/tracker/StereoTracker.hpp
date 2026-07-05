#pragma once

#include "common/Config.hpp"
#include "common/Types.hpp"
#include "feature/FeatureExtractor.hpp"
#include "feature/MadDisparityFilter.hpp"
#include "pose/GPnPSolver.hpp"
#include "pose/InitialPnPSolver.hpp"
#include "visualization/Visualizer.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

namespace gpnp {

// Forward declarations for pre-initialized extractors
class AkazeGpnpExtractor;
class BinaryCornerExtractor;
class TinyTargetExtractor;

class StereoTracker {
public:
    /// Extended constructor: pre-initializes all three feature extractors.
    /// Call configureStrategyChain() per-frame to select the active chain.
    StereoTracker(const Eigen::Matrix3d& K,
                  const Eigen::Matrix3d& R_rl,
                  const Eigen::Vector3d& t_rl,
                  const std::string& template_path,
                  const TrackerConfig& config,
                  const BinaryCornerExtractor::Config& binary_cfg,
                  const std::string& binary_template_dir,
                  const TinyTargetExtractor::Config& tiny_cfg,
                  const std::string& tiny_template_dir);

    ~StereoTracker();

    StereoTracker(const StereoTracker&) = delete;
    StereoTracker& operator=(const StereoTracker&) = delete;

    // ========================================================================
    // Public API
    // ========================================================================

    /// Process one stereo frame with optional ROI.
    /// Automatically selects extraction strategy from ROI area and expands
    /// ROI with padding when using non-AKAZE extractors.
    PipelineResult process(const cv::Mat& left_img,
                           const cv::Mat& right_img,
                           bool visualize = false,
                           const RoiRect* left_roi = nullptr,
                           const RoiRect* right_roi = nullptr);

    /// Process from file paths with optional ROI.
    PipelineResult process(const std::string& left_path,
                           const std::string& right_path,
                           bool visualize = false,
                           const RoiRect* left_roi = nullptr,
                           const RoiRect* right_roi = nullptr);

    void clearCache();

    // ========================================================================
    // Logging
    // ========================================================================

    const std::vector<LogEntry>& getLogs() const;
    void printLogs() const;

    /// Set output directory for visualization images.
    void setOutputDir(const std::string& dir) { output_dir_ = dir; }

    const StereoCameraParams& cameraParams() const { return camera_; }
    const TemplateData& templateData() const { return template_; }
    const TrackerConfig& config() const { return config_; }
    int frameCount() const { return state_.frame_count; }

private:
    StereoCameraParams camera_;
    TrackerConfig config_;
    TemplateData template_;

    // ========================================================================
    // Pre-initialized extractors (created once in constructor, reused every frame)
    // ========================================================================
    std::unique_ptr<AkazeGpnpExtractor> akaze_extractor_;
    std::unique_ptr<BinaryCornerExtractor> binary_extractor_;
    std::unique_ptr<TinyTargetExtractor> tiny_extractor_;

    /// Active primary extractor (points to one of the three above).
    FeatureExtractor* extractor_ = nullptr;

    /// Active fallback chain (pointers into the three above, in order).
    std::vector<FeatureExtractor*> fallback_extractors_;

    // Strategy selection thresholds (read from config)
    int akaze_min_area_ = 40000;
    int tiny_max_area_  = 800;

    // ROI padding for non-AKAZE extractors
    int binary_roi_pad_ = 0;
    int tiny_roi_pad_   = 0;

    std::string output_dir_;

    // Subsystem modules
    InitialPnPSolver initial_pnp_;
    GPnPSolver gpnp_solver_;
    MadDisparityFilter mad_filter_;
    std::unique_ptr<Visualizer> visualizer_;

    TrackingState state_;

    // ========================================================================
    // ROI Helpers
    // ========================================================================

    static RoiRect validateRoi(const RoiRect* roi, const cv::Size& img_size,
                               const std::string& name);

    void offsetResultToOriginal(PipelineResult& result,
                                const cv::Point2d& left_offset,
                                const cv::Point2d& right_offset,
                                const cv::Mat& left_color_orig,
                                const cv::Mat& right_color_orig);

    // ========================================================================
    // Image loading
    // ========================================================================

    static std::pair<cv::Mat, cv::Mat> loadImage(const cv::Mat& img);

    // ========================================================================
    // Degradation helpers
    // ========================================================================

    /// Configure the active extraction chain based on ROI area (called automatically by process()).
    void configureStrategyChain(int roi_area);

    /// Expand ROI with padding when using non-AKAZE extractors for corner context.
    void applyRoiPadding(RoiRect& rl, RoiRect& rr, int roi_area,
                         int left_cols, int left_rows,
                         int right_cols, int right_rows) const;

    /// Run extraction + coordinate restore for one extractor on pre-cropped ROIs.
    bool runExtraction(FeatureExtractor& ext,
                       const cv::Mat& left_gray, const cv::Mat& right_gray,
                       const cv::Mat& left_color, const cv::Mat& right_color,
                       const cv::Point2d& left_offset, const cv::Point2d& right_offset,
                       const cv::Mat& left_color_orig, const cv::Mat& right_color_orig,
                       PipelineResult& result);

    /// ROI crop helper: extract ROI from stereo pair.
    struct StereoRoi {
        cv::Mat left_gray, right_gray;
        cv::Mat left_color, right_color;
        cv::Point2d left_offset, right_offset;
    };
    StereoRoi cropStereoRoi(const cv::Mat& left_img, const cv::Mat& right_img,
                            const RoiRect* left_roi, const RoiRect* right_roi);

    // ========================================================================
    // PnP dispatch — strategy-specific pose estimation
    // ========================================================================

    std::pair<bool, PoseEstimate> dispatchPnP(FeatureExtractor* ext,
                                               PipelineResult& result, bool is_first);

    std::pair<bool, PoseEstimate> runAkazePnP(PipelineResult& result, bool is_first);
    std::pair<bool, PoseEstimate> runBinaryCornerPnP(PipelineResult& result, bool is_first);
    std::pair<bool, PoseEstimate> runTinyTargetPnP(PipelineResult& result);

    void finalizePose(PipelineResult& result, const PoseEstimate& pose);

    // ========================================================================
    // Logging
    // ========================================================================

    void addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used);
};

} // namespace gpnp
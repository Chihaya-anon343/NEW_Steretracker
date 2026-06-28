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

namespace gpnp {

class StereoTracker {
public:
    StereoTracker(const Eigen::Matrix3d& K,
                  const Eigen::Matrix3d& R_rl,
                  const Eigen::Vector3d& t_rl,
                  const std::string& template_path,
                  const TrackerConfig& config = makeTrackerConfig());

    ~StereoTracker();

    StereoTracker(const StereoTracker&) = delete;
    StereoTracker& operator=(const StereoTracker&) = delete;

    // ========================================================================
    // Public API (0626: added left_roi, right_roi)
    // ========================================================================

    /// Process one stereo frame with optional ROI.
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

    /// Replace the feature extraction strategy at runtime.
    void setExtractor(std::unique_ptr<FeatureExtractor> extractor);

    /// Set output directory for visualization images (created by main.cpp).
    void setOutputDir(const std::string& dir) { output_dir_ = dir; }

    const StereoCameraParams& cameraParams() const { return camera_; }
    const TemplateData& templateData() const { return template_; }
    const TrackerConfig& config() const { return config_; }
    int frameCount() const { return state_.frame_count; }

private:
    StereoCameraParams camera_;
    TrackerConfig config_;
    TemplateData template_;

    // Subsystem modules
    std::unique_ptr<FeatureExtractor> extractor_;  ///< Feature extraction strategy
    std::string output_dir_;                       ///< Visualization output directory
    InitialPnPSolver initial_pnp_;
    GPnPSolver gpnp_solver_;
    MadDisparityFilter mad_filter_;        // 0626: MAD extracted to separate class
    std::unique_ptr<Visualizer> visualizer_;

    TrackingState state_;

    // ========================================================================
    // ROI Helpers (0626)
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
    // Logging
    // ========================================================================

    void addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used);
};

} // namespace gpnp

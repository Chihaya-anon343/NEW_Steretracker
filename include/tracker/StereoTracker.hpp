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

    /// Replace the primary feature extraction strategy at runtime.
    void setExtractor(std::unique_ptr<FeatureExtractor> extractor);

    /// Add a fallback feature extractor. Fallbacks are tried in order of registration
    /// when the primary extractor (or a higher-priority fallback) fails.
    void addFallbackExtractor(std::unique_ptr<FeatureExtractor> extractor);

    /// Clear all registered fallback extractors (for per-frame re-registration).
    void clearFallbackExtractors();

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
    // Degradation: fallback extraction chain
    // ========================================================================

    /// Fallback extractors, tried in registration order when primary fails.
    std::vector<std::unique_ptr<FeatureExtractor>> fallback_extractors_;

    // ========================================================================
    // Degradation helpers
    // ========================================================================

    /// Run extraction + coordinate restore for one extractor on pre-cropped ROIs.
    /// Returns true if extraction produced meaningful results.
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

    /// Dispatch PnP based on strategy type, returning {ok, pose}.
    std::pair<bool, PoseEstimate> dispatchPnP(FeatureExtractor* ext,
                                               PipelineResult& result, bool is_first);

    /// Run AKAZE path PnP (MAD pre-filtered data assumed).
    /// Returns {ok, pose}. If is_first && InitialPnP succeeded but GPNP failed,
    /// returns {true, InitialPnP_pose} — do NOT degrade.
    std::pair<bool, PoseEstimate> runAkazePnP(PipelineResult& result, bool is_first);

    /// Run BinaryCorner path PnP.
    /// Returns {ok, pose}. Same InitialPnP fallback semantics as AKAZE.
    std::pair<bool, PoseEstimate> runBinaryCornerPnP(PipelineResult& result, bool is_first);

    /// Run TinyTarget path PnP (cv::solvePnP).
    std::pair<bool, PoseEstimate> runTinyTargetPnP(PipelineResult& result);

    /// Merge a successful pose into final PipelineResult and update tracking state.
    void finalizePose(PipelineResult& result, const PoseEstimate& pose);

    // ========================================================================
    // Logging
    // ========================================================================

    void addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used);
};

} // namespace gpnp
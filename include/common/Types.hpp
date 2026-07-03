#pragma once

/**
 * @file Types.hpp
 * @brief Central data structures for the GPNP stereo tracking system.
 *
 * All inter-module data transfer uses strongly-typed structs defined here.
 * This replaces Python's loosely-typed dict-based data passing.
 */

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <map>
#include <string>
#include <vector>

namespace gpnp {

// ============================================================================
// Camera & Stereo Geometry
// ============================================================================

/// Stereo camera pair parameters (immutable after construction).
struct StereoCameraParams {
    Eigen::Matrix3d K;       ///< 3×3 camera intrinsic matrix
    Eigen::Matrix3d K_inv;   ///< Precomputed inverse of K
    Eigen::Matrix3d R_rl;    ///< Rotation: right camera → left camera
    Eigen::Vector3d t_rl;    ///< Translation: right camera in left frame
    double focal_length;     ///< K(0,0), cached for fast access
    double baseline;         ///< ||t_rl||, cached for fast access
};

// ============================================================================
// Configuration
// ============================================================================

/// Lucas-Kanade optical flow parameters.
struct LKParams {
    cv::Size winSize{21, 21};
    int maxLevel{3};
    cv::TermCriteria criteria{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01};
    double minEigThreshold{1e-4};
};

/// Tracker-level configuration (set once at construction).
struct TrackerConfig {
    double scale{0.5};                   ///< AKAZE image scale factor (0~1)
    int gpnp_min_pts{4};                 ///< Minimum points for GPNP
    bool use_initial_pnp{true};          ///< Enable RANSAC+ITERATIVE initial PnP
    double template_real_width_mm{200.0}; ///< Template physical width (mm)
    double template_real_height_mm{200.0};///< Template physical height (mm)
    LKParams lk_params;                  ///< Optical flow parameters
    int akaze_min_area{40000};           ///< Min ROI area to select AKAZE strategy
    int tiny_max_area{800};              ///< Max ROI area for TinyTarget strategy
};

// ============================================================================
// Feature Data
// ============================================================================

/// Result of AKAZE feature extraction on one image.
struct FeatureSet {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;                        ///< CV_8U, N×61 binary descriptors
    std::vector<cv::Point2f> points;            ///< Keypoint coordinates (N×1×2 → N×2)
    int num_keypoints{0};
};

/// Immutable template data (extracted once at construction).
struct TemplateData {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat color_image;
    cv::Mat gray_image;
    std::vector<Eigen::Vector3d> pts_3d;        ///< 3D coordinates in template frame (mm)
    int template_width{0};
    int template_height{0};

    // --- BinaryCorner / TinyTarget template support ---
    cv::Mat image;                                ///< Raw template PNG image (grayscale)
    cv::Mat image_bool;                           ///< Precomputed binary mask (image > 127)
    std::vector<cv::Point2f> corners;             ///< Corner coordinates from .txt file
    int angle{0};                                 ///< Rotation angle in degrees (from filename)
};

// ============================================================================
// Optical Flow Tracking Result
// ============================================================================

/// Result of LK optical flow L→R with FB check + MAD filtering.
struct TrackResult {
    std::vector<cv::Point2f> pts_left_good;      ///< Valid left feature points
    std::vector<cv::Point2f> pts_right_good;     ///< Valid right feature points
    std::vector<int> idx_from_filtered;          ///< Indices mapping to original kp_left
    std::vector<double> disparity;               ///< = -dx_filtered (stored as negative dx)
    std::vector<double> dx_filtered;             ///< x-disparity after MAD filter
    int num_matched{0};

    // FB check stats (for diagnostics)
    int num_before_fb{0};
    int num_after_fb{0};
    int num_after_mad{0};
    double fb_error_mean{0.0};
};

// ============================================================================
// Stereo Projection Result
// ============================================================================

/// Result of stereo depth estimation and right-image projection.
struct ProjectionResult {
    std::vector<cv::Point2f> pts_right_projected; ///< Projected points on right image
    std::vector<cv::Point2f> pts_left_used;       ///< Left points with valid projection
    std::vector<cv::Point2f> pts_right_used;      ///< Right points with valid projection
    std::vector<bool> valid_mask;                  ///< N-element mask (true = valid projection)
    int num_projected{0};
};

// ============================================================================
// Template Matching Result
// ============================================================================

/// Result of template matching (3-stage: ratio test → cross-check → homography RANSAC).
struct MatchResult {
    std::vector<cv::DMatch> good_matches;        ///< Final filtered DMatch list
    std::vector<cv::Point2f> pts_left_match;     ///< Matched left image points
    std::vector<cv::Point2f> pts_template_match; ///< Matched template points
    int num_matches{0};

    // Filtering statistics (for diagnostics)
    int ratio_test_count{0};
    int cross_check_count{0};
    int homography_count{0};
};

// ============================================================================
// Pose Estimation Result
// ============================================================================

/// Camera pose relative to template: P_cam = R * P_template + t
struct PoseEstimate {
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()}; ///< Rotation: template → camera
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};     ///< Translation (mm)
    bool success{false};
    int num_points{0};

    /// Returns true if the pose represents a valid estimate.
    bool valid() const { return success && num_points > 0; }
};

// ============================================================================
// GPNP Monitoring / Diagnostics
// ============================================================================

/// Detailed monitoring data for GPNP optimization (one per frame).
struct GPNPMonitor {
    // Data flow
    int n_good_matches{0};
    int n_stereo_matched{0};
    int n_query_to_train{0};
    int n_idx_from_filtered{0};
    int n_matched{0};
    int n_intersection{0};
    int n_missing_query{0};
    int n_missing_train{0};
    int n_pts{0};
    int n_rays{0};
    int gpnp_min_pts{0};

    // Optimization
    bool opt_success{false};
    int opt_status{0};
    std::string opt_message;
    int opt_nfev{0};
    double opt_initial_cost{0.0};
    double opt_final_cost{0.0};
    double opt_cost_reduction{0.0};
    double depth_guess{0.0};

    // Result
    std::vector<double> q_opt;   ///< [x, y, z, w] optimized quaternion
    std::vector<double> t_opt;   ///< [tx, ty, tz] optimized translation

    // Error
    std::string failure_reason;
    std::string exception;
    double timing_ms{0.0};

    /// Check if any failure reason was recorded.
    bool failed() const { return !failure_reason.empty() || !exception.empty(); }
};

// ============================================================================
// Logging
// ============================================================================

/// Single-frame log entry (equivalent to Python _add_log dict).
struct LogEntry {
    int frame{0};
    double timestamp{0.0};
    bool is_first{false};
    bool fallback_used{false};

    int n_kp_left{0};
    int n_matched{0};
    int n_projected{0};
    int n_template_match{0};

    bool gpnp_success{false};
    int gpnp_n_pts{0};
    double disparity_median{0.0};

    double total_time_ms{0.0};
    std::map<std::string, double> timing; ///< Per-stage timing (ms)
};

// ============================================================================
// Complete Pipeline Result (single frame)
// ============================================================================

/// Aggregated result from one process() call.
/// Replaces the Python dict returned by process().
struct PipelineResult {
    // --- Feature extraction ---
    std::vector<cv::KeyPoint> kp_left;
    cv::Mat desc_left;
    int n_kp_left{0};

    // --- Optical flow tracking ---
    std::vector<cv::Point2f> pts_left_good;
    std::vector<cv::Point2f> pts_right_good;
    std::vector<double> disparity;
    std::vector<double> dx_filtered;
    std::vector<int> idx_from_filtered;

    // --- Stereo projection ---
    std::vector<cv::Point2f> pts_right_projected;
    std::vector<cv::Point2f> pts_left_used;
    std::vector<cv::Point2f> pts_right_used;

    // --- Template matching ---
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> pts_left_match;
    std::vector<cv::Point2f> pts_template_match;
    int n_template_match{0};

    // --- Pose estimation ---
    Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t{Eigen::Vector3d::Zero()};
    bool gpnp_success{false};
    int gpnp_n_pts{0};

    // --- Timing ---
    std::map<std::string, double> timing; ///< Stage → milliseconds

    // --- Frame metadata ---
    cv::Mat left_color;
    cv::Mat right_color;
    bool is_first_frame{false};
    bool fallback_used{false};
    int n_matched{0};
    int n_projected{0};

    // --- ROI offsets (0626: for mapping ROI coords back to full image) ---
    int left_roi_offset_x{0};
    int left_roi_offset_y{0};
    int right_roi_offset_x{0};
    int right_roi_offset_y{0};

    // --- Intermediate data needed by MAD sync (0626) ---
    std::vector<bool> valid_mask;  ///< valid_mask from projection step

    /// Compute total processing time from all timing entries.
    double total_time_ms() const {
        double total = 0.0;
        for (const auto& [_, t] : timing) total += t;
        return total;
    }
};

// ============================================================================
// ROI Rectangle (0626: left_roi, right_roi parameters)
// ============================================================================

/// Rectangle region of interest: (x, y, width, height).
struct RoiRect {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    /// Returns true if this represents a valid (non-empty) ROI.
    bool valid() const { return width > 0 && height > 0; }

    /// Returns true if offset is zero (full-image mode, no ROI applied).
    bool isZero() const { return x == 0 && y == 0; }
};

// ============================================================================
// MAD Filter Result (0626: returned by MadDisparityFilter::filter())
// ============================================================================

/// Output of MadDisparityFilter::filter(), matching Python dict structure.
struct MadFilterResult {
    std::vector<cv::Point2f> pts_left_filtered;
    std::vector<cv::Point2f> pts_right_filtered;
    std::vector<double> disparity;        ///< = -dx_filtered
    std::vector<double> dx_filtered;
    std::vector<double> dy_filtered;
    std::vector<int> idx_from_filtered;
    std::vector<bool> filter_mask;        ///< N-element boolean mask
    bool downgraded{false};               ///< True if too few points → fallback
};

// ============================================================================
// Mutable Tracking State (Frame-to-Frame Cache)
// ============================================================================

/// Encapsulates all mutable state that persists across frames.
/// Replaces scattered self._has_cache, self._cache, self._frame_count, self._logs.
struct TrackingState {
    bool has_cache{false};
    Eigen::Matrix3d R_prev{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t_prev{0.0, 0.0, 500.0};  ///< Default: 500mm depth
    int frame_count{0};
    std::vector<LogEntry> logs;
    GPNPMonitor last_gpnp_monitor;             ///< Most recent GPNP diagnostics
};

// ============================================================================
// YOLO Detection Types (for ONNX Runtime-based object detection)
// ============================================================================

/// Device backend for ML inference.
enum class DeviceType {
    Auto,   ///< Try CUDA first, fall back to CPU
    CPU,
    CUDA
};

/// Inference status returned by YOLO detector / feature extractors.
enum class Status {
    Success = 0,
    ModelLoadFailed,
    EmptyInput,
    InferenceFailed,
    UnknownError,

    // --- BinaryCorner / TinyTarget status codes ---
    InvalidSize,            ///< Input image has invalid dimensions
    InsufficientFeatures,   ///< Not enough features/corners extracted
    NoSuitableComponent     ///< No suitable connected component found
};

/// Single object detection result.
struct Detection {
    int class_id{0};
    float confidence{0.0f};
    cv::Rect2f bbox;  ///< Bounding box in original image coordinates
};

/// YOLO detector configuration (initialization + ROI generation).
struct YoloConfig {
    std::string model_path;
    DeviceType device{DeviceType::Auto};
    float conf_threshold{0.5f};
    float iou_threshold{0.45f};
    cv::Size input_size{640, 640};
    int intra_op_threads{4};

    // ROI generation parameters
    int target_class_id{0};         ///< Class ID of the target object
    float roi_expand_ratio{0.1f};   ///< Expand ROI by this fraction (0.1 = 10%)
    int roi_min_size{100};          ///< Minimum ROI dimension in pixels
};

} // namespace gpnp

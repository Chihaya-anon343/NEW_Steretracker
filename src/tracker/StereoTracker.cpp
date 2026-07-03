#include "tracker/StereoTracker.hpp"
#include "common/GeometryUtils.hpp"
#include "feature/AkazeGpnpExtractor.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace gpnp {

// ============================================================================
// 构造与初始化 —— 预创建所有三种特征提取器，加载模板
// ============================================================================

StereoTracker::StereoTracker(const Eigen::Matrix3d& K,
                               const Eigen::Matrix3d& R_rl,
                               const Eigen::Vector3d& t_rl,
                               const std::string& template_path,
                               const TrackerConfig& config,
                               const BinaryCornerExtractor::Config& binary_cfg,
                               const std::string& binary_template_dir,
                               const TinyTargetExtractor::Config& tiny_cfg,
                               const std::string& tiny_template_dir)
    : camera_(makeStereoCameraParams(K, R_rl, t_rl))
    , config_(config)
    , initial_pnp_(config.gpnp_min_pts)
    , gpnp_solver_(camera_, config.gpnp_min_pts)
    , mad_filter_(3, 3.0)
    , visualizer_(nullptr)
{
    // ---- ① 预初始化 AKAZE 提取器（主策略，模板加载较慢）----
    akaze_extractor_ = std::make_unique<AkazeGpnpExtractor>(config.scale, config.lk_params);
    akaze_extractor_->initCamera(camera_);
    akaze_extractor_->setTemplateData(template_path,
                                      config.template_real_width_mm,
                                      config.template_real_height_mm);
    template_ = akaze_extractor_->templateData();

    // ---- ② 预初始化 BinaryCorner 提取器（加载 24 个角度模板）----
    binary_extractor_ = std::make_unique<BinaryCornerExtractor>(binary_cfg, binary_template_dir);

    // ---- ③ 预初始化 TinyTarget 提取器（加载多角度模板）----
    tiny_extractor_ = std::make_unique<TinyTargetExtractor>(tiny_cfg, tiny_template_dir);

    // ---- ④ 从 config 读取策略选择阈值 ----
    akaze_min_area_ = config.akaze_min_area;
    tiny_max_area_  = config.tiny_max_area;

    // ---- ④+ 存储 ROI padding 值（用于非 AKAZE 策略时扩大 ROI）----
    binary_roi_pad_ = binary_cfg.roi_pad_pixels;
    tiny_roi_pad_   = tiny_cfg.roi_pad_pixels;

    // ---- ⑤ 默认策略链：AKAZE → BinaryCorner → TinyTarget ----
    extractor_ = akaze_extractor_.get();
    fallback_extractors_.push_back(binary_extractor_.get());
    fallback_extractors_.push_back(tiny_extractor_.get());

    std::cout << "[StereoTracker] Pre-initialized 3 extractors: AkazeGpnp, BinaryCorner, TinyTarget"
              << std::endl;
}

StereoTracker::~StereoTracker() = default;

// ============================================================================
// 策略链配置 —— 根据 ROI 面积选择活跃的主提取器和退化链（仅修改指针，零开销）
// ============================================================================

void StereoTracker::configureStrategyChain(int roi_area) {
    fallback_extractors_.clear();

    if (roi_area >= akaze_min_area_ || roi_area == 0) {
        // Large ROI or no detection → AKAZE → BinaryCorner → TinyTarget
        extractor_ = akaze_extractor_.get();
        fallback_extractors_.push_back(binary_extractor_.get());
        fallback_extractors_.push_back(tiny_extractor_.get());

        std::cout << "[StereoTracker] Strategy chain: AkazeGpnp → BinaryCorner → TinyTarget"
                  << " (roi_area=" << roi_area << ")" << std::endl;

    } else if (roi_area > tiny_max_area_) {
        // Medium ROI → BinaryCorner → TinyTarget
        extractor_ = binary_extractor_.get();
        fallback_extractors_.push_back(tiny_extractor_.get());

        std::cout << "[StereoTracker] Strategy chain: BinaryCorner → TinyTarget"
                  << " (roi_area=" << roi_area << ")" << std::endl;

    } else {
        // Small ROI → TinyTarget only (no further fallback)
        extractor_ = tiny_extractor_.get();
        // fallback_extractors_ remains empty

        std::cout << "[StereoTracker] Strategy chain: TinyTarget only"
                  << " (roi_area=" << roi_area << ")" << std::endl;
    }
}

// ============================================================================
// ROI padding —— 非 AKAZE 策略时扩大 ROI 给角点提取提供周围上下文
// ============================================================================

void StereoTracker::applyRoiPadding(RoiRect& rl, RoiRect& rr, int roi_area,
                                    int left_cols, int left_rows,
                                    int right_cols, int right_rows) const {
    // Only pad when using non-AKAZE strategies (i.e., when roi_area != 0 and < akaze_min_area)
    if (roi_area == 0 || roi_area >= akaze_min_area_)
        return;

    int pad = (roi_area <= tiny_max_area_) ? tiny_roi_pad_
             : (roi_area < akaze_min_area_) ? binary_roi_pad_
             : 0;
    if (pad <= 0) return;

    if (rl.valid()) {
        rl = RoiRect{std::max(0, rl.x - pad),
                     std::max(0, rl.y - pad),
                     std::min(left_cols - std::max(0, rl.x - pad), rl.width  + 2 * pad),
                     std::min(left_rows - std::max(0, rl.y - pad), rl.height + 2 * pad)};
    }
    if (rr.valid()) {
        rr = RoiRect{std::max(0, rr.x - pad),
                     std::max(0, rr.y - pad),
                     std::min(right_cols - std::max(0, rr.x - pad), rr.width  + 2 * pad),
                     std::min(right_rows - std::max(0, rr.y - pad), rr.height + 2 * pad)};
    }
}

// ============================================================================
// 退化辅助函数 —— runExtraction
// ============================================================================

bool StereoTracker::runExtraction(FeatureExtractor& ext,
                                  const cv::Mat& left_gray, const cv::Mat& right_gray,
                                  const cv::Mat& left_color, const cv::Mat& right_color,
                                  const cv::Point2d& left_offset, const cv::Point2d& right_offset,
                                  const cv::Mat& left_color_orig, const cv::Mat& right_color_orig,
                                  PipelineResult& result) {
    result = ext.extract(left_gray, right_gray, left_color, right_color);

    // Restore full-image coordinates
    {
        double lx = left_offset.x, ly = left_offset.y;
        double rx = right_offset.x, ry = right_offset.y;

        if (lx != 0.0 || ly != 0.0) {
            for (auto& kp : result.kp_left) {
                kp.pt.x += static_cast<float>(lx);
                kp.pt.y += static_cast<float>(ly);
            }
            auto offset_left = [lx, ly](std::vector<cv::Point2f>& arr) {
                for (auto& p : arr) { p.x += static_cast<float>(lx); p.y += static_cast<float>(ly); }
            };
            offset_left(result.pts_left_good);
            offset_left(result.pts_left_used);
            offset_left(result.pts_left_match);
            offset_left(result.pts_right_projected);
        }
        if (rx != 0.0 || ry != 0.0) {
            auto offset_right = [rx, ry](std::vector<cv::Point2f>& arr) {
                for (auto& p : arr) { p.x += static_cast<float>(rx); p.y += static_cast<float>(ry); }
            };
            offset_right(result.pts_right_good);
            offset_right(result.pts_right_used);
        }

        result.left_color = left_color_orig;
        result.right_color = right_color_orig;
        result.left_roi_offset_x = static_cast<int>(lx);
        result.left_roi_offset_y = static_cast<int>(ly);
        result.right_roi_offset_x = static_cast<int>(rx);
        result.right_roi_offset_y = static_cast<int>(ry);
    }

    // For non-AKAZE strategies, compute full-image disparity
    bool is_akaze_ext = (ext.name() == "AkazeGpnp");
    if (!is_akaze_ext && !result.pts_left_good.empty() && !result.pts_right_good.empty()) {
        int n = std::min(static_cast<int>(result.pts_left_good.size()),
                         static_cast<int>(result.pts_right_good.size()));
        result.disparity.resize(n);
        result.dx_filtered.resize(n);
        for (int i = 0; i < n; ++i) {
            double d = static_cast<double>(result.pts_left_good[i].x -
                                            result.pts_right_good[i].x);
            result.dx_filtered[i] = d;
            result.disparity[i] = -d;  // = right.x - left.x (match AKAZE convention)
        }
        if (n > 0) {
            std::cout << "  [" << ext.name() << "] Full-image stereo: " << n
                      << " pairs, median_disp=" << computeMedian(result.disparity) << " px"
                      << std::endl;
        }
    }

    // Determine if extraction succeeded
    if (is_akaze_ext) {
        return result.n_kp_left >= 4;  // AKAZE: need minimum keypoints
    } else {
        return !result.pts_left_match.empty();  // BinaryCorner/TinyTarget: need extracted corners
    }
}

// ============================================================================
// PnP Helper: AKAZE path
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runAkazePnP(PipelineResult& result, bool is_first) {
    PoseEstimate pose;
    double gpnp_timing = 0.0;

    // ---- MAD disparity filter ----
    auto t_filter = std::chrono::high_resolution_clock::now();
    MadFilterResult mad_res = mad_filter_.filter(
        result.pts_left_good, result.pts_right_good, result.idx_from_filtered);

    result.pts_left_good = std::move(mad_res.pts_left_filtered);
    result.pts_right_good = std::move(mad_res.pts_right_filtered);
    result.disparity = std::move(mad_res.disparity);
    result.dx_filtered = std::move(mad_res.dx_filtered);
    result.idx_from_filtered = std::move(mad_res.idx_from_filtered);

    auto t_filter_end = std::chrono::high_resolution_clock::now();
    result.timing["filter"] = std::chrono::duration<double, std::milli>(t_filter_end - t_filter).count();

    // ---- Sync pts_left_used/pts_right_used/pts_right_projected after MAD ----
    if (!mad_res.downgraded && !mad_res.filter_mask.empty()) {
        bool all_pass = true;
        for (bool m : mad_res.filter_mask) { if (!m) { all_pass = false; break; } }

        if (!all_pass && !result.valid_mask.empty()) {
            std::vector<int> valid_indices;
            for (size_t i = 0; i < result.valid_mask.size(); ++i)
                if (result.valid_mask[i]) valid_indices.push_back(static_cast<int>(i));

            std::vector<bool> filter_subset(valid_indices.size());
            for (size_t i = 0; i < valid_indices.size(); ++i)
                filter_subset[i] = mad_res.filter_mask[valid_indices[i]];

            auto apply_subset = [&](std::vector<cv::Point2f>& arr) {
                if (arr.size() != filter_subset.size()) return;
                std::vector<cv::Point2f> filtered;
                for (size_t i = 0; i < filter_subset.size(); ++i)
                    if (filter_subset[i]) filtered.push_back(arr[i]);
                arr = std::move(filtered);
            };
            apply_subset(result.pts_left_used);
            apply_subset(result.pts_right_used);
            apply_subset(result.pts_right_projected);
        }
    }

    // ---- AKAZE uses its own template pts_3d if available ----
    const auto& pnp_pts_3d = !extractor_->templateData().pts_3d.empty()
        ? extractor_->templateData().pts_3d
        : template_.pts_3d;

    // ---- Pose estimation ----
    if (is_first && config_.use_initial_pnp) {
        // First frame: try InitialPnP → GPNP
        MatchResult match_res;
        match_res.good_matches = result.good_matches;
        match_res.pts_left_match = result.pts_left_match;
        match_res.pts_template_match = result.pts_template_match;
        PoseEstimate init_pose = initial_pnp_.solve(match_res, pnp_pts_3d, camera_.K);
        result.timing["initial_pnp"] = 0.0;

        if (init_pose.success) {
            // Warm-start GPNP with InitialPnP result
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &init_pose.R, &init_pose.t, gpnp_timing);
            if (!pose.success) {
                // GPNP failed but InitialPnP succeeded → use InitialPnP (no degradation)
                std::cout << "  [AKAZE] GPNP failed, falling back to InitialPnP result" << std::endl;
                pose = init_pose;
                pose.success = true;
            }
        } else {
            // InitialPnP failed → try GPNP with default depth
            std::cout << "  [AKAZE] InitialPnP failed, trying GPNP with default depth" << std::endl;
            Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
            Eigen::Vector3d t_id(0, 0, 5000);
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
        }
    } else if (is_first && !config_.use_initial_pnp) {
        std::cout << "  [InitialPnP] Skipped (use_initial_pnp=false)" << std::endl;
        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_id(0, 0, 5000);
        pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
    } else {
        // Subsequent frame: GPNP with previous frame's pose as warm-start
        const Eigen::Matrix3d* Rp = state_.has_cache ? &state_.R_prev : nullptr;
        const Eigen::Vector3d* tp = state_.has_cache ? &state_.t_prev : nullptr;
        pose = gpnp_solver_.solve(result, pnp_pts_3d, Rp, tp, gpnp_timing);
    }

    return {pose.success, pose};
}

// ============================================================================
// PnP Helper: BinaryCorner path
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runBinaryCornerPnP(PipelineResult& result, bool is_first) {
    PoseEstimate pose;
    double gpnp_timing = 0.0;

    const auto& pnp_pts_3d = !extractor_->templateData().pts_3d.empty()
        ? extractor_->templateData().pts_3d
        : template_.pts_3d;

    if (is_first && config_.use_initial_pnp) {
        // BinaryCorner first frame: try InitialPnP
        MatchResult match_res;
        match_res.good_matches       = result.good_matches;
        match_res.pts_left_match     = result.pts_left_match;
        match_res.pts_template_match = result.pts_template_match;
        PoseEstimate init_pose = initial_pnp_.solve(match_res, pnp_pts_3d, camera_.K);
        result.timing["initial_pnp"] = 0.0;

        if (init_pose.success) {
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &init_pose.R, &init_pose.t, gpnp_timing);
            if (!pose.success) {
                // GPNP failed but InitialPnP succeeded → use InitialPnP (no degradation)
                std::cout << "  [BinaryCorner] GPNP failed, falling back to InitialPnP result" << std::endl;
                pose = init_pose;
                pose.success = true;
            }
        } else {
            // InitialPnP failed → fallback to depth-from-disparity
            std::cout << "  [BinaryCorner] InitialPnP failed, estimating depth from disparity" << std::endl;
            Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
            double depth_from_disp = 500.0;
            if (!result.disparity.empty()) {
                std::vector<double> abs_disp;
                for (double d : result.disparity) abs_disp.push_back(std::abs(d));
                double med_disp = computeMedian(std::move(abs_disp));
                if (med_disp > 1.0) {
                    depth_from_disp = camera_.focal_length * camera_.baseline / med_disp;
                    depth_from_disp = std::clamp(depth_from_disp, 50.0, 5000.0);
                    std::cout << "  [BinaryCorner] Depth from disparity: " << static_cast<int>(depth_from_disp)
                              << "mm (median_disp=" << static_cast<int>(med_disp) << "px)" << std::endl;
                }
            }
            Eigen::Vector3d t_id(0, 0, depth_from_disp);
            pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
        }
    } else if (is_first && !config_.use_initial_pnp) {
        std::cout << "  [InitialPnP] Skipped (use_initial_pnp=false)" << std::endl;
        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
        double depth_from_disp = 500.0;
        if (!result.disparity.empty()) {
            std::vector<double> abs_disp;
            for (double d : result.disparity) abs_disp.push_back(std::abs(d));
            double med_disp = computeMedian(std::move(abs_disp));
            if (med_disp > 1.0) {
                depth_from_disp = camera_.focal_length * camera_.baseline / med_disp;
                depth_from_disp = std::clamp(depth_from_disp, 50.0, 5000.0);
            }
        }
        Eigen::Vector3d t_id(0, 0, depth_from_disp);
        pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
    } else {
        // Subsequent frame
        const Eigen::Matrix3d* Rp = state_.has_cache ? &state_.R_prev : nullptr;
        const Eigen::Vector3d* tp = state_.has_cache ? &state_.t_prev : nullptr;
        pose = gpnp_solver_.solve(result, pnp_pts_3d, Rp, tp, gpnp_timing);
    }

    return {pose.success, pose};
}

// ============================================================================
// PnP Helper: TinyTarget path
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::runTinyTargetPnP(PipelineResult& result) {
    PoseEstimate pose;

    const auto& pnp_pts_3d = !extractor_->templateData().pts_3d.empty()
        ? extractor_->templateData().pts_3d
        : template_.pts_3d;

    if (pnp_pts_3d.empty() || result.pts_left_match.size() < 4)
        return {false, pose};

    std::vector<cv::Point3d> obj_pts;
    for (const auto& p : pnp_pts_3d)
        obj_pts.emplace_back(p.x(), p.y(), p.z());

    std::vector<cv::Point2d> img_pts;
    for (const auto& p : result.pts_left_match)
        img_pts.emplace_back(p.x, p.y);

    cv::Mat K_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
        K_cv.at<double>(r, c) = camera_.K(r, c);

    cv::Mat rvec, tvec;
    bool pnp_ok = cv::solvePnP(obj_pts, img_pts, K_cv, cv::Mat(),
                                rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    if (pnp_ok) {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            pose.R(r, c) = R_cv.at<double>(r, c);
        pose.t = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        pose.num_points = 4;
        pose.success = true;

        std::vector<cv::Point2d> projected;
        cv::projectPoints(obj_pts, rvec, tvec, K_cv, cv::Mat(), projected);
        double rpe_sum = 0.0;
        for (size_t i = 0; i < projected.size(); ++i)
            rpe_sum += cv::norm(projected[i] - img_pts[i]);
        std::cout << "  [TinyTarget] solvePnP OK  n_pts=4  RE="
                  << (rpe_sum / 4.0) << "px" << std::endl;
    } else {
        std::cerr << "  [TinyTarget] solvePnP FAILED" << std::endl;
    }
    result.timing["tiny_pnp"] = 0.0;

    return {pose.success, pose};
}

// ============================================================================
// Finalize pose — merge into PipelineResult and update tracking state
// ============================================================================

void StereoTracker::finalizePose(PipelineResult& result, const PoseEstimate& pose) {
    result.R = pose.R;
    result.t = pose.t;
    result.gpnp_success = pose.success;
    result.gpnp_n_pts = pose.num_points;

    if (pose.success) {
        cv::Mat R_cv(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
            R_cv.at<double>(r, c) = pose.R(r, c);
        cv::Mat rvec;
        cv::Rodrigues(R_cv, rvec);
        std::cout << "  Pose: rvec=[" << rvec.at<double>(0) << ", "
                  << rvec.at<double>(1) << ", " << rvec.at<double>(2) << "]"
                  << "  tvec=[" << pose.t(0) << ", " << pose.t(1) << ", "
                  << pose.t(2) << "] mm  n_pts=" << pose.num_points << std::endl;

        state_.R_prev = pose.R;
        state_.t_prev = pose.t;
        state_.has_cache = true;
    }
}

// ============================================================================
// PnP dispatch helper: route to correct PnP method based on strategy type
// ============================================================================

std::pair<bool, PoseEstimate> StereoTracker::dispatchPnP(FeatureExtractor* ext,
                                                           PipelineResult& result,
                                                           bool is_first) {
    switch (ext->strategyType()) {
        case StrategyType::Akaze:
            return runAkazePnP(result, is_first);
        case StrategyType::BinaryCorner:
            return runBinaryCornerPnP(result, is_first);
        case StrategyType::TinyTarget:
            return runTinyTargetPnP(result);
        default:
            return {false, PoseEstimate{}};
    }
}

// ============================================================================
// 主处理入口 —— 动态退化链条
// ============================================================================

PipelineResult StereoTracker::process(const cv::Mat& left_img,
                                        const cv::Mat& right_img,
                                        bool visualize,
                                        const RoiRect* left_roi,
                                        const RoiRect* right_roi) {
    ++state_.frame_count;

    // ---- Load images ----
    auto [left_color, left_gray] = loadImage(left_img);
    auto [right_color, right_gray] = loadImage(right_img);
    if (left_gray.empty() || right_gray.empty())
        throw std::runtime_error("Cannot read input images");

    // ---- Save originals (for visualization & coordinate restore) ----
    cv::Mat left_color_orig = left_color.clone();
    cv::Mat right_color_orig = right_color.clone();

    // ---- ROI validation & strategy selection ----
    RoiRect roi_l = validateRoi(left_roi, left_gray.size(), "Left ROI");
    RoiRect roi_r = validateRoi(right_roi, right_gray.size(), "Right ROI");

    // Automatically select extraction strategy chain from ROI area
    int roi_area = roi_l.valid() ? roi_l.width * roi_l.height : 0;
    configureStrategyChain(roi_area);

    // Expand ROI with padding for non-AKAZE extractors (corner context)
    applyRoiPadding(roi_l, roi_r, roi_area,
                    left_gray.cols, left_gray.rows,
                    right_gray.cols, right_gray.rows);

    // ---- Cropping (with possibly expanded ROI) ----
    cv::Point2d left_offset(0.0, 0.0), right_offset(0.0, 0.0);
    cv::Mat left_cropped  = left_gray;
    cv::Mat right_cropped = right_gray;
    cv::Mat left_color_cropped  = left_color;
    cv::Mat right_color_cropped = right_color;

    if (roi_l.valid()) {
        left_cropped  = left_gray(cv::Rect(roi_l.x, roi_l.y, roi_l.width, roi_l.height)).clone();
        left_color_cropped = left_color(cv::Rect(roi_l.x, roi_l.y, roi_l.width, roi_l.height)).clone();
        left_offset = cv::Point2d(static_cast<double>(roi_l.x), static_cast<double>(roi_l.y));
    }
    if (roi_r.valid()) {
        right_cropped = right_gray(cv::Rect(roi_r.x, roi_r.y, roi_r.width, roi_r.height)).clone();
        right_color_cropped = right_color(cv::Rect(roi_r.x, roi_r.y, roi_r.width, roi_r.height)).clone();
        right_offset = cv::Point2d(static_cast<double>(roi_r.x), static_cast<double>(roi_r.y));
    }

    bool is_first = !state_.has_cache;
    PipelineResult result;
    std::string winning_strategy;
    bool fallback_used = false;
    bool pose_ok = false;
    PoseEstimate final_pose;

    // ========================================================================
    // Build the degradation chain dynamically
    // ========================================================================

    // Collect all candidates: [primary, fallback[0], fallback[1], ...]
    std::vector<FeatureExtractor*> chain;
    chain.push_back(extractor_);
    for (auto* fb : fallback_extractors_)
        chain.push_back(fb);

    // Remove duplicates: if a fallback's type matches primary, skip it
    StrategyType primary_type = extractor_->strategyType();
    {
        auto it = std::remove_if(chain.begin() + 1, chain.end(),
            [primary_type](FeatureExtractor* e) { return e->strategyType() == primary_type; });
        chain.erase(it, chain.end());
    }

    for (size_t i = 0; i < chain.size(); ++i) {
        FeatureExtractor* ext = chain[i];
        bool is_primary = (i == 0);
        bool is_fb = !is_primary;

        std::string strategy_name = ext->name();

        if (is_fb) {
            fallback_used = true;
            std::string from = chain[i - 1]->name();
            std::cout << "[Degradation] " << from << " failed → " << strategy_name << std::endl;
        }

        bool extract_ok = runExtraction(*ext, left_cropped, right_cropped,
                                        left_color_cropped, right_color_cropped,
                                        left_offset, right_offset,
                                        left_color_orig, right_color_orig, result);

        if (!extract_ok) {
            if (is_primary) {
                std::cout << "[Degradation] " << strategy_name << " extraction failed"
                          << " (n_kp=" << result.n_kp_left << ")" << std::endl;
            } else {
                std::cout << "[Degradation] " << strategy_name << " extraction failed" << std::endl;
            }
            continue;
        }

        auto [ok, pose] = dispatchPnP(ext, result, is_first);
        if (ok) {
            pose_ok = true;
            final_pose = pose;
            winning_strategy = strategy_name;
            break;
        }
    }

    if (!pose_ok) {
        std::cerr << "[Degradation] All strategies failed for frame " << state_.frame_count << std::endl;
    }

    // ---- Finalize pose (if successful) ----
    if (pose_ok) {
        finalizePose(result, final_pose);
    }

    // ---- Visualization ----
    if (visualize && pose_ok) {
        std::string prefix = "_f" + std::to_string(state_.frame_count);

        if (winning_strategy == "AkazeGpnp") {
            if (!visualizer_) visualizer_ = std::make_unique<Visualizer>(camera_.K, output_dir_);
            cv::Mat tmpl_color = template_.color_image;
            if (tmpl_color.empty()) {
                cv::cvtColor(template_.gray_image.empty()
                    ? cv::Mat(200, 200, CV_8UC1, cv::Scalar(128))
                    : template_.gray_image,
                    tmpl_color, cv::COLOR_GRAY2BGR);
            }
            visualizer_->generateAll(
                left_color_orig, right_color_orig, tmpl_color,
                result.kp_left,
                result.pts_left_used, result.pts_right_used,
                result.pts_right_projected,
                result.good_matches,
                result.pts_left_match, result.pts_template_match,
                result.disparity, prefix,
                result.R, result.t, result.gpnp_success);
        } else {
            // BinaryCorner / TinyTarget visualization
            auto expandRect = [](const RoiRect& roi, const cv::Size& imgSz) -> cv::Rect {
                int cx = roi.x + roi.width / 2;
                int cy = roi.y + roi.height / 2;
                int ew = roi.width * 5;
                int eh = roi.height * 5;
                int x = std::max(0, cx - ew / 2);
                int y = std::max(0, cy - eh / 2);
                int w = std::min(ew, imgSz.width - x);
                int h = std::min(eh, imgSz.height - y);
                return cv::Rect(x, y, w, h);
            };
            cv::Rect expand_L = expandRect(roi_l, left_color_orig.size());
            cv::Rect expand_R = expandRect(roi_r, right_color_orig.size());

            cv::Mat view_L = left_color_orig(expand_L).clone();
            cv::Mat view_R = right_color_orig(expand_R).clone();

            float elx = static_cast<float>(expand_L.x), ely = static_cast<float>(expand_L.y);
            float erx = static_cast<float>(expand_R.x), ery = static_cast<float>(expand_R.y);
            auto toView_L = [&](const cv::Point2f& p) { return cv::Point2f(p.x - elx, p.y - ely); };
            auto toView_R = [&](const cv::Point2f& p) { return cv::Point2f(p.x - erx, p.y - ery); };

            auto proj = [&](const Eigen::Vector3d& P) -> cv::Point {
                if (std::abs(P.z()) < 1e-6) return cv::Point(-1, -1);
                Eigen::Vector2d uv = projectToImage(P, camera_.K);
                return cv::Point(static_cast<int>(uv.x()), static_cast<int>(uv.y()));
            };

            const cv::Scalar CORNER_COLORS[] = {
                {0,0,255}, {0,255,0}, {0,255,255}, {255,0,0}, {255,0,255},
                {255,255,0}, {128,0,255}, {0,128,255}, {255,128,0}, {128,255,0}
            };

            const auto& pnp_pts_3d = template_.pts_3d;
            bool is_tiny = (winning_strategy == "TinyTarget");

            // --- Panel 0: 二值图像（左 | 右），角点叠加 (BinaryCorner only) ---
            if (!is_tiny) {
                // Use pre-initialized extractor for debug state
                auto* bce = binary_extractor_.get();
                if (bce) {
                    cv::Mat bl_bgr, br_bgr;
                    cv::cvtColor(bce->lastLeftBinary(),  bl_bgr, cv::COLOR_GRAY2BGR);
                    cv::cvtColor(bce->lastRightBinary(), br_bgr, cv::COLOR_GRAY2BGR);
                    float lx_roi = static_cast<float>(left_offset.x);
                    float ly_roi = static_cast<float>(left_offset.y);
                    float rx_roi = static_cast<float>(right_offset.x);
                    float ry_roi = static_cast<float>(right_offset.y);
                    for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                        cv::Point p(static_cast<int>(result.pts_left_match[i].x - lx_roi),
                                     static_cast<int>(result.pts_left_match[i].y - ly_roi));
                        cv::circle(bl_bgr, p, 1, CORNER_COLORS[i % 10], -1);
                    }
                    for (size_t i = 0; i < result.pts_right_good.size(); ++i) {
                        cv::Point p(static_cast<int>(result.pts_right_good[i].x - rx_roi),
                                     static_cast<int>(result.pts_right_good[i].y - ry_roi));
                        cv::circle(br_bgr, p, 1, CORNER_COLORS[i % 10], -1);
                    }
                    cv::Mat p0;
                    cv::hconcat(bl_bgr, br_bgr, p0);
                    cv::imwrite(output_dir_ + "/binary_corner_binary" + prefix + ".png", p0);
                }
            }

            // --- Panel 0b: 旋转回正二值图 (BinaryCorner only) ---
            if (!is_tiny) {
                auto* bce = binary_extractor_.get();
                if (bce && !bce->lastUprightBinary().empty()) {
                    cv::Mat up;
                    cv::cvtColor(bce->lastUprightBinary(), up, cv::COLOR_GRAY2BGR);
                    cv::imwrite(output_dir_ + "/binary_corner_upright" + prefix + ".png", up);
                }
            }

            // --- Panel 1: 3D axes on expanded left view ---
            {
                cv::Mat p1 = view_L.clone();
                double axis_len = 100.0;
                Eigen::Vector3d o  = final_pose.R * Eigen::Vector3d(0,0,0) + final_pose.t;
                Eigen::Vector3d ax = final_pose.R * Eigen::Vector3d(axis_len,0,0) + final_pose.t;
                Eigen::Vector3d ay = final_pose.R * Eigen::Vector3d(0,axis_len,0) + final_pose.t;
                Eigen::Vector3d az = final_pose.R * Eigen::Vector3d(0,0,axis_len) + final_pose.t;
                cv::Point o_p = proj(o);  o_p.x -= expand_L.x; o_p.y -= expand_L.y;
                cv::Point ax_p = proj(ax); ax_p.x -= expand_L.x; ax_p.y -= expand_L.y;
                cv::Point ay_p = proj(ay); ay_p.x -= expand_L.x; ay_p.y -= expand_L.y;
                cv::Point az_p = proj(az); az_p.x -= expand_L.x; az_p.y -= expand_L.y;
                cv::line(p1, o_p, ax_p, cv::Scalar(0,0,255), 2, cv::LINE_AA);
                cv::line(p1, o_p, ay_p, cv::Scalar(0,255,0), 2, cv::LINE_AA);
                cv::line(p1, o_p, az_p, cv::Scalar(255,0,0), 2, cv::LINE_AA);
                cv::imwrite(output_dir_ + "/binary_corner_axes" + prefix + ".png", p1);
            }

            // --- Panel 2: template corner correspondence (BinaryCorner only) ---
            if (!is_tiny) {
                cv::Mat p2_l = view_L.clone();
                for (size_t i = 0; i < result.pts_left_match.size(); ++i) {
                    cv::Point2f pv = toView_L(result.pts_left_match[i]);
                    cv::circle(p2_l, cv::Point(static_cast<int>(pv.x), static_cast<int>(pv.y)),
                               1, CORNER_COLORS[i % 10], -1);
                }
                auto* bce = binary_extractor_.get();
                const TemplateData* matched_tmpl = bce ? bce->lastMatchedTemplate() : nullptr;
                cv::Mat p2_tmpl;
                if (matched_tmpl) {
                    cv::cvtColor(matched_tmpl->image, p2_tmpl, cv::COLOR_GRAY2BGR);
                } else {
                    p2_tmpl = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128,128,128));
                }
                cv::resize(p2_tmpl, p2_tmpl, p2_l.size(), 0, 0, cv::INTER_NEAREST);
                if (matched_tmpl) {
                    double dsx = static_cast<double>(p2_tmpl.cols) / matched_tmpl->image.cols;
                    double dsy = static_cast<double>(p2_tmpl.rows) / matched_tmpl->image.rows;
                    for (size_t i = 0; i < matched_tmpl->corners.size(); ++i) {
                        cv::Point p(static_cast<int>(matched_tmpl->corners[i].x * dsx),
                                    static_cast<int>(matched_tmpl->corners[i].y * dsy));
                        cv::circle(p2_tmpl, p, 1, CORNER_COLORS[i % 10], -1);
                    }
                }
                cv::Mat p2;
                cv::hconcat(p2_l, p2_tmpl, p2);
                cv::imwrite(output_dir_ + "/binary_corner_template" + prefix + ".png", p2);
            }

            // --- Panel 3: Left-right expanded view corner correspondence ---
            {
                cv::Mat p3_l = view_L.clone(), p3_r = view_R.clone();
                for (size_t i = 0; i < result.pts_left_match.size() &&
                                i < result.pts_right_good.size(); ++i) {
                    cv::Point2f pvl = toView_L(result.pts_left_match[i]);
                    cv::Point2f pvr = toView_R(result.pts_right_good[i]);
                    cv::circle(p3_l, cv::Point(static_cast<int>(pvl.x), static_cast<int>(pvl.y)),
                               1, CORNER_COLORS[i % 10], -1);
                    cv::circle(p3_r, cv::Point(static_cast<int>(pvr.x), static_cast<int>(pvr.y)),
                               1, CORNER_COLORS[i % 10], -1);
                }
                cv::Mat p3;
                cv::hconcat(p3_l, p3_r, p3);
                cv::imwrite(output_dir_ + "/binary_corner_stereo" + prefix + ".png", p3);
            }

            // --- Panel 4: Reprojection on expanded left view ---
            if (!pnp_pts_3d.empty()) {
                cv::Mat p4 = view_L.clone();

                int max_qidx = 0;
                for (const auto& m : result.good_matches)
                    max_qidx = std::max(max_qidx, m.queryIdx);
                std::vector<int> train_lut(max_qidx + 1, -1);
                for (const auto& m : result.good_matches)
                    train_lut[m.queryIdx] = m.trainIdx;

                std::vector<int> obs_to_3d(result.pts_left_good.size(), -1);
                std::vector<int> used_3d_indices;
                for (size_t j = 0; j < result.pts_left_good.size() &&
                                    j < result.idx_from_filtered.size(); ++j) {
                    int kp_idx = result.idx_from_filtered[j];
                    if (kp_idx >= 0 && kp_idx < static_cast<int>(train_lut.size())) {
                        int tr_idx = train_lut[kp_idx];
                        if (tr_idx >= 0 && tr_idx < static_cast<int>(pnp_pts_3d.size())) {
                            obs_to_3d[j] = tr_idx;
                            used_3d_indices.push_back(tr_idx);
                        }
                    }
                }

                std::vector<cv::Point3d> obj_pts;
                obj_pts.reserve(used_3d_indices.size());
                for (int idx : used_3d_indices)
                    obj_pts.emplace_back(pnp_pts_3d[idx].x(), pnp_pts_3d[idx].y(), pnp_pts_3d[idx].z());

                std::vector<cv::Point2d> projected;
                if (!obj_pts.empty()) {
                    cv::Mat rvec4, tvec4(3, 1, CV_64F);
                    tvec4.at<double>(0) = final_pose.t(0);
                    tvec4.at<double>(1) = final_pose.t(1);
                    tvec4.at<double>(2) = final_pose.t(2);
                    cv::Mat R4(3, 3, CV_64F);
                    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                        R4.at<double>(r, c) = final_pose.R(r, c);
                    cv::Rodrigues(R4, rvec4);
                    cv::Mat K4(3, 3, CV_64F);
                    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                        K4.at<double>(r, c) = camera_.K(r, c);
                    cv::projectPoints(obj_pts, rvec4, tvec4, K4, cv::Mat(), projected);
                }

                std::unordered_map<int, size_t> proj_map;
                for (size_t k = 0; k < used_3d_indices.size(); ++k)
                    proj_map[used_3d_indices[k]] = k;

                for (size_t j = 0; j < result.pts_left_good.size(); ++j) {
                    int tr_idx = obs_to_3d[j];
                    if (tr_idx < 0) continue;
                    auto it = proj_map.find(tr_idx);
                    if (it == proj_map.end() || it->second >= projected.size()) continue;
                    cv::Point2f obs_v = toView_L(result.pts_left_good[j]);
                    cv::Point pd(static_cast<int>(projected[it->second].x - expand_L.x),
                                 static_cast<int>(projected[it->second].y - expand_L.y));
                    cv::Point pm(static_cast<int>(obs_v.x), static_cast<int>(obs_v.y));
                    cv::circle(p4, pd, 1, cv::Scalar(0, 255, 0), -1);
                    cv::circle(p4, pm, 1, cv::Scalar(0, 0, 255), 1);
                    cv::line(p4, pd, pm, cv::Scalar(0, 255, 255), 1);
                }
                cv::imwrite(output_dir_ + "/binary_corner_reproj" + prefix + ".png", p4);
            }

            // --- Panel 5: Stereo projection (disparity-based) ---
            if (!result.pts_left_good.empty() && !result.pts_right_good.empty()) {
                cv::Mat p5 = view_L.clone();
                const double fx = camera_.focal_length;
                const double fy = camera_.K(1,1);
                const double cx = camera_.K(0,2);
                const double cy = camera_.K(1,2);
                const double b  = camera_.baseline;
                const Eigen::Matrix3d& R_rl = camera_.R_rl;
                const Eigen::Vector3d& t_rl = camera_.t_rl;

                int n_stereo_proj = std::min(static_cast<int>(result.pts_left_good.size()),
                                              static_cast<int>(result.pts_right_good.size()));

                for (int i = 0; i < n_stereo_proj; ++i) {
                    double uL = result.pts_left_good[i].x;
                    double vL = result.pts_left_good[i].y;
                    double uR = result.pts_right_good[i].x;
                    double vR = result.pts_right_good[i].y;

                    double disp = uL - uR;
                    double abs_disp = std::abs(disp);
                    if (abs_disp < 0.5) continue;

                    double depth = fx * b / abs_disp;
                    if (depth <= 0.0 || depth > 100000.0) continue;

                    double rx = (uL - cx) / fx;
                    double ry = (vL - cy) / fy;
                    double rn = std::sqrt(rx*rx + ry*ry + 1.0);
                    rx /= rn; ry /= rn;
                    double rz = 1.0 / rn;

                    double Px = rx * depth;
                    double Py = ry * depth;
                    double Pz = rz * depth;

                    double dx = Px - t_rl(0);
                    double dy = Py - t_rl(1);
                    double dz = Pz - t_rl(2);
                    double PRx = dx*R_rl(0,0) + dy*R_rl(1,0) + dz*R_rl(2,0);
                    double PRy = dx*R_rl(0,1) + dy*R_rl(1,1) + dz*R_rl(2,1);
                    double PRz = dx*R_rl(0,2) + dy*R_rl(1,2) + dz*R_rl(2,2);

                    if (std::abs(PRz) < 1e-6) continue;

                    double proj_x = fx * PRx / PRz + cx;
                    double proj_y = fy * PRy / PRz + cy;

                    cv::Point2f obs_v = toView_L(result.pts_left_good[i]);
                    cv::Point pt_L(static_cast<int>(obs_v.x), static_cast<int>(obs_v.y));
                    cv::Point pt_R(static_cast<int>(proj_x - expand_L.x),
                                   static_cast<int>(proj_y - expand_L.y));

                    cv::circle(p5, pt_L, 4, cv::Scalar(255, 0, 0), -1);
                    cv::drawMarker(p5, pt_R, cv::Scalar(0, 0, 255),
                                   cv::MARKER_TILTED_CROSS, 10, 2);
                    cv::line(p5, pt_L, pt_R, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                }
                cv::imwrite(output_dir_ + "/binary_corner_stereo_proj" + prefix + ".png", p5);
            }
        }
    }

    // ---- Logging ----
    result.is_first_frame = is_first;
    result.n_matched = static_cast<int>(result.pts_left_good.size());
    result.n_projected = static_cast<int>(result.pts_right_projected.size());
    addLogEntry(result, is_first, fallback_used);
    return result;
}

PipelineResult StereoTracker::process(const std::string& left_path,
                                        const std::string& right_path,
                                        bool visualize,
                                        const RoiRect* left_roi,
                                        const RoiRect* right_roi) {
    cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
    return process(left, right, visualize, left_roi, right_roi);
}

void StereoTracker::clearCache() { state_ = TrackingState{}; }

// ============================================================================
// ROI 辅助函数 —— 校验、裁剪、坐标偏移
// ============================================================================

RoiRect StereoTracker::validateRoi(const RoiRect* roi, const cv::Size& img_size,
                                     const std::string& name) {
    if (roi == nullptr || !roi->valid()) return RoiRect{};
    int x = roi->x, y = roi->y, w = roi->width, h = roi->height;
    if (x < 0 || y < 0 || w <= 0 || h <= 0)
        throw std::invalid_argument(name + " invalid: x=" + std::to_string(x) +
            ",y=" + std::to_string(y) + ",w=" + std::to_string(w) + ",h=" + std::to_string(h));
    if (x + w > img_size.width || y + h > img_size.height)
        throw std::invalid_argument(name + " out of bounds (" +
            std::to_string(img_size.width) + "x" + std::to_string(img_size.height) + "): " +
            std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
    return RoiRect{x, y, w, h};
}

void StereoTracker::offsetResultToOriginal(PipelineResult& result,
                                             const cv::Point2d& left_offset,
                                             const cv::Point2d& right_offset,
                                             const cv::Mat& left_color_orig,
                                             const cv::Mat& right_color_orig) {
    if (left_offset.x == 0.0 && left_offset.y == 0.0 &&
        right_offset.x == 0.0 && right_offset.y == 0.0)
        return;

    double lx = left_offset.x, ly = left_offset.y;
    double rx = right_offset.x, ry = right_offset.y;

    // Offset keypoints
    for (auto& kp : result.kp_left) {
        kp.pt.x += static_cast<float>(lx);
        kp.pt.y += static_cast<float>(ly);
    }

    // Offset left-image coordinate arrays
    auto offset_left = [lx, ly](std::vector<cv::Point2f>& arr) {
        for (auto& p : arr) { p.x += static_cast<float>(lx); p.y += static_cast<float>(ly); }
    };
    offset_left(result.pts_left_good);
    offset_left(result.pts_left_used);
    offset_left(result.pts_left_match);
    offset_left(result.pts_right_projected);

    // Offset right-image coordinate arrays
    auto offset_right = [rx, ry](std::vector<cv::Point2f>& arr) {
        for (auto& p : arr) { p.x += static_cast<float>(rx); p.y += static_cast<float>(ry); }
    };
    offset_right(result.pts_right_good);
    offset_right(result.pts_right_used);

    // Restore original full-size color images
    result.left_color = left_color_orig;
    result.right_color = right_color_orig;
    result.left_roi_offset_x = static_cast<int>(lx);
    result.left_roi_offset_y = static_cast<int>(ly);
    result.right_roi_offset_x = static_cast<int>(rx);
    result.right_roi_offset_y = static_cast<int>(ry);
}

// ============================================================================
// 图像加载 —— 分离彩色图和灰度图
// ============================================================================

std::pair<cv::Mat, cv::Mat> StereoTracker::loadImage(const cv::Mat& img) {
    if (img.empty()) return {cv::Mat(), cv::Mat()};
    cv::Mat color, gray;
    if (img.channels() == 3) {
        color = img.clone();
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 1) {
        cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);
        gray = img.clone();
    } else {
        throw std::runtime_error("Unsupported image: " + std::to_string(img.channels()) + " channels");
    }
    return {color, gray};
}

// ============================================================================
// 日志记录 —— 保存每帧的处理数据，支持表格化输出
// ============================================================================

const std::vector<LogEntry>& StereoTracker::getLogs() const { return state_.logs; }

void StereoTracker::addLogEntry(const PipelineResult& result, bool is_first, bool fallback_used) {
    double total_time = result.total_time_ms();
    double disp_median = 0.0;
    if (!result.disparity.empty()) {
        std::vector<double> abs_disp;
        for (double d : result.disparity) abs_disp.push_back(std::abs(d));
        disp_median = computeMedian(std::move(abs_disp));
    }
    LogEntry entry;
    entry.frame = state_.frame_count;
    entry.timestamp = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count()) / 1e9;
    entry.is_first = is_first;
    entry.fallback_used = fallback_used;
    entry.n_kp_left = result.n_kp_left;
    entry.n_matched = static_cast<int>(result.pts_left_good.size());
    entry.n_projected = static_cast<int>(result.pts_right_projected.size());
    entry.n_template_match = result.n_template_match;
    entry.gpnp_success = result.gpnp_success;
    entry.gpnp_n_pts = result.gpnp_n_pts;
    entry.disparity_median = disp_median;
    entry.total_time_ms = total_time;
    entry.timing = result.timing;
    state_.logs.push_back(std::move(entry));
}

void StereoTracker::printLogs() const {
    const auto& logs = state_.logs;
    if (logs.empty()) { std::cout << "[Log is empty]" << std::endl; return; }

    const std::vector<std::string> timing_keys = {"akaze", "flow", "filter", "proj", "match_template", "gpnp"};
    const std::map<std::string, std::string> timing_labels = {
        {"akaze", "AKAZE"}, {"flow", "OptFlow"}, {"filter", "Filter"},
        {"proj", "Proj"}, {"match_template", "Match"}, {"gpnp", "PnP"}};

    std::vector<std::string> used_keys;
    for (const auto& k : timing_keys)
        for (const auto& log : logs)
            if (auto it = log.timing.find(k); it != log.timing.end() && it->second > 0.0)
                { used_keys.push_back(k); break; }

    std::vector<size_t> widths = {5, 12, 8, 8, 8, 10, 8, 10, 10};
    for (const auto& k : used_keys) widths.push_back(10);

    auto print_sep = [&](char c) {
        std::cout << "+";
        for (auto w : widths) std::cout << std::string(w, c) << "+";
        std::cout << std::endl;
    };
    auto print_row = [&](const std::vector<std::string>& cols) {
        std::cout << "|";
        for (size_t i = 0; i < cols.size() && i < widths.size(); ++i)
            std::cout << std::setw(static_cast<int>(widths[i])) << cols[i] << "|";
        std::cout << std::endl;
    };

    print_sep('-');
    std::vector<std::string> header = {"Fr#", "Timestamp", "1st", "FB", "nKp", "nMatch",
                                        "nProj", "nTmpl", "Disp(px)", "PnP"};
    for (const auto& k : used_keys) {
        auto it = timing_labels.find(k);
        header.push_back(it != timing_labels.end() ? it->second : k);
    }
    print_row(header);
    print_sep('-');

    for (const auto& log : logs) {
        std::vector<std::string> row;
        row.push_back(std::to_string(log.frame));
        std::ostringstream ts; ts << std::fixed << std::setprecision(3) << log.timestamp;
        row.push_back(ts.str());
        row.push_back(log.is_first ? "Y" : "N");
        row.push_back(log.fallback_used ? "Y" : "N");
        row.push_back(std::to_string(log.n_kp_left));
        row.push_back(std::to_string(log.n_matched));
        row.push_back(std::to_string(log.n_projected));
        row.push_back(std::to_string(log.n_template_match));
        std::ostringstream ds; ds << std::fixed << std::setprecision(1) << log.disparity_median;
        row.push_back(ds.str());
        row.push_back(log.gpnp_success ? "OK" : "FAIL");
        for (const auto& k : used_keys) {
            auto it = log.timing.find(k);
            std::ostringstream ts2;
            ts2 << std::fixed << std::setprecision(1)
                << (it != log.timing.end() ? it->second : 0.0);
            row.push_back(ts2.str());
        }
        print_row(row);
    }
    print_sep('-');
}

} // namespace gpnp
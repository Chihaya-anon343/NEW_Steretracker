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

namespace gpnp {

// ============================================================================
// 构造与初始化 —— 创建默认 AKAZE 提取器，加载模板特征
// ============================================================================

StereoTracker::StereoTracker(const Eigen::Matrix3d& K,
                               const Eigen::Matrix3d& R_rl,
                               const Eigen::Vector3d& t_rl,
                               const std::string& template_path,
                               const TrackerConfig& config)
    : camera_(makeStereoCameraParams(K, R_rl, t_rl))
    , config_(config)
    , initial_pnp_(config.gpnp_min_pts)
    , gpnp_solver_(camera_, config.gpnp_min_pts)
    , mad_filter_(3, 3.0)   // 0626: min_pts=3, mad_factor=3.0
    , visualizer_(nullptr)
{
    // Create default AKAZE extractor and init camera params
    auto akaze = std::make_unique<AkazeGpnpExtractor>(config.scale, config.lk_params);
    akaze->initCamera(camera_);

    // Load template via the extractor
    akaze->setTemplateData(template_path,
                           config.template_real_width_mm,
                           config.template_real_height_mm);
    template_ = akaze->templateData();

    extractor_ = std::move(akaze);
}

StereoTracker::~StereoTracker() = default;

// ============================================================================
// 主处理入口 —— 双目帧 → 图像裁剪 → 特征提取 → 视差 → PnP → 可视化
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

    // ---- ROI validation & cropping (0626) ----
    RoiRect roi_l = validateRoi(left_roi, left_gray.size(), "Left ROI");
    RoiRect roi_r = validateRoi(right_roi, right_gray.size(), "Right ROI");

    cv::Point2d left_offset(0.0, 0.0), right_offset(0.0, 0.0);
    if (roi_l.valid()) {
        left_gray = left_gray(cv::Rect(roi_l.x, roi_l.y, roi_l.width, roi_l.height)).clone();
        left_color = left_color(cv::Rect(roi_l.x, roi_l.y, roi_l.width, roi_l.height)).clone();
        left_offset = cv::Point2d(static_cast<double>(roi_l.x), static_cast<double>(roi_l.y));
    }
    if (roi_r.valid()) {
        right_gray = right_gray(cv::Rect(roi_r.x, roi_r.y, roi_r.width, roi_r.height)).clone();
        right_color = right_color(cv::Rect(roi_r.x, roi_r.y, roi_r.width, roi_r.height)).clone();
        right_offset = cv::Point2d(static_cast<double>(roi_r.x), static_cast<double>(roi_r.y));
    }

    // ---- Pipeline routing ----
    bool is_first = !state_.has_cache;
    PipelineResult result = extractor_->extract(left_gray, right_gray, left_color, right_color);

    // ---- ROI坐标还原：局部 → 全图 ----
    offsetResultToOriginal(result, left_offset, right_offset,
                           left_color_orig, right_color_orig);

    // ---- MAD disparity filter (only for AKAZE; skip for BinaryCorner/TinyTarget) ----
    bool is_akaze = (extractor_->name() == "AkazeGpnp");

    // ---- 非AKAZE策略：在全图坐标下计算视差（extract()中是ROI局部坐标）----
    if (!is_akaze && !result.pts_left_good.empty() && !result.pts_right_good.empty()) {
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
            std::cout << "  [BinaryCorner] Full-image stereo: " << n
                      << " pairs, median_disp=" << computeMedian(result.disparity) << " px"
                      << std::endl;
        }
    }

    // ---- 选取 PnP 物点：优先用提取器自己的 pts_3d，否则用 AKAZE 模板 ----
    const auto& pnp_pts_3d = !extractor_->templateData().pts_3d.empty()
        ? extractor_->templateData().pts_3d
        : template_.pts_3d;
    if (is_akaze) {
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
    }

    // ---- Pose Estimation ----
    PoseEstimate pose;
    double gpnp_timing = 0.0;
    bool is_tiny = (extractor_->name() == "TinyTarget");

    if (is_tiny) {
        // ---- TinyTarget: cv::solvePnP (4 corners ↔ 4 known square vertices) ----
        if (!pnp_pts_3d.empty() && result.pts_left_match.size() >= 4) {
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

                // Reprojection error
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
        }
        result.timing["tiny_pnp"] = 0.0;

    } else if (is_first && config_.use_initial_pnp && is_akaze) {
        // InitialPnP requires descriptor-based matches (AKAZE only)
        MatchResult match_res;
        match_res.good_matches = result.good_matches;
        match_res.pts_left_match = result.pts_left_match;
        match_res.pts_template_match = result.pts_template_match;
        PoseEstimate init_pose = initial_pnp_.solve(match_res, pnp_pts_3d, camera_.K);
        result.timing["initial_pnp"] = 0.0;

        const Eigen::Matrix3d& R_warm = init_pose.success ? init_pose.R : Eigen::Matrix3d::Identity();
        const Eigen::Vector3d& t_warm = init_pose.success ? init_pose.t : Eigen::Vector3d(0, 0, 500);
        pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_warm, &t_warm, gpnp_timing);
    } else if (is_first && !config_.use_initial_pnp) {
        std::cout << "  [InitialPnP] Skipped (use_initial_pnp=false)" << std::endl;
        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_id(0, 0, 5000);
        pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
    } else if (is_first && !is_akaze) {
        // Non-AKAZE first frame (BinaryCorner): skip InitialPnP, use identity warm-start
        std::cout << "  [InitialPnP] Skipped (non-AKAZE strategy: " << extractor_->name() << ")" << std::endl;
        Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_id(0, 0, 5000);
        pose = gpnp_solver_.solve(result, pnp_pts_3d, &R_id, &t_id, gpnp_timing);
    } else {
        const Eigen::Matrix3d* Rp = state_.has_cache ? &state_.R_prev : nullptr;
        const Eigen::Vector3d* tp = state_.has_cache ? &state_.t_prev : nullptr;
        pose = gpnp_solver_.solve(result, pnp_pts_3d, Rp, tp, gpnp_timing);
    }

    result.R = pose.R; result.t = pose.t;
    result.gpnp_success = pose.success; result.gpnp_n_pts = pose.num_points;

    // ---- 位姿结果输出 ----
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
    }

    // ---- 可视化（根据策略不同，生成不同的调试图像）----
    if (visualize) {
        std::string prefix = "_f" + std::to_string(state_.frame_count);

        if (is_akaze) {
            // ---- AKAZE visualization (unchanged) ----
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

        } else if (!is_akaze && pose.success) {
            // ---- BinaryCorner visualization: 4 panels ----
            // Crop from full-size original images, centred on ROI, expanded 2×.
            // All point coords are full-image. Subtract expand rect top-left to plot.

            // --- Compute expanded view rects (left & right) ---
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

            // Crop expanded view
            cv::Mat view_L = left_color_orig(expand_L).clone();
            cv::Mat view_R = right_color_orig(expand_R).clone();

            // Coordinate shift: full-image → expanded-view local
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

            // --- Panel 0: 二值图像（左 | 右），角点叠加 ---
            {
                auto* bce = dynamic_cast<BinaryCornerExtractor*>(extractor_.get());
                if (bce) {
                    cv::Mat bl_bgr, br_bgr;
                    cv::cvtColor(bce->lastLeftBinary(),  bl_bgr, cv::COLOR_GRAY2BGR);
                    cv::cvtColor(bce->lastRightBinary(), br_bgr, cv::COLOR_GRAY2BGR);
                    // 角点坐标从全图转回二值图坐标系（= ROI 局部坐标）
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

            // --- Panel 0b: 旋转回正二值图（仅 BinaryCorner）---
            {
                auto* bce = dynamic_cast<BinaryCornerExtractor*>(extractor_.get());
                if (bce && !bce->lastUprightBinary().empty()) {
                    cv::Mat up;
                    cv::cvtColor(bce->lastUprightBinary(), up, cv::COLOR_GRAY2BGR);
                    cv::imwrite(output_dir_ + "/binary_corner_upright" + prefix + ".png", up);
                }
            }

            // --- Panel 1: 3D axes on expanded left view ---
            {
                cv::Mat p1 = view_L.clone();
                double axis_len = 100.0; // mm — adjust to your target size
                Eigen::Vector3d o  = pose.R * Eigen::Vector3d(0,0,0) + pose.t;
                Eigen::Vector3d ax = pose.R * Eigen::Vector3d(axis_len,0,0) + pose.t;
                Eigen::Vector3d ay = pose.R * Eigen::Vector3d(0,axis_len,0) + pose.t;
                Eigen::Vector3d az = pose.R * Eigen::Vector3d(0,0,axis_len) + pose.t;
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
                auto* bce = dynamic_cast<BinaryCornerExtractor*>(extractor_.get());
                const TemplateData* matched_tmpl = bce ? bce->lastMatchedTemplate() : nullptr;
                cv::Mat p2_tmpl;
                if (matched_tmpl) {
                    cv::cvtColor(matched_tmpl->image, p2_tmpl, cv::COLOR_GRAY2BGR);
                } else {
                    p2_tmpl = cv::Mat(100, 100, CV_8UC3, cv::Scalar(128,128,128));
                }
                cv::resize(p2_tmpl, p2_tmpl, p2_l.size(), 0, 0, cv::INTER_NEAREST);
                // 模板角点按显示尺寸重新缩放（原 pts_template_match 是ROI尺寸的，不匹配）
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
            if (pose.success && !pnp_pts_3d.empty()) {
                cv::Mat p4 = view_L.clone();
                std::vector<cv::Point3d> obj_pts;
                for (const auto& p3d : pnp_pts_3d)
                    obj_pts.emplace_back(p3d.x(), p3d.y(), p3d.z());
                std::vector<cv::Point2d> projected;
                cv::Mat rvec4, tvec4(3, 1, CV_64F);
                tvec4.at<double>(0) = pose.t(0); tvec4.at<double>(1) = pose.t(1);
                tvec4.at<double>(2) = pose.t(2);
                cv::Mat R4(3, 3, CV_64F);
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                    R4.at<double>(r, c) = pose.R(r, c);
                cv::Rodrigues(R4, rvec4);
                cv::Mat K4(3, 3, CV_64F);
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                    K4.at<double>(r, c) = camera_.K(r, c);
                cv::projectPoints(obj_pts, rvec4, tvec4, K4, cv::Mat(), projected);

                for (size_t i = 0; i < projected.size() && i < result.pts_left_match.size(); ++i) {
                    cv::Point2f obs_v = toView_L(result.pts_left_match[i]);
                    cv::Point pd(static_cast<int>(projected[i].x - expand_L.x),
                                 static_cast<int>(projected[i].y - expand_L.y));
                    cv::Point pm(static_cast<int>(obs_v.x), static_cast<int>(obs_v.y));
                    cv::circle(p4, pd, 1, cv::Scalar(0, 255, 0), -1);
                    cv::circle(p4, pm, 1, cv::Scalar(0, 0, 255), 1);
                    cv::line(p4, pd, pm, cv::Scalar(0, 255, 255), 1);
                }
                cv::imwrite(output_dir_ + "/binary_corner_reproj" + prefix + ".png", p4);
            }
        }
    }

    // ---- 缓存更新：保存本帧位姿供下帧 warm-start ----
    if (pose.success) { state_.R_prev = pose.R; state_.t_prev = pose.t; }
    state_.has_cache = true;

    // ---- Logging ----
    result.is_first_frame = is_first;
    result.n_matched = static_cast<int>(result.pts_left_good.size());
    result.n_projected = static_cast<int>(result.pts_right_projected.size());
    addLogEntry(result, is_first, false);
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
// 策略切换 —— 根据 ROI 面积动态更换特征提取器
// ============================================================================

void StereoTracker::setExtractor(std::unique_ptr<FeatureExtractor> extractor) {
    if (!extractor) return;

    // If the new extractor is AKAZE-based, ensure camera is initialized
    // (other strategies don't need camera params at extract time)
    extractor_ = std::move(extractor);
    std::cout << "[StereoTracker] Switched to extractor: "
              << extractor_->name() << std::endl;
}

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
    offset_left(result.pts_right_projected); // projected points are in left image

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
    if (std::find(used_keys.begin(), used_keys.end(), "gpnp") == used_keys.end())
        used_keys.push_back("gpnp");

    int n_frames = static_cast<int>(logs.size());
    std::string sep(150, '=');
    std::cout << "\n" << sep << "\n                     StereoTracker Log (" << n_frames << " frames)\n" << sep << "\n";

    std::ostringstream h1, h2;
    h1 << std::setw(4) << "Frm | " << std::setw(10) << "Type | " << std::setw(7) << "Feat | "
       << std::setw(6) << "Match | " << std::setw(6) << "Proj | " << std::setw(7) << "Templ | "
       << std::setw(6) << "GPNP | " << std::setw(8) << "Disp";
    for (const auto& k : used_keys) h1 << " | " << std::setw(7) << (timing_labels.count(k) ? timing_labels.at(k) : k);
    h1 << " | " << std::setw(8) << "Total";
    std::cout << h1.str() << "\n" << std::string(150, '-') << "\n";

    for (const auto& log : logs) {
        std::ostringstream row;
        row << std::setw(4) << log.frame << " | " << std::setw(10) << (log.is_first ? "FullAKAZE" : "AKAZE+Flow")
            << " | " << std::setw(7) << log.n_kp_left << " | " << std::setw(6) << log.n_matched
            << " | " << std::setw(6) << log.n_projected << " | " << std::setw(7) << log.n_template_match
            << " | " << std::setw(6) << (log.gpnp_success ? std::to_string(log.gpnp_n_pts)+"pts" : "FAIL")
            << " | " << std::setw(8) << std::fixed << std::setprecision(1) << log.disparity_median;
        for (const auto& k : used_keys)
            row << " | " << std::setw(7) << std::fixed << std::setprecision(1)
                << (log.timing.count(k) ? log.timing.at(k) : 0.0);
        row << " | " << std::setw(8) << std::fixed << std::setprecision(1) << log.total_time_ms;
        std::cout << row.str() << "\n";
    }
    std::cout << sep << "\n" << std::endl;
}

} // namespace gpnp

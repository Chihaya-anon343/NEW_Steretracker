#include "visualization/Visualizer.hpp"
#include "common/GeometryUtils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

namespace gpnp {

Visualizer::Visualizer(const Eigen::Matrix3d& K, const std::string& output_dir)
    : K_(K)
    , output_dir_(output_dir)
{
}

// ============================================================================
// Generate All
// ============================================================================

void Visualizer::generateAll(
    const cv::Mat& left_color,
    const cv::Mat& right_color,
    const cv::Mat& template_color,
    const std::vector<cv::KeyPoint>& kp_left,
    const std::vector<cv::Point2f>& pts_left_used,
    const std::vector<cv::Point2f>& pts_right_used,
    const std::vector<cv::Point2f>& pts_right_proj,
    const std::vector<cv::DMatch>& good_matches,
    const std::vector<cv::Point2f>& pts_left_match,
    const std::vector<cv::Point2f>& pts_template_match,
    const std::vector<double>& disparity,
    const std::string& prefix,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    bool gpnp_success)
{
    drawStereoMatched(left_color, right_color, kp_left,
                      pts_left_used, pts_right_used, disparity, prefix);

    drawStereoProjection(left_color, pts_left_used, pts_right_proj, prefix);

    drawTemplateMatch(left_color, template_color, good_matches,
                      pts_left_match, pts_template_match, prefix);

    if (gpnp_success) {
        drawPoseAxes(left_color, R, t, prefix);
    }
}

// ============================================================================
// Panel 1: Stereo Matched
// ============================================================================

void Visualizer::drawStereoMatched(
    const cv::Mat& left_color,
    const cv::Mat& right_color,
    const std::vector<cv::KeyPoint>& kp_left,
    const std::vector<cv::Point2f>& pts_left_used,
    const std::vector<cv::Point2f>& pts_right_used,
    const std::vector<double>& disparity,
    const std::string& prefix)
{
    int h1 = left_color.rows, w1 = left_color.cols;
    int h2 = right_color.rows, w2 = right_color.cols;

    // Pad to same height
    int new_h = std::max(h1, h2);
    cv::Mat lv(new_h, w1, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat rv(new_h, w2, CV_8UC3, cv::Scalar(0, 0, 0));
    left_color.copyTo(lv(cv::Rect(0, 0, w1, h1)));
    right_color.copyTo(rv(cv::Rect(0, 0, w2, h2)));

    // Stitch horizontally
    cv::Mat stitched;
    cv::hconcat(lv, rv, stitched);

    // Draw matched point connections
    std::mt19937 rng(42); // fixed seed for reproducible colors
    std::uniform_int_distribution<int> dist(0, 255);
    int n_used = static_cast<int>(pts_left_used.size());

    for (int i = 0; i < n_used; ++i) {
        cv::Scalar color(dist(rng), dist(rng), dist(rng));
        cv::Point pt_l(static_cast<int>(pts_left_used[i].x),
                       static_cast<int>(pts_left_used[i].y));
        cv::Point pt_r(static_cast<int>(pts_right_used[i].x + w1),
                       static_cast<int>(pts_right_used[i].y));
        cv::line(stitched, pt_l, pt_r, color, 1, cv::LINE_AA);
    }

    // Annotate
    double disp_median = 0.0;
    if (!disparity.empty()) {
        std::vector<double> abs_disp;
        for (double d : disparity) abs_disp.push_back(std::abs(d));
        disp_median = computeMedian(std::move(abs_disp));
    }

    std::ostringstream oss;
    oss << "Stereo Matched" << prefix << " | KP:" << kp_left.size()
        << " Matched:" << n_used << " Disp:" << disp_median << "px";
    cv::putText(stitched, oss.str(), cv::Point(10, new_h - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    std::string path = output_dir_ + "/stereo_matched" + prefix + ".png";
    cv::imwrite(path, stitched);
}

// ============================================================================
// Panel 2: Stereo Projection
// ============================================================================

void Visualizer::drawStereoProjection(
    const cv::Mat& left_color,
    const std::vector<cv::Point2f>& pts_left_used,
    const std::vector<cv::Point2f>& pts_right_proj,
    const std::string& prefix)
{
    cv::Mat left_ann = left_color.clone();

    int n = static_cast<int>(pts_left_used.size());

    // Draw left feature points (blue filled circles)
    for (int i = 0; i < n; ++i) {
        cv::circle(left_ann, cv::Point(static_cast<int>(pts_left_used[i].x),
                                        static_cast<int>(pts_left_used[i].y)),
                   4, cv::Scalar(255, 0, 0), -1);
    }

    // Draw projected right points (red tilted crosses)
    for (int i = 0; i < static_cast<int>(pts_right_proj.size()); ++i) {
        cv::drawMarker(left_ann,
                       cv::Point(static_cast<int>(pts_right_proj[i].x),
                                 static_cast<int>(pts_right_proj[i].y)),
                       cv::Scalar(0, 0, 255),
                       cv::MARKER_TILTED_CROSS, 10, 2);
    }

    // Connect left↔projected with green lines
    for (int i = 0; i < n && i < static_cast<int>(pts_right_proj.size()); ++i) {
        cv::Point pt_l(static_cast<int>(pts_left_used[i].x),
                       static_cast<int>(pts_left_used[i].y));
        cv::Point pt_p(static_cast<int>(pts_right_proj[i].x),
                       static_cast<int>(pts_right_proj[i].y));
        cv::line(left_ann, pt_l, pt_p, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }

    std::string path = output_dir_ + "/stereo_projection" + prefix + ".png";
    cv::imwrite(path, left_ann);
}

// ============================================================================
// Panel 3: Template Match
// ============================================================================

void Visualizer::drawTemplateMatch(
    const cv::Mat& left_color,
    const cv::Mat& template_color,
    const std::vector<cv::DMatch>& good_matches,
    const std::vector<cv::Point2f>& pts_left_match,
    const std::vector<cv::Point2f>& pts_template_match,
    const std::string& prefix)
{
    int h1 = left_color.rows, w1 = left_color.cols;
    int ht = template_color.rows, wt = template_color.cols;

    int nh = std::max(h1, ht);
    cv::Mat lv2(nh, w1, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat tv(nh, wt, CV_8UC3, cv::Scalar(0, 0, 0));
    left_color.copyTo(lv2(cv::Rect(0, 0, w1, h1)));
    template_color.copyTo(tv(cv::Rect(0, 0, wt, ht)));

    cv::Mat st_lt;
    cv::hconcat(lv2, tv, st_lt);

    // Draw match lines
    std::mt19937 rng(123); // fixed seed
    std::uniform_int_distribution<int> dist(0, 255);
    int n_match = static_cast<int>(good_matches.size());

    for (int i = 0; i < std::min(n_match, static_cast<int>(pts_left_match.size())); ++i) {
        cv::Point pt_l(static_cast<int>(pts_left_match[i].x),
                       static_cast<int>(pts_left_match[i].y));
        cv::Point pt_t(static_cast<int>(pts_template_match[i].x + w1),
                       static_cast<int>(pts_template_match[i].y));
        cv::Scalar color(dist(rng), dist(rng), dist(rng));
        cv::line(st_lt, pt_l, pt_t, color, 1, cv::LINE_AA);
    }

    std::string path = output_dir_ + "/stereo_left_template_match" + prefix + ".png";
    cv::imwrite(path, st_lt);
}

// ============================================================================
// Panel 4: Pose Axes
// ============================================================================

void Visualizer::drawPoseAxes(
    const cv::Mat& left_color,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const std::string& prefix)
{
    cv::Mat left_axes = left_color.clone();

    const double axis_length_mm = 50.0; // mm

    // Template-frame axis endpoints
    const Eigen::Vector3d origin_T(0.0, 0.0, 0.0);
    const Eigen::Vector3d axis_x_T(axis_length_mm, 0.0, 0.0);
    const Eigen::Vector3d axis_y_T(0.0, axis_length_mm, 0.0);
    const Eigen::Vector3d axis_z_T(0.0, 0.0, axis_length_mm);

    // Transform to camera frame
    const Eigen::Vector3d origin_c = R * origin_T + t;
    const Eigen::Vector3d axis_x_c = R * axis_x_T + t;
    const Eigen::Vector3d axis_y_c = R * axis_y_T + t;
    const Eigen::Vector3d axis_z_c = R * axis_z_T + t;

    // Project to image using K_
    auto proj = [this](const Eigen::Vector3d& P) -> cv::Point {
        if (std::abs(P.z()) < 1e-6) return cv::Point(-1, -1);
        Eigen::Vector2d uv = projectToImage(P, K_);
        return cv::Point(static_cast<int>(uv.x()), static_cast<int>(uv.y()));
    };

    cv::Point o_pt = proj(origin_c);
    cv::Point x_pt = proj(axis_x_c);
    cv::Point y_pt = proj(axis_y_c);
    cv::Point z_pt = proj(axis_z_c);

    const int thickness = 2;
    cv::line(left_axes, o_pt, x_pt, cv::Scalar(0, 0, 255), thickness, cv::LINE_AA);   // X = Red
    cv::line(left_axes, o_pt, y_pt, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);   // Y = Green
    cv::line(left_axes, o_pt, z_pt, cv::Scalar(255, 0, 0), thickness, cv::LINE_AA);   // Z = Blue
    cv::putText(left_axes, "X", x_pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(left_axes, "Y", y_pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    cv::putText(left_axes, "Z", z_pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

    // Annotate
    std::ostringstream oss;
    oss << "GPNP Pose" << prefix << " | t=["
        << static_cast<int>(t(0)) << ","
        << static_cast<int>(t(1)) << ","
        << static_cast<int>(t(2)) << "]mm";
    cv::putText(left_axes, oss.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    std::string path = output_dir_ + "/stereo_axes" + prefix + ".png";
    cv::imwrite(path, left_axes);
}

} // namespace gpnp

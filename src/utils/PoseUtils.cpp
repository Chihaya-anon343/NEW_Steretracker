#include "utils/PoseUtils.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

namespace gpnp {

// ============================================================================
// Template Loading
// ============================================================================

std::vector<TemplateData> loadTemplates(
    const std::string& template_dir,
    bool reshape_to_1x2) {

    std::vector<TemplateData> templates;
    namespace fs = std::filesystem;

    if (!fs::exists(template_dir) || !fs::is_directory(template_dir)) {
        std::cerr << "[PoseUtils] Template directory not found: "
                  << template_dir << std::endl;
        return templates;
    }

    // Regex to match "ANGLE_degrees.txt" and extract the angle
    std::regex txt_regex(R"(^(\d+)_degrees\.txt$)");
    std::smatch match;

    for (const auto& entry : fs::directory_iterator(template_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string filename = entry.path().filename().string();
        if (!std::regex_match(filename, match, txt_regex)) continue;

        int angle = std::stoi(match[1].str());

        // Build corresponding .png path
        fs::path png_path = entry.path();
        png_path.replace_extension(".png");

        if (!fs::exists(png_path)) {
            std::cerr << "[PoseUtils] Missing PNG for angle " << angle
                      << ": " << png_path << std::endl;
            continue;
        }

        // Read corner coordinates
        std::vector<cv::Point2f> corners = readCorners(entry.path().string());
        if (corners.empty()) {
            std::cerr << "[PoseUtils] No corners found in: "
                      << entry.path() << std::endl;
            continue;
        }

        // Load template image
        cv::Mat image = cv::imread(png_path.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "[PoseUtils] Failed to load: " << png_path << std::endl;
            continue;
        }

        if (reshape_to_1x2) {
            image = image.reshape(1, 2);
        }

        // Build TemplateData
        TemplateData tmpl;
        tmpl.angle = angle;
        tmpl.image = image;
        tmpl.image_bool = (image > 127);
        tmpl.corners = std::move(corners);
        tmpl.template_width = image.cols;
        tmpl.template_height = image.rows;

        templates.push_back(std::move(tmpl));
    }

    // Sort by ascending angle
    std::sort(templates.begin(), templates.end(),
              [](const TemplateData& a, const TemplateData& b) {
                  return a.angle < b.angle;
              });

    std::cout << "[PoseUtils] Loaded " << templates.size()
              << " templates from " << template_dir << std::endl;

    return templates;
}

// ============================================================================
// Overlap / IoU
// ============================================================================

double calculateOverlap(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.size() == b.size());
    CV_Assert(a.type() == CV_8UC1 && b.type() == CV_8UC1);

    cv::Mat intersection, union_;
    cv::bitwise_and(a, b, intersection);
    cv::bitwise_or(a, b, union_);

    int inter_count = cv::countNonZero(intersection);
    int union_count = cv::countNonZero(union_);

    if (union_count == 0) return 0.0;
    return static_cast<double>(inter_count) / static_cast<double>(union_count);
}

// ============================================================================
// ROI Normalization
// ============================================================================

std::pair<cv::Mat, bool> extractAndNormalizeRoi(
    const cv::Mat& binary_img, int target_size) {

    if (binary_img.empty() || target_size <= 0) {
        return {cv::Mat(), false};
    }

    // Find all non-zero (white) pixels
    std::vector<cv::Point> pts;
    cv::findNonZero(binary_img, pts);
    if (pts.empty()) {
        return {cv::Mat(), false};
    }

    // Compute bounding box
    int x_min = binary_img.cols, x_max = 0;
    int y_min = binary_img.rows, y_max = 0;
    for (const auto& pt : pts) {
        x_min = std::min(x_min, pt.x);
        x_max = std::max(x_max, pt.x);
        y_min = std::min(y_min, pt.y);
        y_max = std::max(y_max, pt.y);
    }

    // Expand to square
    int side = std::max(x_max - x_min + 1, y_max - y_min + 1);
    int cx = (x_min + x_max) / 2;
    int cy = (y_min + y_max) / 2;
    int x1 = std::max(0, cx - side / 2);
    int y1 = std::max(0, cy - side / 2);
    int x2 = std::min(binary_img.cols, x1 + side);
    int y2 = std::min(binary_img.rows, y1 + side);

    // Recompute x1/y1 in case x2/y2 were clamped
    x1 = std::max(0, x2 - side);
    y1 = std::max(0, y2 - side);

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    if (roi.width <= 0 || roi.height <= 0) {
        return {cv::Mat(), false};
    }

    cv::Mat square = binary_img(roi);
    cv::Mat result;
    cv::resize(square, result, cv::Size(target_size, target_size),
               0, 0, cv::INTER_NEAREST);

    return {result, true};
}

// ============================================================================
// Corner File Parsing
// ============================================================================

std::vector<cv::Point2f> readCorners(const std::string& txt_path) {
    std::vector<cv::Point2f> corners;
    std::ifstream file(txt_path);
    if (!file.is_open()) {
        std::cerr << "[PoseUtils] Cannot open: " << txt_path << std::endl;
        return corners;
    }

    // Regex: "Corner_N: X, Y"  (coordinates may be negative or decimal)
    std::regex corner_regex(R"(Corner_\d+:\s*([-\d.]+),\s*([-\d.]+))");
    std::string line;

    while (std::getline(file, line)) {
        // Skip comment lines and empty lines
        if (line.empty() || line[0] == '#') continue;

        std::smatch match;
        if (std::regex_search(line, match, corner_regex) && match.size() == 3) {
            float x = std::stof(match[1].str());
            float y = std::stof(match[2].str());
            corners.emplace_back(x, y);
        }
    }

    return corners;
}

// ============================================================================
// Point Ordering
// ============================================================================

std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts) {
    if (pts.size() != 4) return pts;

    std::vector<cv::Point2f> ordered(4);

    // Compute sum and diff for each point
    std::vector<double> sum(4), diff(4);
    for (int i = 0; i < 4; ++i) {
        sum[i]  = pts[i].x + pts[i].y;
        diff[i] = pts[i].y - pts[i].x;
    }

    // TL = min(sum), BR = max(sum)
    int tl_idx = static_cast<int>(std::distance(
        sum.begin(), std::min_element(sum.begin(), sum.end())));
    int br_idx = static_cast<int>(std::distance(
        sum.begin(), std::max_element(sum.begin(), sum.end())));

    ordered[0] = pts[tl_idx];  // TL
    ordered[2] = pts[br_idx];  // BR

    // Remaining two: TR has larger diff (y-x), BL has smaller
    std::vector<int> remaining;
    for (int i = 0; i < 4; ++i) {
        if (i != tl_idx && i != br_idx) remaining.push_back(i);
    }

    if (diff[remaining[0]] > diff[remaining[1]]) {
        ordered[1] = pts[remaining[0]];  // TR
        ordered[3] = pts[remaining[1]];  // BL
    } else {
        ordered[1] = pts[remaining[1]];  // TR
        ordered[3] = pts[remaining[0]];  // BL
    }

    return ordered;
}

} // namespace gpnp

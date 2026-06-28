#include "feature/AkazeExtractor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace gpnp {

AkazeExtractor::AkazeExtractor(double scale)
    : scale_(scale)
{
    if (scale <= 0.0 || scale > 1.0) {
        throw std::invalid_argument("AkazeExtractor: scale must be in (0, 1]");
    }
    akaze_ = cv::AKAZE::create();
}

AkazeExtractor::~AkazeExtractor() = default;

AkazeExtractor::AkazeExtractor(AkazeExtractor&&) noexcept = default;
AkazeExtractor& AkazeExtractor::operator=(AkazeExtractor&&) noexcept = default;

// ============================================================================
// Feature Extraction
// ============================================================================

FeatureSet AkazeExtractor::extract(const cv::Mat& gray) {
    FeatureSet result;

    cv::Mat working_img;
    if (scale_ < 1.0) {
        cv::resize(gray, working_img, cv::Size(), scale_, scale_, cv::INTER_LINEAR);
    } else {
        working_img = gray; // shallow copy, no data duplication
    }

    // Detect and compute on (possibly scaled) image
    akaze_->detectAndCompute(working_img, cv::noArray(),
                             result.keypoints, result.descriptors);

    // Rescale keypoint coordinates back to original image size
    if (scale_ < 1.0) {
        double inv_scale = 1.0 / scale_;
        for (auto& kp : result.keypoints) {
            kp.pt.x *= inv_scale;
            kp.pt.y *= inv_scale;
            kp.size *= inv_scale;
        }
    }

    // Convert keypoints to points (equivalent to cv2.KeyPoint_convert)
    cv::KeyPoint::convert(result.keypoints, result.points);
    result.num_keypoints = static_cast<int>(result.keypoints.size());

    return result;
}

// ============================================================================
// Template Extraction
// ============================================================================

TemplateData AkazeExtractor::extractTemplate(const std::string& template_path,
                                              double real_width_mm,
                                              double real_height_mm) {
    TemplateData tmpl;

    // Load template images
    tmpl.color_image = cv::imread(template_path, cv::IMREAD_COLOR);
    tmpl.gray_image = cv::imread(template_path, cv::IMREAD_GRAYSCALE);

    if (tmpl.gray_image.empty()) {
        throw std::runtime_error("Cannot read template image: " + template_path);
    }

    tmpl.template_width = tmpl.gray_image.cols;
    tmpl.template_height = tmpl.gray_image.rows;

    // Extract features at original scale (no downscaling for template)
    cv::Ptr<cv::AKAZE> full_akaze = cv::AKAZE::create();
    full_akaze->detectAndCompute(tmpl.gray_image, cv::noArray(),
                                  tmpl.keypoints, tmpl.descriptors);

    // Compute 3D coordinates (template is planar, z=0)
    // Equivalent to Python:
    //   pts_3d[i,0] = kp.pt[0] / tw * real_width_mm
    //   pts_3d[i,1] = kp.pt[1] / th * real_height_mm
    //   pts_3d[i,2] = 0.0
    const double tw = static_cast<double>(tmpl.template_width);
    const double th = static_cast<double>(tmpl.template_height);
    tmpl.pts_3d.reserve(tmpl.keypoints.size());

    for (const auto& kp : tmpl.keypoints) {
        Eigen::Vector3d pt;
        pt.x() = kp.pt.x / tw * real_width_mm;
        pt.y() = kp.pt.y / th * real_height_mm;
        pt.z() = 0.0;
        tmpl.pts_3d.push_back(pt);
    }

    return tmpl;
}

} // namespace gpnp

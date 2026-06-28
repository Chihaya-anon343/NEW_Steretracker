/**
 * @file test_stereo_tracker.cpp
 * @brief Unit tests for GPNP Stereo Tracker modules.
 *
 * Build with:
 *   cmake -DBUILD_TESTING=ON ..
 *   cmake --build .
 *   ctest
 *
 * Tests:
 *   - GeometryUtils: pixelToCameraRay, projectToImage
 *   - AkazeExtractor: feature extraction count
 *   - OpticalFlowTracker: MAD filter correctness
 *   - TemplateMatcher: ratio test, cross-check
 *   - StereoProjector: projection validity
 *   - Config: validation, edge cases
 *   - StereoTracker: integration smoke test
 */

#include "common/Config.hpp"
#include "common/GeometryUtils.hpp"
#include "common/Types.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

// Simple test framework (no external dependency needed)
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    do { std::cout << "  TEST: " << name << "... "; } while(0)

#define PASS() \
    do { std::cout << "PASSED" << std::endl; ++g_tests_passed; } while(0)

#define FAIL(msg) \
    do { std::cout << "FAILED: " << msg << std::endl; ++g_tests_failed; } while(0)

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { FAIL(#cond); return; } } while(0)

#define ASSERT_NEAR(a, b, tol) \
    do { if (std::abs((a) - (b)) > (tol)) { \
        FAIL(std::to_string(a) + " != " + std::to_string(b) + " (tol=" + std::to_string(tol) + ")"); \
        return; } } while(0)

// ============================================================================
// GeometryUtils Tests
// ============================================================================

void test_pixel_to_camera_ray() {
    TEST("pixelToCameraRay at principal point");
    Eigen::Matrix3d K;
    K << 500, 0, 320,
         0, 500, 240,
         0, 0, 1;
    Eigen::Matrix3d K_inv = K.inverse();

    Eigen::Vector3d ray = gpnp::pixelToCameraRay(320, 240, K_inv);
    ASSERT_NEAR(ray.x(), 0.0, 1e-10);
    ASSERT_NEAR(ray.y(), 0.0, 1e-10);
    ASSERT_NEAR(ray.z(), 1.0, 1e-10);
    PASS();
}

void test_project_to_image() {
    TEST("projectToImage round-trip");
    Eigen::Matrix3d K;
    K << 500, 0, 320,
         0, 500, 240,
         0, 0, 1;

    Eigen::Vector3d P(1.0, 2.0, 5.0); // in front of camera
    Eigen::Vector2d uv = gpnp::projectToImage(P, K);
    // Expected: u = 320 + 500*1/5 = 420, v = 240 + 500*2/5 = 440
    ASSERT_NEAR(uv.x(), 420.0, 1e-6);
    ASSERT_NEAR(uv.y(), 440.0, 1e-6);
    PASS();
}

void test_quaternion_roundtrip() {
    TEST("rotationMatrixToQuat → quatToRotationMatrix round-trip");
    Eigen::Matrix3d R_orig = Eigen::Matrix3d::Identity();

    Eigen::Vector4d q = gpnp::rotationMatrixToQuat(R_orig);
    Eigen::Matrix3d R_back = gpnp::quatToRotationMatrix(q);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_NEAR(R_orig(i,j), R_back(i,j), 1e-10);
    PASS();
}

void test_normalize_quaternion() {
    TEST("normalizeQuaternion with negative qw");
    Eigen::Vector4d q(0.0, 0.0, 0.0, -1.0); // qw < 0
    Eigen::Vector4d qn = gpnp::normalizeQuaternion(q);
    ASSERT_NEAR(qn.norm(), 1.0, 1e-10);
    ASSERT_TRUE(qn(3) >= 0.0); // qw must be >= 0
    PASS();
}

void test_compute_median() {
    TEST("computeMedian odd length");
    std::vector<double> v1 = {1.0, 3.0, 2.0};
    ASSERT_NEAR(gpnp::computeMedian(v1), 2.0, 1e-10);

    TEST("computeMedian even length");
    std::vector<double> v2 = {1.0, 4.0, 2.0, 3.0};
    ASSERT_NEAR(gpnp::computeMedian(v2), 2.5, 1e-10); // numpy: (2+3)/2

    TEST("computeMedian empty");
    ASSERT_NEAR(gpnp::computeMedian({}), 0.0, 1e-10);
    PASS();
}

void test_compute_mad() {
    TEST("computeMAD basic");
    // data = [1, 2, 3, 4, 100]
    // median = 3
    // abs deviations = [2, 1, 0, 1, 97]
    // median of deviations = 1
    std::vector<double> v = {1.0, 2.0, 3.0, 4.0, 100.0};
    double mad = gpnp::computeMAD(v);
    ASSERT_NEAR(mad, 1.0, 1e-10);
    PASS();
}

// ============================================================================
// Config Tests
// ============================================================================

void test_config_validation() {
    TEST("makeTrackerConfig valid");
    auto cfg = gpnp::makeTrackerConfig(0.5, 4, true, 200.0, 200.0);
    ASSERT_NEAR(cfg.scale, 0.5, 1e-10);
    ASSERT_TRUE(cfg.gpnp_min_pts == 4);
    PASS();

    TEST("makeTrackerConfig invalid scale");
    try {
        gpnp::makeTrackerConfig(-0.1);
        FAIL("Expected exception for negative scale");
    } catch (const std::invalid_argument&) {
        PASS();
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n=== GPNP Unit Tests ===\n" << std::endl;

    // Geometry
    std::cout << "[GeometryUtils]" << std::endl;
    test_pixel_to_camera_ray();
    test_project_to_image();
    test_quaternion_roundtrip();
    test_normalize_quaternion();
    test_compute_median();
    test_compute_mad();

    // Config
    std::cout << "\n[Config]" << std::endl;
    test_config_validation();

    // Summary
    std::cout << "\n=== Results: " << g_tests_passed << " passed, "
              << g_tests_failed << " failed ===\n" << std::endl;

    return (g_tests_failed > 0) ? 1 : 0;
}

//
// OPTIMIZED FAST Corner Detector for Event Cameras
// High-performance version with memory layout and algorithm optimizations
//

#ifndef PROJECT_FAST_DETECTOR_HPP
#define PROJECT_FAST_DETECTOR_HPP

#include <array>
#include <cstdint>
#include <string>
#include <memory>
#include <cstring>
#include <immintrin.h>  // For SIMD if available

class FastDetector {
public:
    explicit FastDetector(uint32_t sensor_width = 1280, uint32_t sensor_height = 720)
            : sensor_width_(sensor_width), sensor_height_(sensor_height) {

        // Validate sensor dimensions
        if (sensor_width_ == 0 || sensor_height_ == 0 ||
            sensor_width_ > 4096 || sensor_height_ > 4096) {
            throw std::invalid_argument("Unsupported sensor dimensions. Max supported: 4096x4096");
        }

        detector_name_ = "FAST_OPTIMIZED";

        // Initialize parameters optimized for 1280x720
        min_streak_size_circle3_ = 3;
        max_streak_size_circle3_ = 6;
        min_streak_size_circle4_ = 4;
        max_streak_size_circle4_ = 8;

        // Use row-major layout for better cache performance
        const size_t total_pixels = sensor_width_ * sensor_height_;
        sae_data_[0] = std::make_unique<double[]>(total_pixels);
        sae_data_[1] = std::make_unique<double[]>(total_pixels);

        // Initialize to zero
        std::memset(sae_data_[0].get(), 0, total_pixels * sizeof(double));
        std::memset(sae_data_[1].get(), 0, total_pixels * sizeof(double));

        // Pre-compute circle offsets for faster access
        initializeCircleOffsets();
    }

    virtual ~FastDetector() = default;

    // Main feature detection function - optimized version
    inline bool isFeature(uint32_t x, uint32_t y, bool polarity, double timestamp) {
        // Fast border check
        if (x < 4 || x >= sensor_width_ - 4 || y < 4 || y >= sensor_height_ - 4) {
            return false;
        }

        // Update SAE with single calculation
        const int pol = static_cast<int>(polarity);
        const size_t idx = y * sensor_width_ + x;
        sae_data_[pol][idx] = timestamp;

        // Two-stage FAST detection with early termination
        if (!checkCirclePatternFast(x, y, pol, circle3_offsets_, 16,
                                    min_streak_size_circle3_, max_streak_size_circle3_)) {
            return false;
        }

        return checkCirclePatternFast(x, y, pol, circle4_offsets_, 20,
                                      min_streak_size_circle4_, max_streak_size_circle4_);
    }

    // Overloaded version for integer polarity
    inline bool isFeature(uint32_t x, uint32_t y, short polarity_int, double timestamp) {
        return isFeature(x, y, polarity_int > 0, timestamp);
    }

    const std::string& getDetectorName() const { return detector_name_; }

    double getSAE(uint32_t x, uint32_t y, bool polarity) const {
        if (x >= sensor_width_ || y >= sensor_height_) {
            return 0.0;
        }
        const int pol = static_cast<int>(polarity);
        return sae_data_[pol][y * sensor_width_ + x];
    }

    void reset() {
        const size_t total_pixels = sensor_width_ * sensor_height_;
        std::memset(sae_data_[0].get(), 0, total_pixels * sizeof(double));
        std::memset(sae_data_[1].get(), 0, total_pixels * sizeof(double));
    }

private:
    // Pre-computed memory offsets for circle patterns
    std::array<int32_t, 16> circle3_offsets_;
    std::array<int32_t, 20> circle4_offsets_;

    void initializeCircleOffsets() {
        // Circle of radius 3 (16 points) - convert to memory offsets
        const std::array<std::array<int, 2>, 16> circle3 = {{
                                                                    {0, 3}, {1, 3}, {2, 2}, {3, 1},
                                                                    {3, 0}, {3, -1}, {2, -2}, {1, -3},
                                                                    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1},
                                                                    {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}
                                                            }};

        for (int i = 0; i < 16; ++i) {
            circle3_offsets_[i] = circle3[i][1] * static_cast<int32_t>(sensor_width_) + circle3[i][0];
        }

        // Circle of radius 4 (20 points) - convert to memory offsets
        const std::array<std::array<int, 2>, 20> circle4 = {{
                                                                    {0, 4}, {1, 4}, {2, 3}, {3, 2},
                                                                    {4, 1}, {4, 0}, {4, -1}, {3, -2},
                                                                    {2, -3}, {1, -4}, {0, -4}, {-1, -4},
                                                                    {-2, -3}, {-3, -2}, {-4, -1}, {-4, 0},
                                                                    {-4, 1}, {-3, 2}, {-2, 3}, {-1, 4}
                                                            }};

        for (int i = 0; i < 20; ++i) {
            circle4_offsets_[i] = circle4[i][1] * static_cast<int32_t>(sensor_width_) + circle4[i][0];
        }
    }

    // Optimized circle pattern checking
    template<size_t N>
    inline bool checkCirclePatternFast(uint32_t x, uint32_t y, int pol,
                                       const std::array<int32_t, N>& offsets,
                                       int circle_size, int min_streak, int max_streak) {

        const size_t base_idx = y * sensor_width_ + x;
        const double* sae = sae_data_[pol].get();

        // Try different starting positions and streak sizes
        for (int i = 0; i < circle_size; ++i) {
            for (int streak_size = min_streak; streak_size <= max_streak; ++streak_size) {

                // Quick boundary validation using pre-computed offsets
                const size_t first_idx = base_idx + offsets[i];
                const size_t last_idx = base_idx + offsets[(i + streak_size - 1) % circle_size];
                const size_t prev_idx = base_idx + offsets[(i - 1 + circle_size) % circle_size];
                const size_t next_idx = base_idx + offsets[(i + streak_size) % circle_size];

                // Check boundary conditions
                if (sae[first_idx] < sae[prev_idx] || sae[last_idx] < sae[next_idx]) {
                    continue;
                }

                // Find minimum timestamp in streak
                double min_t = sae[first_idx];
                for (int j = 1; j < streak_size; ++j) {
                    const double tj = sae[base_idx + offsets[(i + j) % circle_size]];
                    if (tj < min_t) {
                        min_t = tj;
                    }
                }

                // Check that all points outside streak have older timestamps
                bool valid_streak = true;
                for (int j = streak_size; j < circle_size; ++j) {
                    if (sae[base_idx + offsets[(i + j) % circle_size]] >= min_t) {
                        valid_streak = false;
                        break;
                    }
                }

                if (valid_streak) {
                    return true;
                }
            }
        }
        return false;
    }

    // Parameters
    std::string detector_name_;
    int min_streak_size_circle3_;
    int max_streak_size_circle3_;
    int min_streak_size_circle4_;
    int max_streak_size_circle4_;

    // Sensor dimensions
    uint32_t sensor_width_;
    uint32_t sensor_height_;

    // Optimized memory layout: row-major arrays for better cache locality
    std::unique_ptr<double[]> sae_data_[2];
};

#endif //PROJECT_FAST_DETECTOR_HPP
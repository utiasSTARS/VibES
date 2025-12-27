/**
 * @file ev2image.hpp
 * @brief Utilities to convert event buffers into various 2D image representations.
 *
 * This file implements methods to visualize high-temporal resolution event data
 * as static frames (Time Surfaces, Event Counts, Binary maps, etc.).
 *
 * The visualization techniques (Time Decay, Average Timestamp) are adapted from:
 * He, B., Wang, Z., Zhou, Y., Chen, J., Singh, C. D., Li, H., ... & Fermüller, C. (2024).
 * "Microsaccade-inspired event camera for robotics". Science Robotics, 9(90), eadj8124.
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#ifndef PROJECT_EV2IMAGE_HPP
#define PROJECT_EV2IMAGE_HPP

#include <metavision/sdk/core/pipeline/stage.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

#include "utils.hpp"

/**
 * @struct ImageResults
 * @brief Container for multiple image representations generated simultaneously.
 *
 * Useful when you need to debug or visualize all aspects of the event stream at once.
 */
struct ImageResults {
    cv::Mat img_bin;          ///< Binary map (0 or 255) indicating event presence.
    cv::Mat img_cnt_gray;     ///< Raw event count (16-bit).
    cv::Mat img_ts;           ///< Time Surface (exponential decay).
    cv::Mat img_avgts;        ///< Average Timestamp per pixel.
    cv::Mat img_cnt_color;    ///< Event count with JET colormap.
    cv::Mat img_ts_color;     ///< Time Surface with JET colormap.
    cv::Mat img_avgts_color;  ///< Average Timestamp with BONE colormap (smoothed).
};

/**
 * @enum ImageType
 * @brief Selects the specific type of image to generate for the optimized converter.
 */
enum ImageType {
    BIN,              ///< Simple binary map (Event = 255, No Event = 0).
    COUNT,            ///< Event accumulation count with JET colormap.
    COUNT_GRAY,       ///< Raw 16-bit event accumulation count.
    TS,               ///< Time Surface (Exponential decay) with JET colormap.
    TS_GRAY,          ///< Raw 32-bit float Time Surface.
    AVERAGE_TS,       ///< Average timestamp per pixel with BONE colormap.
    AVERAGE_TS_GRAY   ///< Raw 32-bit float Average Timestamp.
};

/**
 * @brief Generates all available image representations from an event buffer.
 *
 * This function computes binary, count, time surface, and average timestamp images
 * simultaneously. It is computationally more expensive than generating a single type.
 *
 * @param evs Input buffer of Metavision EventCD events.
 * @param MAT_ROWS Height of the sensor/image.
 * @param MAT_COLS Width of the sensor/image.
 * @return ImageResults Struct containing all generated OpenCV matrices.
 */
inline ImageResults ev2img_metavision(Metavision::Stage::EventBuffer &evs, int MAT_ROWS, int MAT_COLS) {
    ImageResults results;

    if (evs.empty()) {
        // Return empty matrices if no events
        results.img_bin = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
        results.img_cnt_gray = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_16UC1);
        results.img_ts = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
        results.img_avgts = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_32FC1);
        results.img_cnt_color = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC3);
        results.img_ts_color = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC3);
        results.img_avgts_color = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC3);
        return results;
    }

    // Get timestamp of the last event to serve as reference t0
    auto t_end = (evs.end() - 1)->t;

    // Initialize accumulation matrices
    cv::Mat eventBinary = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_8UC1);
    cv::Mat eventCount = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_16UC1);
    cv::Mat eventTS = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_32FC1);
    cv::Mat eventSum = cv::Mat::zeros(cv::Size(MAT_COLS, MAT_ROWS), CV_32FC1);

    // Single pass accumulation
    for (auto &ev: evs) {
        // Calculate deltaT: time elapsed since the event occurred relative to the batch end
        float deltaT = float(t_end - ev.t) / 1000000.0f; // Microseconds to seconds

        // Update event count with overflow protection (uint16 max is 65535)
        if (eventCount.at<uint16_t>(ev.y, ev.x) < 65535) {
            eventCount.at<uint16_t>(ev.y, ev.x) += 1;
        }

        // Set binary image pixel
        eventBinary.at<uchar>(ev.y, ev.x) = 255;

        // Calculate exponential time decay: exp(-deltaT / tau)
        // Tau is set to 0.02s (20ms)
        eventTS.at<float>(ev.y, ev.x) = std::exp(-deltaT / 0.02f);

        // Accumulate time sum for average calculation
        eventSum.at<float>(ev.y, ev.x) += deltaT;
    }

    // Assign basic results
    results.img_bin = eventBinary;
    eventCount.copyTo(results.img_cnt_gray);
    results.img_ts = eventTS;

    // Calculate average timestamp: Sum(deltaT) / Count
    cv::divide(eventSum, eventCount, results.img_avgts, 1.0f, CV_32FC1);

    // --- Visualization & Normalization ---
    cv::Mat eventCount_norm, eventTS_norm, img_avgts_color_temp;

    // 1. Event Count Visualization
    cv::normalize(eventCount, eventCount_norm, 0, 255, cv::NORM_MINMAX);
    eventCount_norm.convertTo(eventCount_norm, CV_8UC1);
    cv::applyColorMap(eventCount_norm, results.img_cnt_color, cv::COLORMAP_JET);

    // 2. Time Surface Visualization
    cv::normalize(eventTS, eventTS_norm, 0, 255, cv::NORM_MINMAX);
    eventTS_norm.convertTo(eventTS_norm, CV_8UC1);
    cv::applyColorMap(eventTS_norm, results.img_ts_color, cv::COLORMAP_JET);

    // 3. Average Timestamp Visualization
    cv::normalize(results.img_avgts, img_avgts_color_temp, 0, 255, cv::NORM_MINMAX);
    img_avgts_color_temp.convertTo(img_avgts_color_temp, CV_8UC1);
    // Apply median blur to reduce salt-and-pepper noise in sparse regions
    cv::medianBlur(img_avgts_color_temp, img_avgts_color_temp, 3);
    cv::applyColorMap(img_avgts_color_temp, results.img_avgts_color, cv::COLORMAP_BONE);

    return results;
}

/**
 * @brief Optimized converter that generates a single specific image type.
 *
 * This function is efficient for runtime pipelines as it only allocates memory
 * and performs calculations required for the requested image type.
 *
 * @param evs Input buffer of Metavision events.
 * @param output Output OpenCV matrix (will be reallocated/resized if necessary).
 * @param width Sensor width.
 * @param height Sensor height.
 * @param type The specific type of visualization required.
 */
inline void ev2img_metavision(Metavision::Stage::EventBuffer &evs, cv::Mat &output, int width, int height, ImageType type) {
    if (evs.empty()) {
        if(output.empty() || output.cols != width || output.rows != height) {
            output = cv::Mat::zeros(height, width, (type == COUNT_GRAY) ? CV_16UC1 : ((type == TS_GRAY || type == AVERAGE_TS_GRAY) ? CV_32FC1 : CV_8UC1));
        } else {
            output.setTo(0);
        }
        return;
    }

    Metavision::timestamp t_end = (evs.end() - 1)->t;
    const float microsec_to_sec = 1.0f / 1000000.0f;
    const float time_constant = 0.02f;

    // Determine requirements based on requested type
    bool need_count = (type == COUNT || type == COUNT_GRAY || type == AVERAGE_TS || type == AVERAGE_TS_GRAY);
    bool need_sum   = (type == AVERAGE_TS || type == AVERAGE_TS_GRAY);
    bool need_ts    = (type == TS || type == TS_GRAY);

    cv::Mat count_mat, sum_mat, ts_mat;

    // Allocate only what is needed
    if (need_count) count_mat = cv::Mat::zeros(height, width, CV_16UC1);
    if (need_sum)   sum_mat   = cv::Mat::zeros(height, width, CV_32FC1);
    if (need_ts)    ts_mat    = cv::Mat::zeros(height, width, CV_32FC1);

    // Single pass processing
    if (type == BIN) {
        output = cv::Mat::zeros(height, width, CV_8UC1);
        for (auto &ev : evs) {
            output.at<uchar>(ev.y, ev.x) = 255;
        }
        return;
    }

    for (auto &ev : evs) {
        float deltaT = float(t_end - ev.t) * microsec_to_sec;

        if (need_count) {
            if (count_mat.at<uint16_t>(ev.y, ev.x) < 65535) {
                count_mat.at<uint16_t>(ev.y, ev.x)++;
            }
        }

        if (need_sum) {
            sum_mat.at<float>(ev.y, ev.x) += deltaT;
        }

        if (need_ts) {
            ts_mat.at<float>(ev.y, ev.x) = std::exp(-deltaT / time_constant);
        }
    }

    // Final Post-processing based on requested type
    switch (type) {
        case COUNT_GRAY:
            output = count_mat;
            break;

        case COUNT: {
            cv::Mat count_norm;
            cv::normalize(count_mat, count_norm, 0, 255, cv::NORM_MINMAX);
            count_norm.convertTo(count_norm, CV_8UC1);
            cv::applyColorMap(count_norm, output, cv::COLORMAP_JET);
            break;
        }

        case TS_GRAY:
            output = ts_mat;
            break;

        case TS: {
            cv::Mat ts_norm;
            cv::normalize(ts_mat, ts_norm, 0, 255, cv::NORM_MINMAX);
            ts_norm.convertTo(ts_norm, CV_8UC1);
            cv::applyColorMap(ts_norm, output, cv::COLORMAP_JET);
            break;
        }

        case AVERAGE_TS_GRAY:
            // Safe division handling
            output = cv::Mat::zeros(height, width, CV_32FC1);
            cv::divide(sum_mat, count_mat, output, 1.0f, CV_32FC1);
            break;

        case AVERAGE_TS: {
            cv::Mat avg_ts;
            cv::divide(sum_mat, count_mat, avg_ts, 1.0f, CV_32FC1);

            cv::Mat avg_norm;
            cv::normalize(avg_ts, avg_norm, 0, 255, cv::NORM_MINMAX);
            avg_norm.convertTo(avg_norm, CV_8UC1);
            cv::medianBlur(avg_norm, avg_norm, 3);
            cv::applyColorMap(avg_norm, output, cv::COLORMAP_BONE);
            break;
        }
        default:
            break;
    }
}

#endif //PROJECT_EV2IMAGE_HPP
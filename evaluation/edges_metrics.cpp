/**
 * @file image_processor.cpp
 * @brief Evaluation utility for vibration analysis using Computer Vision.
 *
 * This tool quantifies the overlap between "vibration" pixels (detected from events)
 * and "structural" pixels (from standard grayscale frames).
 *
 * Pipeline:
 * 1. **Preprocessing:** Otsu thresholding and Canny edge detection on the reference gray image.
 * 2. **Registration:** Aligns the event-based "vibration" image to the gray image using ECC (Enhanced Correlation Coefficient).
 * 3. **Morphology:** Cleans up noise using dilation/closing and area filtering (bwareaopen).
 * 4. **Analysis:** Counts overlapping pixels to determine a "Match Ratio".
 *
 * @author Vincenzo Polizzi - STARS Lab
 * @date Dec 27 2025
 */

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>

using namespace cv;
using namespace std;

class ImageProcessor {
private:
    // Configuration
    const int IMG_NUM = 700;        ///< Total number of frames to process.
    const int SEARCH_RADIUS = 2;    ///< Pixel radius for relaxed matching (tolerance).

    // Statistics Buffers
    vector<int> grey_num;              ///< Count of structural pixels in the gray image.
    vector<int> vib_match_num_in_gray; ///< Count of vibration pixels that match structure.
    vector<int> vib_no_match_num;      ///< Count of vibration pixels that do NOT match structure (noise).
    vector<int> vib_match_num;         ///< Total vibration pixels considered matches.
    vector<int> vib_all_num;           ///< Total vibration pixels detected.

    // Data Paths (Configure these before running)
    string frame_gray_path = "PATH_TO_AMIEV_FRAMES/";
    string frame_novib_path = "PATH_TO_AMIEV_GRAY_NO_VIB/";
    string frame_vib_path = "PATH_TO_AMIEV_GRAY_VIB/";

    // Registration state
    Mat transformation_matrix;

public:
    ImageProcessor() {
        // Pre-allocate memory
        grey_num.resize(IMG_NUM, 0);
        vib_match_num_in_gray.resize(IMG_NUM, 0);
        vib_no_match_num.resize(IMG_NUM, 0);
        vib_match_num.resize(IMG_NUM, 0);
        vib_all_num.resize(IMG_NUM, 0);

        // Initialize with identity transform
        transformation_matrix = Mat::eye(2, 3, CV_32F);
    }

    /**
     * @brief Applies a fixed binary threshold.
     * @param thresh Threshold value normalized [0, 1].
     */
    Mat applyThreshold(const Mat &image, double thresh) {
        Mat binary;
        threshold(image, binary, thresh * 255, 255, THRESH_BINARY);
        return binary;
    }

    /**
     * @brief Applies Otsu's binarization algorithm.
     * Automatically finds the optimal threshold to minimize intra-class variance.
     */
    Mat applyOtsuThreshold(const Mat &image) {
        Mat binary;
        threshold(image, binary, 0, 255, THRESH_BINARY + THRESH_OTSU);
        return binary;
    }

    /**
     * @brief Performs morphological closing and dilation to connect fragmented edges.
     */
    Mat morphologicalOperations(const Mat &image) {
        Mat result = image.clone();

        // 1. Close: Dilation -> Erosion (Fill holes)
        Mat kernel_square = getStructuringElement(MORPH_RECT, Size(7, 7));
        morphologyEx(result, result, MORPH_CLOSE, kernel_square);

        // 2. Dilate Vertical (Connect vertical gaps)
        Mat kernel_line_v = getStructuringElement(MORPH_RECT, Size(1, 3));
        dilate(result, result, kernel_line_v);

        // 3. Dilate Horizontal (Connect horizontal gaps)
        Mat kernel_line_h = getStructuringElement(MORPH_RECT, Size(3, 1));
        dilate(result, result, kernel_line_h);

        return result;
    }

    /**
     * @brief Removes small connected components (noise/speckles).
     * Equivalent to MATLAB's bwareaopen.
     * @param minArea Minimum area in pixels to keep a component.
     */
    Mat bwareaopen(const Mat &binary, int minArea) {
        Mat result = Mat::zeros(binary.size(), CV_8U);
        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) >= minArea) {
                drawContours(result, contours, (int) i, Scalar(255), -1);
            }
        }
        return result;
    }

    /**
     * @brief Removes large connected components (background/artifacts).
     * @param maxArea Maximum area in pixels to keep a component.
     */
    Mat bwareaopenlarge(const Mat &binary, int maxArea) {
        Mat result = Mat::zeros(binary.size(), CV_8U);
        vector<vector<Point>> contours;
        // Use RETR_LIST to catch nested contours if necessary
        findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) <= maxArea) {
                drawContours(result, contours, (int) i, Scalar(255), -1);
            }
        }
        return result;
    }

    /**
     * @brief Aligns the 'moving' image to the 'fixed' image using ECC.
     * ECC (Enhanced Correlation Coefficient) is robust to illumination changes.
     */
    Mat registerImages(const Mat &fixed, const Mat &moving, bool update_transform = true) {
        try {
            Mat moving_f, fixed_f;
            moving.convertTo(moving_f, CV_32F);
            fixed.convertTo(fixed_f, CV_32F);

            // 2x3 Affine Matrix [a11 a12 b1; a21 a22 b2]
            Mat warp_matrix = Mat::eye(2, 3, CV_32F);

            // Termination: 1000 iterations or epsilon 1e-5
            TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 1e-5);

            try {
                findTransformECC(fixed_f, moving_f, warp_matrix, MOTION_AFFINE, criteria);
                // cout << "ECC registration successful" << endl; // Verbose off
            } catch (const Exception &e) {
                cout << "ECC registration failed, using identity transform" << endl;
            }

            Mat registered;
            warpAffine(moving, registered, warp_matrix, fixed.size());

            return registered;

        } catch (const Exception &e) {
            cout << "Registration error: " << e.what() << endl;
            return moving.clone();
        }
    }

    /**
     * @brief Loads and preprocesses the ground truth gray image.
     */
    Mat processGrayImage(int index) {
        string filename = "image_" + to_string(index) + ".jpg";
        string gray_path = frame_gray_path + filename;

        cout << "Loading reference: " << gray_path << endl;
        Mat gray_img = imread(gray_path, IMREAD_COLOR);

        if (gray_img.empty()) {
            throw std::runtime_error("Failed to load gray image: " + gray_path);
        }

        Mat gray;
        cvtColor(gray_img, gray, COLOR_BGR2GRAY);

        // 1. Otsu Thresholding
        Mat gray_thresh = applyOtsuThreshold(gray);

        // 2. Invert (assuming structure is dark on light background, or vice versa depending on intent)
        Mat gray_edge;
        bitwise_not(gray_thresh, gray_edge);

        // 3. Filter blobs by size
        gray_edge = bwareaopenlarge(gray_edge, 100000); // Remove massive blobs
        gray_edge = bwareaopen(gray_edge, 100);         // Remove noise

        // 4. Edge Detection
        Mat canny_edges;
        Canny(gray_edge, canny_edges, 50, 150);

        // 5. Final Binary Mask
        Mat result;
        threshold(canny_edges, result, 127, 255, THRESH_BINARY);

        return result;
    }

    /**
     * @brief Processes a single pair of Vibration/No-Vibration frames.
     *
     * 1. Loads images.
     * 2. Thresholds and cleans them.
     * 3. Registers the vibration image to the ground truth.
     * 4. Computes overlap statistics.
     */
    void processImagePair(int i, Mat gray_img) {
        string novib_path = frame_novib_path + to_string(i) + ".png";
        string vib_path = frame_vib_path + to_string(i) + ".png";

        Mat novib = imread(novib_path, IMREAD_COLOR);
        Mat vib = imread(vib_path, IMREAD_COLOR);

        if (novib.empty() || vib.empty()) {
            cout << "Skipping missing index " << i << endl;
            return;
        }

        Mat novib_gray, vib_gray;
        cvtColor(novib, novib_gray, COLOR_BGR2GRAY);
        cvtColor(vib, vib_gray, COLOR_BGR2GRAY);

        // Thresholding
        Mat vib_edge = applyThreshold(vib_gray, 0.22);
        Mat novib_edge = applyThreshold(novib_gray, 0.26);

        // Morphology
        novib_edge = morphologicalOperations(novib_edge);

        // Filtering
        medianBlur(vib_edge, vib_edge, 3);
        medianBlur(novib_edge, novib_edge, 3);
        novib_edge = bwareaopen(novib_edge, 100);

        // Registration: Align vibration events to the gray image structure
        Mat mv_vib_edge = registerImages(vib_edge, gray_img);

        // --- Visualization Composing ---
        Mat imgcolor = Mat::zeros(vib_edge.rows, vib_edge.cols, CV_8UC3);
        vector<Mat> channels1 = {Mat::zeros(vib_edge.size(), CV_8U), novib_edge, vib_edge};
        merge(channels1, imgcolor);

        Mat imgcolor2 = Mat::zeros(vib_edge.rows, vib_edge.cols, CV_8UC3);
        vector<Mat> channels2 = {Mat::zeros(vib_edge.size(), CV_8U), gray_img, mv_vib_edge};
        merge(channels2, imgcolor2);

        Mat display1, display2, full_display;
        Mat novib_c, vib_c;
        cvtColor(novib_edge, novib_c, COLOR_GRAY2BGR);
        cvtColor(vib_edge, vib_c, COLOR_GRAY2BGR);

        hconcat(vector<Mat>{novib_c, vib_c}, display1);
        hconcat(vector<Mat>{imgcolor, imgcolor2}, display2);
        vconcat(vector<Mat>{display1, display2}, full_display);

        resize(full_display, full_display, Size(full_display.cols / 2, full_display.rows / 2));
        imshow("Processing Results", full_display);
        waitKey(1);

        // --- Statistics Calculation ---
        vib_all_num[i - 1] = countNonZero(mv_vib_edge);

        Mat mv_vib_edge_tmp = mv_vib_edge.clone();
        int rows = gray_img.rows;
        int cols = gray_img.cols;

        grey_num[i - 1] = 0;
        vib_match_num_in_gray[i - 1] = 0;

        // Iterate through ground truth pixels
        for (int ii = 0; ii < rows; ii++) {
            for (int jj = 0; jj < cols; jj++) {
                // If this pixel is structural (white in gray_img)
                if (gray_img.at<uchar>(ii, jj) == 255) {
                    grey_num[i - 1]++;

                    // Search for a matching vibration pixel within radius
                    int up = max(0, ii - SEARCH_RADIUS);
                    int down = min(rows - 1, ii + SEARCH_RADIUS);
                    int left = max(0, jj - SEARCH_RADIUS);
                    int right = min(cols - 1, jj + SEARCH_RADIUS);

                    Rect search_region(left, up, right - left + 1, down - up + 1);
                    Mat region = mv_vib_edge(search_region);

                    if (countNonZero(region) > 0) {
                        vib_match_num_in_gray[i - 1]++;
                    }

                    // Prevent double counting
                    mv_vib_edge_tmp(search_region) = 0;
                }
            }
        }

        vib_no_match_num[i - 1] = countNonZero(mv_vib_edge_tmp);
        vib_match_num[i - 1] = vib_all_num[i - 1] - vib_no_match_num[i - 1];

        if (i % 50 == 0) {
            cout << "Processed " << i << "/" << IMG_NUM
                 << " | Ratio: " << (float)vib_match_num_in_gray[i-1]/grey_num[i-1] << endl;
        }
    }

    /**
     * @brief Generates a simple line plot of the matching ratio.
     */
    void plotResults() {
        vector<double> match_ratios;
        for (int i = 0; i < IMG_NUM; i++) {
            match_ratios.push_back(grey_num[i] > 0 ? (double)vib_match_num_in_gray[i] / grey_num[i] : 0.0);
        }

        int W = 1200, H = 600;
        Mat plot_img = Mat::ones(H, W, CV_8UC3) * 255;

        if (match_ratios.empty()) return;

        double max_val = *max_element(match_ratios.begin(), match_ratios.end());
        if (max_val <= 0) max_val = 1.0;

        // Draw Plot
        for (size_t i = 1; i < match_ratios.size(); i++) {
            Point pt1(50 + (i - 1) * (W - 100) / match_ratios.size(), H - 50 - (match_ratios[i - 1] / max_val * (H - 100)));
            Point pt2(50 + i * (W - 100) / match_ratios.size(), H - 50 - (match_ratios[i] / max_val * (H - 100)));
            line(plot_img, pt1, pt2, Scalar(255, 0, 0), 2);
        }

        putText(plot_img, "Match Ratio (Detected / Ground Truth)", Point(W/3, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0), 2);
        imshow("Match Ratios", plot_img);
        waitKey(0);
    }

    void run() {
        cout << "=== Image Processing Evaluation Start ===" << endl;

        // 1. Process Reference
        Mat gray_img = processGrayImage(1);

        // 2. Process Sequence
        // Note: Loop start adjusted to 1 for standard sequence processing
        for (int i = 1; i <= IMG_NUM; i++) {
            processImagePair(i, gray_img);
        }

        // 3. Output
        plotResults();

        ofstream results_file("evaluation_results.csv");
        results_file << "Index,GreyNum,VibMatch,Ratio\n";
        for (int i = 0; i < IMG_NUM; i++) {
            double ratio = (grey_num[i] > 0) ? (double)vib_match_num_in_gray[i] / grey_num[i] : 0.0;
            results_file << i + 1 << "," << grey_num[i] << "," << vib_match_num_in_gray[i] << "," << ratio << "\n";
        }
        results_file.close();
        cout << "Saved evaluation_results.csv" << endl;
    }
};

int main() {
    try {
        ImageProcessor processor;
        processor.run();
    } catch (const exception &e) {
        cerr << "Fatal Error: " << e.what() << endl;
        return -1;
    }
    return 0;
}
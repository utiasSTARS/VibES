//
// Created by viciopoli on 08/08/25.
// Fixed version with proper image registration and processing
//
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

using namespace cv;
using namespace std;

class ImageProcessor {
private:
    const int IMG_NUM = 700;
    const int SEARCH_RADIUS = 2;

    vector<int> grey_num;
    vector<int> vib_match_num_in_gray;
    vector<int> vib_no_match_num;
    vector<int> vib_match_num;
    vector<int> vib_all_num;

    // Update paths to match MATLAB structure
    string frame_gray_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/frames_amiev/";
    string frame_novib_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/results/amiev/ev/img_bin/";
    string frame_vib_path = "/home/viciopoli/datasets/event_harmeda/harmeda_dataset/results/amiev/harmeda/img_gray/";

    // Store the transformation matrix for registration
    Mat transformation_matrix;
    bool has_initial_transform = false;

public:
    ImageProcessor() {
        grey_num.resize(IMG_NUM, 0);
        vib_match_num_in_gray.resize(IMG_NUM, 0);
        vib_no_match_num.resize(IMG_NUM, 0);
        vib_match_num.resize(IMG_NUM, 0);
        vib_all_num.resize(IMG_NUM, 0);

        // Initialize with identity transform
        transformation_matrix = Mat::eye(2, 3, CV_32F);
    }

    Mat applyThreshold(const Mat &image, double thresh) {
        Mat binary;
        threshold(image, binary, thresh * 255, 255, THRESH_BINARY);
        return binary;
    }

    Mat applyOtsuThreshold(const Mat &image) {
        Mat binary;
        threshold(image, binary, 0, 255, THRESH_BINARY + THRESH_OTSU);
        return binary;
    }

    Mat morphologicalOperations(const Mat &image) {
        Mat result = image.clone();

        // Close operation with square kernel (7x7)
        Mat kernel_square = getStructuringElement(MORPH_RECT, Size(7, 7));
        morphologyEx(result, result, MORPH_CLOSE, kernel_square);

        // Dilate with vertical line (3x1) - 90 degrees
        Mat kernel_line_v = getStructuringElement(MORPH_RECT, Size(1, 3));
        dilate(result, result, kernel_line_v);

        // Dilate with horizontal line (3x1) - 0 degrees
        Mat kernel_line_h = getStructuringElement(MORPH_RECT, Size(3, 1));
        dilate(result, result, kernel_line_h);

        return result;
    }

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

    Mat bwareaopenlarge(const Mat &binary, int maxArea) {
        Mat result = Mat::zeros(binary.size(), CV_8U);
        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {
            if (contourArea(contours[i]) <= maxArea) {
                drawContours(result, contours, (int) i, Scalar(255), -1);
            }
        }
        return result;
    }

    Mat registerImages(const Mat &fixed, const Mat &moving, bool update_transform = true) {
        try {
            // Convert images to appropriate format
            Mat moving_f, fixed_f;
            moving.convertTo(moving_f, CV_32F);
            fixed.convertTo(fixed_f, CV_32F);

            // Create proper 2x3 affine matrix
            Mat warp_matrix = Mat::eye(2, 3, CV_32F);

            // Set termination criteria
            TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 1e-5);

            try {
                // Perform registration
                findTransformECC(fixed_f, moving_f, warp_matrix, MOTION_AFFINE, criteria);
                cout << "ECC registration successful" << endl;
            } catch (const Exception &e) {
                cout << "ECC registration failed, using identity transform" << endl;
                // warp_matrix remains as identity
            }

            // Apply transformation
            Mat registered;
            warp_matrix.convertTo(warp_matrix, CV_32F); // Ensure correct type
            warpAffine(moving, registered, warp_matrix, fixed.size());

            return registered;

        } catch (const Exception &e) {
            cout << "Registration error: " << e.what() << endl;
            return moving.clone();
        }
    }

    Mat processGrayImage(int index) {
        // Build filename with 4-digit zero padding
        string filename = "image_" + to_string(index) + ".jpg";
        string gray_path = frame_gray_path + filename;

        cout << "Loading gray image: " << gray_path << endl;
        Mat gray_img = imread(gray_path, IMREAD_COLOR);

        if (gray_img.empty()) {
            cout << "Could not load gray image: " << gray_path << endl;
            throw std::runtime_error("Failed to load gray image");
        }

        // Convert to grayscale
        Mat gray;
        cvtColor(gray_img, gray, COLOR_BGR2GRAY);

        // Apply Otsu thresholding (equivalent to graythresh + im2bw)
        Mat gray_thresh = applyOtsuThreshold(gray);

        // Invert (~gray_edge in MATLAB)
        Mat gray_edge;
        bitwise_not(gray_thresh, gray_edge);

        // Apply bwareaopenlarge (remove large areas > 100000)
        gray_edge = bwareaopenlarge(gray_edge, 100000);

        // Apply bwareaopen (remove small areas < 100)
        gray_edge = bwareaopen(gray_edge, 100);

        // Apply edge detection
        Mat canny_edges;
        Canny(gray_edge, canny_edges, 50, 150);

        // Convert back to binary format for consistency
        Mat result;
        threshold(canny_edges, result, 127, 255, THRESH_BINARY);

        // show the processed gray image
        imshow("Processed Gray Image", result);
        waitKey(1);

        return result;
    }

    void processImagePair(int i, Mat gray_img) {
        // Build filenames
        string novib_path = frame_novib_path + to_string(i) + ".png";
        string vib_path = frame_vib_path + to_string(i) + ".png";

        Mat novib = imread(novib_path, IMREAD_COLOR);
        Mat vib = imread(vib_path, IMREAD_COLOR);

        if (novib.empty() || vib.empty()) {
            cout << "Could not load images for index " << i << endl;
            return;
        }

        // Convert to grayscale
        Mat novib_gray, vib_gray;
        cvtColor(novib, novib_gray, COLOR_BGR2GRAY);
        cvtColor(vib, vib_gray, COLOR_BGR2GRAY);

        // Apply thresholding with fixed values from MATLAB
        Mat vib_edge = applyThreshold(vib_gray, 0.22);
        Mat novib_edge = applyThreshold(novib_gray, 0.26);

        // Morphological operations on novib_edge
        novib_edge = morphologicalOperations(novib_edge);

        // Median filtering
        medianBlur(vib_edge, vib_edge, 3);
        medianBlur(novib_edge, novib_edge, 3);

        // Remove small areas
        novib_edge = bwareaopen(novib_edge, 100);

        // Register vib_edge to gray_img
        Mat mv_vib_edge = registerImages(vib_edge, gray_img);

        // Create visualizations
        Mat imgcolor = Mat::zeros(vib_edge.rows, vib_edge.cols, CV_8UC3);
        vector<Mat> channels1(3);
        channels1[0] = Mat::zeros(vib_edge.size(), CV_8U);  // Blue
        channels1[1] = novib_edge;  // Green
        channels1[2] = vib_edge;    // Red (original vib_edge)
        merge(channels1, imgcolor);

        Mat imgcolor2 = Mat::zeros(vib_edge.rows, vib_edge.cols, CV_8UC3);
        vector<Mat> channels2(3);
        channels2[0] = Mat::zeros(vib_edge.size(), CV_8U);  // Blue
        channels2[1] = gray_img;    // Green
        channels2[2] = mv_vib_edge; // Red (registered vib_edge)
        merge(channels2, imgcolor2);

        Mat novib_color, vib_color;
        cvtColor(novib_edge, novib_color, COLOR_GRAY2BGR);
        cvtColor(vib_edge, vib_color, COLOR_GRAY2BGR);

        // Display results in subplots
        Mat display1, display2;
        hconcat(vector<Mat>{novib_color, vib_color}, display1);
        hconcat(vector<Mat>{imgcolor, imgcolor2}, display2);
        Mat full_display;
        vconcat(vector<Mat>{display1, display2}, full_display);

        // Resize for display
        resize(full_display, full_display, Size(full_display.cols / 2, full_display.rows / 2));
        imshow("Processing Results", full_display);
        waitKey(1);

        // Count vibration pixels (equivalent to find(mv_vib_edge==1))
        vib_all_num[i - 1] = countNonZero(mv_vib_edge);

        // Matching analysis
        Mat mv_vib_edge_tmp = mv_vib_edge.clone();
        int rows = gray_img.rows;
        int cols = gray_img.cols;

        grey_num[i - 1] = 0;
        vib_match_num_in_gray[i - 1] = 0;

        for (int ii = 0; ii < rows; ii++) {
            for (int jj = 0; jj < cols; jj++) {
                if (gray_img.at<uchar>(ii, jj) == 255) { // Changed from novib_edge to gray_img
                    grey_num[i - 1]++;

                    // Define search region
                    int up = max(0, ii - SEARCH_RADIUS);
                    int down = min(rows - 1, ii + SEARCH_RADIUS);
                    int left = max(0, jj - SEARCH_RADIUS);
                    int right = min(cols - 1, jj + SEARCH_RADIUS);

                    // Extract region
                    Rect search_region(left, up, right - left + 1, down - up + 1);
                    Mat region = mv_vib_edge(search_region);

                    // Check for matches (equivalent to find(mv_vib_edge(...)==1))
                    if (countNonZero(region) > 0) {
                        vib_match_num_in_gray[i - 1]++;
                    }

                    // Clear the region in temporary image
                    mv_vib_edge_tmp(search_region) = 0;
                }
            }
        }

        vib_no_match_num[i - 1] = countNonZero(mv_vib_edge_tmp);
        vib_match_num[i - 1] = vib_all_num[i - 1] - vib_no_match_num[i - 1];

        // Progress indicator
        if (i % 100 == 0 || i <= 10) {
            cout << "Processed " << i << "/" << IMG_NUM << " images" << endl;
            cout << "  Grey pixels: " << grey_num[i - 1]
                 << ", Vib matches in gray: " << vib_match_num_in_gray[i - 1]
                 << ", Total vib pixels: " << vib_all_num[i - 1] << endl;
        }
    }

    void plotResults() {
        // Calculate match ratios
        vector<double> match_ratios;
        for (int i = 0; i < IMG_NUM; i++) {
            if (grey_num[i] > 0) {
                match_ratios.push_back(static_cast<double>(vib_match_num_in_gray[i]) / grey_num[i]);
            } else {
                match_ratios.push_back(0.0);
            }
        }

        // Create a plot using OpenCV
        int plot_width = 1200;
        int plot_height = 600;
        Mat plot_img = Mat::ones(plot_height, plot_width, CV_8UC3) * 255;

        if (!match_ratios.empty()) {
            double max_ratio = *max_element(match_ratios.begin(), match_ratios.end());
            double min_ratio = *min_element(match_ratios.begin(), match_ratios.end());

            cout << "Match ratio range: " << min_ratio << " to " << max_ratio << endl;

            if (max_ratio > 0) {
                // Draw axes
                line(plot_img, Point(50, plot_height - 50), Point(plot_width - 50, plot_height - 50), Scalar(0, 0, 0),
                     2);
                line(plot_img, Point(50, plot_height - 50), Point(50, 50), Scalar(0, 0, 0), 2);

                // Plot data
                for (size_t i = 1; i < match_ratios.size(); i++) {
                    Point pt1(50 + static_cast<int>((i - 1) * (plot_width - 100) / match_ratios.size()),
                              plot_height - 50 -
                              static_cast<int>(match_ratios[i - 1] / max_ratio * (plot_height - 100)));
                    Point pt2(50 + static_cast<int>(i * (plot_width - 100) / match_ratios.size()),
                              plot_height - 50 - static_cast<int>(match_ratios[i] / max_ratio * (plot_height - 100)));
                    line(plot_img, pt1, pt2, Scalar(255, 0, 0), 2);
                }

                // Add title and labels
                putText(plot_img, "Match Ratios (vib_match_num_in_gray / grey_num)",
                        Point(plot_width / 2 - 200, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
                putText(plot_img, "Index", Point(plot_width / 2 - 20, plot_height - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
                putText(plot_img, "Ratio", Point(10, plot_height / 2),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
            }
        }

        imshow("Match Ratios", plot_img);
        waitKey(0);
    }

    void run() {
        cout << "Starting image processing..." << endl;

        // Process gray image (first loop in MATLAB)
        Mat gray_img = processGrayImage(1);
        cout << "Gray image processed successfully" << endl;

        // Process all image pairs (second loop in MATLAB)
        for (int i = 500; i <= IMG_NUM; i++) {
            processImagePair(i, gray_img);
        }

        // Plot results
        plotResults();

        cout << "Processing complete!" << endl;

        // Save results to file
        ofstream results_file("evaluation_results.csv");
        results_file << "Index,GreyNum,VibMatchInGray,VibNoMatch,VibMatch,VibAll,Ratio\n";
        for (int i = 0; i < IMG_NUM; i++) {
            double ratio = (grey_num[i] > 0) ?
                           static_cast<double>(vib_match_num_in_gray[i]) / grey_num[i] : 0.0;
            results_file << i + 1 << "," << grey_num[i] << "," << vib_match_num_in_gray[i]
                         << "," << vib_no_match_num[i] << "," << vib_match_num[i]
                         << "," << vib_all_num[i] << "," << ratio << "\n";
        }
        results_file.close();

        cout << "Results saved to evaluation_results.csv" << endl;
    }
};

int main() {
    try {
        ImageProcessor processor;
        processor.run();
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}
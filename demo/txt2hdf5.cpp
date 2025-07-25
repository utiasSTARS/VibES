// This application demonstrates how to read a text file with t, x, y, p data and convert it to HDF5 format
// using Metavision SDK HDF5EventFileWriter

#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <sstream>
#include <regex>
#include <iomanip>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/hdf5_event_file_writer.h>
#include <metavision/sdk/base/events/event_cd.h>

namespace po = boost::program_options;

struct EventData {
    Metavision::timestamp t;
    unsigned short x;
    unsigned short y;
    short p;
};

struct FileInfo {
    std::vector<EventData> events;
    int width = 0;
    int height = 0;
    std::string source_info;
    std::string extraction_date;
};

FileInfo read_txt_file(const std::filesystem::path &txt_file_path) {
    FileInfo file_info;
    std::ifstream file(txt_file_path);

    if (!file.is_open()) {
        MV_LOG_ERROR() << "Error: Could not open input file: " << txt_file_path;
        return file_info;
    }

    std::string line;
    size_t line_number = 0;
    std::regex resolution_regex(R"(#\s*Camera resolution:\s*(\d+)x(\d+))");
    std::regex source_regex(R"(#\s*DVS events extracted from\s+(.+))");
    std::regex date_regex(R"(#\s*Extraction date:\s+(.+))");
    std::smatch match;

    MV_LOG_INFO() << "Reading events from " << txt_file_path << "...";

    while (std::getline(file, line)) {
        line_number++;

        // Parse header information from comments
        if (line[0] == '#') {
            // Extract camera resolution
            if (std::regex_search(line, match, resolution_regex)) {
                file_info.width = std::stoi(match[1]);
                file_info.height = std::stoi(match[2]);
                MV_LOG_INFO() << "Found camera resolution in header: " << file_info.width << "x" << file_info.height;
            }
                // Extract source information
            else if (std::regex_search(line, match, source_regex)) {
                file_info.source_info = match[1];
                MV_LOG_INFO() << "Source: " << file_info.source_info;
            }
                // Extract extraction date
            else if (std::regex_search(line, match, date_regex)) {
                file_info.extraction_date = match[1];
                MV_LOG_INFO() << "Extraction date: " << file_info.extraction_date;
            }
            continue;
        }

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        EventData event;

        if (!(iss >> event.t >> event.x >> event.y >> event.p)) {
            MV_LOG_WARNING() << "Warning: Could not parse line " << line_number << ": " << line;
            continue;
        }

        // Validate polarity (should be 0 or 1)
        if (event.p != 0 && event.p != 1) {
            MV_LOG_WARNING() << "Warning: Invalid polarity " << event.p << " at line " << line_number
                             << ", expected 0 or 1. Setting to 0.";
            event.p = 0;
        }

        file_info.events.push_back(event);
    }

    file.close();

    if (file_info.events.empty()) {
        MV_LOG_ERROR() << "Error: No valid events found in input file";
        return file_info;
    }

    // Sort events by timestamp to ensure chronological order
    std::sort(file_info.events.begin(), file_info.events.end(), [](const EventData &a, const EventData &b) {
        return a.t < b.t;
    });

    MV_LOG_INFO() << "Successfully read " << file_info.events.size() << " events";
    MV_LOG_INFO() << "Time range: " << file_info.events.front().t << " - " << file_info.events.back().t
                  << " microseconds";
    MV_LOG_INFO() << "Duration: " << (file_info.events.back().t - file_info.events.front().t) / 1000.0
                  << " milliseconds";

    return file_info;
}

int convert_txt_to_hdf5(const std::filesystem::path &txt_file_path,
                        const std::filesystem::path &hdf5_file_path,
                        int width = 0, int height = 0) {

    if (txt_file_path == hdf5_file_path) {
        MV_LOG_ERROR() << "Error: output file is the same as input file, please specify a different path.";
        return 1;
    }

    // Create output directory if it doesn't exist
    if (!hdf5_file_path.parent_path().empty() && !std::filesystem::exists(hdf5_file_path.parent_path())) {
        std::filesystem::create_directories(hdf5_file_path.parent_path());
    }

    // Read events from text file
    auto file_info = read_txt_file(txt_file_path);
    if (file_info.events.empty()) {
        return 1;
    }

    // Use resolution from file header, or provided parameters, or auto-detect
    int final_width = width;
    int final_height = height;

    if (file_info.width > 0 && file_info.height > 0) {
        if (width > 0 && height > 0) {
            MV_LOG_INFO() << "Using command line resolution (" << width << "x" << height
                          << ") instead of file header (" << file_info.width << "x" << file_info.height << ")";
        } else {
            final_width = file_info.width;
            final_height = file_info.height;
            MV_LOG_INFO() << "Using resolution from file header: " << final_width << "x" << final_height;
        }
    } else if (width == 0 || height == 0) {
        // Auto-detect from data if not specified in file or command line
        auto max_x = std::max_element(file_info.events.begin(), file_info.events.end(),
                                      [](const EventData &a, const EventData &b) { return a.x < b.x; });
        auto max_y = std::max_element(file_info.events.begin(), file_info.events.end(),
                                      [](const EventData &a, const EventData &b) { return a.y < b.y; });

        if (final_width == 0) final_width = max_x->x + 1;
        if (final_height == 0) final_height = max_y->y + 1;

        MV_LOG_INFO() << "Auto-detected sensor resolution: " << final_width << "x" << final_height;
    }

    // Convert to Metavision EventCD format
    std::vector<Metavision::EventCD> cd_events;
    cd_events.reserve(file_info.events.size());

    Metavision::timestamp first_event_t = file_info.events.front().t;
    // Convert events
    for (const auto &event_data: file_info.events) {
        Metavision::EventCD cd_event;
        cd_event.x = event_data.x;
        cd_event.y = event_data.y;
        cd_event.p = event_data.p;
        cd_event.t = event_data.t - first_event_t;
        cd_events.push_back(cd_event);
    }

    // Create HDF5 writer
    Metavision::HDF5EventFileWriter hdf5_writer(hdf5_file_path.string());

    // Add standard Metavision metadata to match camera format
    hdf5_writer.add_metadata("version", "1.0");
    hdf5_writer.add_metadata("system_ID", "28");
    hdf5_writer.add_metadata("serial_number", "00000001");  // Default for converted files
    hdf5_writer.add_metadata("integrator_name", "Prophesee");
    hdf5_writer.add_metadata("geometry", std::to_string(final_width) + "x" + std::to_string(final_height));
    hdf5_writer.add_metadata("generation", "3.1");
    hdf5_writer.add_metadata("firmware_version", "3.2.0");

    // Add current date in the same format
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream date_ss;
    date_ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    hdf5_writer.add_metadata("Date", date_ss.str());

    // Add header information if available
    if (!file_info.source_info.empty()) {
        hdf5_writer.add_metadata("original_source", file_info.source_info);
    }
    if (!file_info.extraction_date.empty()) {
        hdf5_writer.add_metadata("original_extraction_date", file_info.extraction_date);
    }

    // Write events in batches to show progress
    const size_t batch_size = 50000;
    size_t written_events = 0;

    MV_LOG_INFO() << "Writing " << cd_events.size() << " events to HDF5 file...";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < cd_events.size(); i += batch_size) {
        size_t end_idx = std::min(i + batch_size, cd_events.size());
        size_t current_batch_size = end_idx - i;

        // Write batch of events
        hdf5_writer.add_events(cd_events.data() + i, cd_events.data() + end_idx);

        written_events += current_batch_size;

        // Progress feedback
        if (written_events % 100000 == 0 || written_events == cd_events.size()) {
            double progress = (double) written_events / cd_events.size() * 100.0;
            MV_LOG_INFO() << "Progress: " << written_events << "/" << cd_events.size()
                          << " events (" << std::fixed << std::setprecision(1) << progress << "%)";
        }
    }

    // Close the file
    hdf5_writer.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    MV_LOG_INFO() << "Successfully wrote HDF5 file: " << hdf5_file_path;
    MV_LOG_INFO() << "Conversion completed in " << duration.count() << " ms";
    MV_LOG_INFO() << "Event statistics:";
    MV_LOG_INFO() << "  - Total events: " << cd_events.size();
    MV_LOG_INFO() << "  - Resolution: " << final_width << "x" << final_height;
    MV_LOG_INFO() << "  - Duration: " << (cd_events.back().t - cd_events.front().t) / 1000.0 << " ms";
    MV_LOG_INFO() << "  - Average event rate: "
                  << (cd_events.size() * 1000000.0) / (cd_events.back().t - cd_events.front().t) << " events/sec";

    // Count polarities
    size_t positive_events = std::count_if(cd_events.begin(), cd_events.end(),
                                           [](const Metavision::EventCD &e) { return e.p == 1; });
    size_t negative_events = cd_events.size() - positive_events;

    MV_LOG_INFO() << "  - Positive events: " << positive_events << " ("
                  << (positive_events * 100.0 / cd_events.size()) << "%)";
    MV_LOG_INFO() << "  - Negative events: " << negative_events << " ("
                  << (negative_events * 100.0 / cd_events.size()) << "%)";

    return 0;
}

int main(int argc, char *argv[]) {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    int width = 0;
    int height = 0;
    bool verbose = false;

    const std::string program_desc(
            "Application to convert text file with event data (t, x, y, p) to HDF5 format using Metavision SDK.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
            ("help,h", "Produce help message.")
            ("input,i", po::value<std::filesystem::path>(&input_path)->required(),
             "Path to input text file with t, x, y, p data.")
            ("output,o", po::value<std::filesystem::path>(&output_path)->default_value(""),
             "Path to output HDF5 file. If not specified, will use input filename with .hdf5 extension.")
            ("width,w", po::value<int>(&width)->default_value(0),
             "Sensor width. If 0, will be auto-detected from data.")
            ("height,h", po::value<int>(&height)->default_value(0),
             "Sensor height. If 0, will be auto-detected from data.");
    // clang-format on

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        MV_LOG_INFO() << "\nInput file format:";
        MV_LOG_INFO() << "Each line should contain: timestamp x_coordinate y_coordinate polarity";
        MV_LOG_INFO() << "Example:";
        MV_LOG_INFO() << "1000 100 200 1";
        MV_LOG_INFO() << "1001 101 201 0";
        MV_LOG_INFO() << "1002 102 202 1";
        MV_LOG_INFO() << "\nWhere:";
        MV_LOG_INFO() << "- timestamp: microseconds (uint64)";
        MV_LOG_INFO() << "- x_coordinate: pixel x position (uint16)";
        MV_LOG_INFO() << "- y_coordinate: pixel y position (uint16)";
        MV_LOG_INFO() << "- polarity: 0 (negative) or 1 (positive)";
        return 0;
    }

    try {
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error: " << e.what();
        return 1;
    }

    // Check if input file exists
    if (!std::filesystem::exists(input_path)) {
        MV_LOG_ERROR() << "Error: Input file does not exist: " << input_path;
        return 1;
    }

    // Generate output path if not specified
    if (output_path.empty()) {
        output_path = input_path;
        output_path.replace_extension(".hdf5");
    }

    MV_LOG_INFO() << "Input file: " << input_path;
    MV_LOG_INFO() << "Output file: " << output_path;
    if (width > 0 && height > 0) {
        MV_LOG_INFO() << "Sensor resolution: " << width << "x" << height;
    } else {
        MV_LOG_INFO() << "Sensor resolution: auto-detect from data";
    }

    return convert_txt_to_hdf5(input_path, output_path, width, height);
}
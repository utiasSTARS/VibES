#!/bin/bash

# Check if results path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <results_path> [num_iterations]"
    echo "Example: $0 /home/user/project/results 3"
    exit 1
fi

# Configuration
RESULTS_PATH="$1"
NUM_ITERATIONS="${2:-2}"  # Default to 2 if not provided
OUTPUT_FILE="${RESULTS_PATH}/profiler_stats.csv"

# Dataset configurations
declare -A datasets=(
    ["logo_harmeda"]="-c /home/viciopoli/datasets/event_harmeda/intrinsics.json -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/logo_harmeda.hdf5 --tracker-x 557 --tracker-y 242"
    ["amiev_r"]="-i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/amiev_r.hdf5 --tracker-x 537 --tracker-y 186"
    ["checkerpattern_harmeda"]="-c /home/viciopoli/datasets/event_harmeda/intrinsics.json -i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/checkerpattern_harmeda.hdf5 --tracker-x 636 --tracker-y 302"
    ["harmeda_10hz_sm_texture"]="-i /home/viciopoli/datasets/event_harmeda/harmeda_dataset/harmeda_10hz_sm_texture.hdf5 --tracker-x 300 --tracker-y 195"
)

function extract_profiler_data {
    local file=$1
    local dataset=$2
    local iteration=$3

    echo "    DEBUG: Checking file: $file"

    if [[ -f "$file" ]]; then
        echo "    DEBUG: File exists, size: $(wc -l < "$file") lines"

        # Show first few lines of the file for debugging
        echo "    DEBUG: First 10 lines of output:"
        head -10 "$file" | sed 's/^/      /'

        # Show all lines containing "PROFILE" for debugging
        echo "    DEBUG: All PROFILE lines:"
        grep -i "profile" "$file" | sed 's/^/      /'

        # Extract profiler values with more flexible patterns
        local event_callback=$(grep -i "\[PROFILE\].*Event_Callback" "$file" | sed 's/.*Event_Callback.*: \([0-9]*\).*/\1/')
        local nufft_compute=$(grep -i "\[PROFILE\].*NUFFTHelixEstimator_compute" "$file" | sed 's/.*NUFFTHelixEstimator_compute.*: \([0-9]*\).*/\1/')

        # Extract new tracker metrics
        local tracker_get_rel_estimate=$(grep -i "\[PROFILE\].*tracker_getRelEstimate" "$file" | sed 's/.*tracker_getRelEstimate.*: \([0-9]*\).*/\1/')
        local tracker_feed=$(grep -i "\[PROFILE\].*tracker_feed" "$file" | sed 's/.*tracker_feed.*: \([0-9]*\).*/\1/')

        echo "    DEBUG: Extracted event_callback: '$event_callback'"
        echo "    DEBUG: Extracted nufft_compute: '$nufft_compute'"
        echo "    DEBUG: Extracted tracker_get_rel_estimate: '$tracker_get_rel_estimate'"
        echo "    DEBUG: Extracted tracker_feed: '$tracker_feed'"

        # Check if at least some values were found
        if [[ -n "$event_callback" || -n "$nufft_compute" || -n "$tracker_get_rel_estimate" || -n "$tracker_feed" ]]; then
            # Use N/A for missing values
            event_callback=${event_callback:-"N/A"}
            nufft_compute=${nufft_compute:-"N/A"}
            tracker_get_rel_estimate=${tracker_get_rel_estimate:-"N/A"}
            tracker_feed=${tracker_feed:-"N/A"}

            echo "$dataset,$iteration,$event_callback,$nufft_compute,$tracker_get_rel_estimate,$tracker_feed" >> "$OUTPUT_FILE"
            echo "    ✓ Iteration $iteration: Event_Callback=$event_callback ns/ev, NUFFT=$nufft_compute ns, TrackerGetRel=$tracker_get_rel_estimate ns, TrackerFeed=$tracker_feed ns"
        else
            echo "    ✗ Iteration $iteration: Could not extract any profiler data"
            echo "    DEBUG: Looking for alternative patterns..."

            # Try alternative patterns
            echo "    DEBUG: All lines with numbers and 'ns':"
            grep -E "[0-9]+.*ns" "$file" | sed 's/^/      /'

            # Try broader tracker patterns
            echo "    DEBUG: Looking for tracker patterns:"
            grep -i "tracker" "$file" | sed 's/^/      /'

            echo "$dataset,$iteration,N/A,N/A,N/A,N/A" >> "$OUTPUT_FILE"
        fi
    else
        echo "    ✗ Iteration $iteration: Output file not found"
        echo "$dataset,$iteration,N/A,N/A,N/A,N/A" >> "$OUTPUT_FILE"
    fi
}

function calculate_statistics {
    local dataset=$1
    local column=$2
    local metric_name=$3

    local stats=$(awk -F, -v dataset="$dataset" -v col="$column" '
        $1 == dataset && $col != "N/A" && $col != "" {
            values[NR] = $col;
            sum += $col;
            count++
        }
        END {
            if (count > 0) {
                mean = sum/count;
                # Calculate standard deviation
                for (i in values) {
                    sumsq += (values[i] - mean)^2
                }
                std = sqrt(sumsq/count);
                printf "  %s: Mean=%.2f ns, Std=%.2f ns (n=%d)\n", metric_name, mean, std, count;
            } else {
                printf "  %s: No valid data\n", metric_name;
            }
        }' "$OUTPUT_FILE")

    echo "$stats"
}

function run_compensation_study {
    echo "Running compensation study with $NUM_ITERATIONS iterations per dataset"
    echo "====================================================================="

    # Create output directories
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    mkdir -p "${RESULTS_PATH}/temp"

    # Create CSV header with new columns
    echo "Dataset,Iteration,Event_Callback,NUFFTHelixEstimator_compute_thr,tracker_getRelEstimate,tracker_feed" > "$OUTPUT_FILE"

    # Run each dataset multiple times
    for dataset in "${!datasets[@]}"; do
        echo ""
        echo "Processing dataset: $dataset"
        echo "--------------------------------"

        for ((i=1; i<=NUM_ITERATIONS; i++)); do
            echo "  Running iteration $i/$NUM_ITERATIONS..."

            # Create temporary output file for this iteration
            temp_output="${RESULTS_PATH}/temp/${dataset}_iter_${i}.txt"

            # Check if the compensation binary exists
            if [[ ! -f "./cmake-build-release/compensation" ]]; then
                echo "    ✗ Error: compensation binary not found at ./cmake-build-release/compensation"
                continue
            fi

            # Run compensation with current dataset parameters
            echo "    DEBUG: Running command: ./cmake-build-release/compensation ${datasets[$dataset]}"
            ./cmake-build-release/compensation ${datasets[$dataset]} > "$temp_output" 2>&1

            # Check the exit code
            local exit_code=$?
            echo "    DEBUG: Command exit code: $exit_code"

            # Extract and store profiler data
            extract_profiler_data "$temp_output" "$dataset" "$i"
        done

        echo "  ✓ Completed $NUM_ITERATIONS iterations for $dataset"
    done

    echo ""
    echo "Study completed! Results saved to: $OUTPUT_FILE"

    # Show summary statistics
    echo ""
    echo "Summary Statistics:"
    echo "==================="

    for dataset in "${!datasets[@]}"; do
        echo ""
        echo "Dataset: $dataset"
        echo "----------------"

        # Calculate statistics for each metric
        calculate_statistics "$dataset" 3 "Event_Callback (ns/ev)"
        calculate_statistics "$dataset" 4 "NUFFT_compute (ns)"
        calculate_statistics "$dataset" 5 "tracker_getRelEstimate (ns)"
        calculate_statistics "$dataset" 6 "tracker_feed (ns)"
    done

    echo ""
    echo "Raw data available in: $OUTPUT_FILE"

    # Show a sample of the data
    echo ""
    echo "Sample data (first 10 rows):"
    echo "============================"
    head -10 "$OUTPUT_FILE" | column -t -s ','
}

function generate_summary_report {
    echo ""
    echo "Generating summary report..."
    echo "============================"

    local summary_file="${RESULTS_PATH}/profiler_summary.txt"

    {
        echo "Profiler Performance Summary"
        echo "Generated on: $(date)"
        echo "============================="
        echo ""

        # Count total valid measurements per metric
        echo "Data Completeness:"
        echo "-----------------"

        for metric_col in 3 4 5 6; do
            case $metric_col in
                3) metric_name="Event_Callback" ;;
                4) metric_name="NUFFT_compute" ;;
                5) metric_name="tracker_getRelEstimate" ;;
                6) metric_name="tracker_feed" ;;
            esac

            local count=$(awk -F, -v col="$metric_col" 'NR > 1 && $col != "N/A" && $col != "" { count++ } END { print count+0 }' "$OUTPUT_FILE")
            local total=$(awk 'NR > 1 { count++ } END { print count+0 }' "$OUTPUT_FILE")
            local percentage=$(awk -v c="$count" -v t="$total" 'BEGIN { if(t>0) printf "%.1f", (c/t)*100; else print "0" }')

            echo "$metric_name: $count/$total measurements ($percentage%)"
        done

        echo ""
        echo "Per-Dataset Statistics:"
        echo "----------------------"

        for dataset in "${!datasets[@]}"; do
            echo ""
            echo "Dataset: $dataset"
            calculate_statistics "$dataset" 3 "Event_Callback (ns/ev)"
            calculate_statistics "$dataset" 4 "NUFFT_compute (ns)"
            calculate_statistics "$dataset" 5 "tracker_getRelEstimate (ns)"
            calculate_statistics "$dataset" 6 "tracker_feed (ns)"
        done

    } > "$summary_file"

    echo "Summary report saved to: $summary_file"
}

# Main execution
if [ $# -eq 0 ]; then
    echo "Error: Please provide the results path"
    echo "Usage: $0 <results_path> [num_iterations]"
    exit 1
fi

echo "Compensation Performance Study (Enhanced)"
echo "========================================"
echo "Results path: $RESULTS_PATH"
echo "Iterations per dataset: $NUM_ITERATIONS"
echo "Output file: $OUTPUT_FILE"
echo "Tracking metrics: Event_Callback, NUFFT_compute, tracker_getRelEstimate, tracker_feed"
echo ""

run_compensation_study
generate_summary_report

echo ""
echo "Done!"
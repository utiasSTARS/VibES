//
// Created by viciopoli on 27/06/25.
//

#ifndef PROJECT_FLOW_HPP
#define PROJECT_FLOW_HPP

#include "SliceNormalFlowEstimator.h"


class FlowCentroid {
public:
    FlowCentroid(
            const std::string &model_path,
            int max_num_points,
            int W,
            int H,
            int D,
            int pxl_radius
    ) : flow_estimator(model_path.c_str(), max_num_points, W, H, D, pxl_radius) {}

    void estimateFlow(
            float *events_txy,
            int *target_indices,
            float dt,
            int num_targets,
            const float *pred_flow
    ) {
        int size = 0;
        pred_flow = flow_estimator.predict_flows(events_txy, size, target_indices, num_targets, events_txy[0],
                                                 dt / 2000.f);
    }

    void estimate_centroid() {
        // This function can be implemented to estimate the centroid based on the flow_estimator
        // For now, it is a placeholder
    }

private:
    SliceNormalFlowEstimator flow_estimator;
};

#endif //PROJECT_FLOW_HPP

//
// Created by viciopoli on 20/12/24.
//

#ifndef PROJECT_METRIC_H
#define PROJECT_METRIC_H


class EvalMetric {
    // Optimal Dataset Scale metric
public:

    EvalMetric() = delete;

    EvalMetric(double threshold) : threshold(threshold) {}





    double f1_score() const {
        return 2 * precision * recall / (precision + recall);
    }

private:
    double precision = 0.0, recall = 0.0;
    const double threshold = 0.0;
};

#endif //PROJECT_METRIC_H

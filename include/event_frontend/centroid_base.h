//
// Created by viciopoli on 29/06/25.
//

#ifndef PROJECT_CENTROID_BASE_H
#define PROJECT_CENTROID_BASE_H

#include <optional>
#include <metavision/sdk/base/events/event_cd.h>

struct Centroid {
    double t;
    double x;
    double y;
};


class CentroidBase {
public:
    virtual ~CentroidBase() = default;

    /**
     * @brief Feed a new event to the centroid calculation.
     * This method processes the event and returns the centroid if available.
     * @param e The event to be processed, containing x, y coordinates and timestamp t.
     * @return An optional tuple containing the centroid (x, y, t) if available.
     */
    virtual std::optional<Centroid> feed(const Metavision::EventCD &event) = 0;
};

#endif //PROJECT_CENTROID_BASE_H

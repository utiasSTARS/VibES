//
// Created by viciopoli on 26/07/25.
//

#ifndef PROJECT_SWITCHER_HPP
#define PROJECT_SWITCHER_HPP

/**
 * Peak and Drop Detection Class
 */
struct Detection {
    enum Type { PEAK, DROP };
    Type type;
    Metavision::timestamp timestamp;
    double rate;
    double baseline;
    double intensity;

    Detection(Type t, Metavision::timestamp ts, double r, double b)
            : type(t), timestamp(ts), rate(r), baseline(b),
              intensity(t == PEAK ? r/b : b/r) {}
};

#endif //PROJECT_SWITCHER_HPP

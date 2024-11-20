//
// Created by viciopoli on 20/11/24.
//

#ifndef PROJECT_SINUSOID_SIM_H
#define PROJECT_SINUSOID_SIM_H

#include <cmath>

template<typename T>
class SinusoidSim {
public:
    SinusoidSim() = delete;

    SinusoidSim(const T amplitude, const T frequency, const T phase) : _amplitude(amplitude),
                                                                       _frequency(frequency),
                                                                       _phase(phase) {

    }

    T operator()(const T x) {
        return _amplitude * std::sin(_frequency * x + _phase);
    }

private:
    const T _amplitude;
    const T _frequency;
    const T _phase;
};

#endif //PROJECT_SINUSOID_SIM_H

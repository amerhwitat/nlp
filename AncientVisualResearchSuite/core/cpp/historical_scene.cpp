#include "historical_scene.hpp"
#include <cmath>
#include <stdexcept>

namespace avrs {
void HistoricalScene::setDomain(std::size_t domain, const std::array<double,16>& values) {
    if (domain >= 8) throw std::out_of_range("128D domain must be 0..7");
    for (std::size_t i=0;i<16;++i) tensor[domain*16+i] = values[i];
}

std::vector<Frame> HistoricalScene::simulate(double seconds, double step) const {
    if (seconds < 0 || step <= 0) throw std::invalid_argument("invalid simulation interval");
    std::vector<Frame> out;
    for (double t=0; t<=seconds+1e-9; t+=step) {
        Frame f; f.timeSeconds=t;
        for (const auto& c: characters) {
            double active = c.activity == "idle" || c.activity == "sleep" ? 0.0 : 0.5;
            double phase = c.heading * M_PI / 180.0 + t*0.15;
            f.characters.push_back({c.id, c.x + std::cos(phase)*active*t,
                                    c.y + std::sin(phase)*active*t, c.heading});
        }
        out.push_back(std::move(f));
    }
    return out;
}
} // namespace avrs

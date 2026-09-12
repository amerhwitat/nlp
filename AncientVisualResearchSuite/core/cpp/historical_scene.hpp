#pragma once
#include <array>
#include <string>
#include <vector>
namespace avrs {
enum class EvidenceClass { Observed, Supported, Inferred, Speculative, Visualization };
struct Evidence { std::string id, source, notes; EvidenceClass kind{EvidenceClass::Observed}; double confidence{1.0}; };
struct Character { std::string id, role; double x{}, y{}, heading{}; std::string activity; };
struct Event { std::string id, label; double startSeconds{}, durationSeconds{}; std::vector<std::string> actors; };
struct FrameCharacter { std::string id; double x{}, y{}, heading{}; };
struct Frame { double timeSeconds{}; std::vector<FrameCharacter> characters; };
class HistoricalScene {
public:
    std::string id, title;
    double latitude{}, longitude{}, elevationMeters{};
    std::array<double,128> tensor{};
    std::vector<Evidence> evidence;
    std::vector<Character> characters;
    std::vector<Event> events;
    void setDomain(std::size_t domain, const std::array<double,16>& values);
    std::vector<Frame> simulate(double seconds, double step=.25) const;
};
}

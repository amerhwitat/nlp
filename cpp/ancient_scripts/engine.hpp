#pragma once
#include <algorithm>
#include <string>
#include <vector>

namespace ancient_scripts {
struct Reading {
    std::string script, text, transliteration, unicode_text;
    double confidence{0.0};
    std::vector<std::string> alternates, damage, provenance;
};
struct TranslationCandidate {
    std::string text;
    double confidence{0.0};
    std::vector<std::string> evidence;
};
inline void rank(std::vector<TranslationCandidate>& c) {
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b){ return a.confidence > b.confidence; });
}
inline bool requires_review(const Reading& r) { return r.confidence < .85 || !r.damage.empty(); }
}

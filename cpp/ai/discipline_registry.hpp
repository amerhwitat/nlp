#pragma once
#include <array>
#include <string_view>

namespace chimera::ai {
enum class Discipline { ML, DeepLearning, ReinforcementLearning, SymbolicAI, ComputerVision, NLP };
struct DisciplineInfo {
    Discipline discipline;
    std::string_view approach;
    std::string_view data_dependency;
};
inline constexpr std::array<DisciplineInfo, 6> kDisciplines{{
    {Discipline::ML, "statistical pattern learning", "structured/tabular"},
    {Discipline::DeepLearning, "hierarchical neural networks", "text/audio/image/tensor"},
    {Discipline::ReinforcementLearning, "reward-driven policy optimization", "interaction/simulation"},
    {Discipline::SymbolicAI, "rules, logic, ontology and inference", "facts/rules/knowledge graphs"},
    {Discipline::ComputerVision, "spatial pattern and geometry extraction", "image/video/3D"},
    {Discipline::NLP, "computational linguistics and language models", "text/speech/corpora"}
}};
}

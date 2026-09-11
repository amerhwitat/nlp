#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>

namespace iso_tool {
struct Dependency { std::string name; std::vector<std::string> commands; bool required; std::string category; std::string path; std::string status; };
std::vector<Dependency> DetectDependencies();
std::string DependenciesJson(const std::vector<Dependency>&);
}

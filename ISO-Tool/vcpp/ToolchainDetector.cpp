#include "ToolchainDetector.hpp"
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <windows.h>

namespace fs = std::filesystem;
namespace iso_tool {
static std::string where(const std::string& exe) {
    char buf[32768]{}; DWORD n = SearchPathA(nullptr, exe.c_str(), nullptr, sizeof(buf), buf, nullptr);
    return n && n < sizeof(buf) ? std::string(buf, n) : std::string{};
}
static std::string home(const std::string& p) {
    fs::path x(p); return x.parent_path().filename() == "bin" ? x.parent_path().parent_path().string() : x.parent_path().string();
}
std::vector<ToolchainResult> DetectToolchains() {
    const struct S { const char* n; const char* e; const char* v; } tools[] = {
        {"GCC/MinGW","gcc.exe","MINGW_HOME"},{"G++","g++.exe","MINGW_HOME"},{"MSVC","cl.exe","MSVC_HOME"},
        {"NASM","nasm.exe","NASM_HOME"},{"MASM","ml.exe","MASM_HOME"},{"Go","go.exe","GOROOT"},
        {"Rust","rustc.exe","RUST_HOME"},{"Java","javac.exe","JAVA_HOME"},{"Python","python.exe","PYTHON_HOME"},
        {"Clang","clang.exe","LLVM_HOME"},{"CMake","cmake.exe","CMAKE_HOME"},{"Ninja","ninja.exe","NINJA_HOME"},
        {"MSBuild","MSBuild.exe","MSBUILD_HOME"},{"Git","git.exe","GIT_HOME"},{"xorriso","xorriso.exe","XORRISO_HOME"},{"Oscdimg","oscdimg.exe","OSCDIMG_HOME"}
    };
    std::vector<ToolchainResult> out;
    for (const auto& t : tools) { auto p = where(t.e); out.push_back({t.n,t.e,t.v,p,p.empty()?"not-found":"found"}); }
    return out;
}
std::string ToolchainsJson() {
    std::ostringstream o; o << "{\"platform\":\"windows\",\"tools\":["; bool first=true;
    for (const auto& t : DetectToolchains()) { if(!first)o<<','; first=false; o<<"{\"name\":\""<<t.name<<"\",\"executable\":\""<<t.executable<<"\",\"environmentVariable\":\""<<t.environment<<"\",\"status\":\""<<t.status<<"\",\"executablePath\":\""<<t.path<<"\"}"; }
    return o.str()+"]}";
}
}

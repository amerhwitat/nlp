#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace chimera::thamudic {

struct Box { int x1{}, y1{}, x2{}, y2{}; };
struct Word { std::u32string source; std::string transliteration; Box box{}; };

constexpr char32_t kFirst = 0x10A80;
constexpr char32_t kLast  = 0x10A9F;

bool isThamudic(char32_t cp);
std::vector<char32_t> codePoints(std::u8string_view utf8);
std::u8string toUtf8(std::u32string_view text);
std::u32string extractThamudic(std::u32string_view text);
std::string transliterate(std::u32string_view text, const std::unordered_map<char32_t,std::string>& map);

std::vector<Box> connectedComponents(const std::vector<std::uint8_t>& binary, int width, int height, int minArea = 30);
std::vector<std::vector<Box>> groupLines(std::vector<Box> boxes, int yTolerance = 14);
std::vector<std::vector<Box>> groupWords(const std::vector<Box>& line, int gapThreshold = 18);

} // namespace chimera::thamudic

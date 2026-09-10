#include <iostream>
#include "thamudic/thamudic.hpp"
int main(){std::string s;std::getline(std::cin,s);auto cps=chimera::thamudic::codePoints(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));std::u32string t(cps.begin(),cps.end());std::cout<<"Thamudic code points: "<<chimera::thamudic::toUtf8(chimera::thamudic::extractThamudic(t)).c_str()<<"\n";}

#include <iostream>
#include <string>
#include "thamudic/thamudic.hpp"
int main(){std::string s;std::getline(std::cin,s);auto cps=chimera::thamudic::codePoints(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));std::u32string t(cps.begin(),cps.end());auto u=chimera::thamudic::toUtf8(chimera::thamudic::extractThamudic(t));std::cout<<"Thamudic: "<<reinterpret_cast<const char*>(u.c_str())<<"\n";}

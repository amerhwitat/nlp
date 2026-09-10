#include "thamudic/thamudic.hpp"
#include <cassert>
int main(){using namespace chimera::thamudic;assert(isThamudic(0x10A80));assert(!isThamudic(U'A'));auto cps=codePoints(u8"𐪀A");assert(cps.size()==2);auto x=extractThamudic(U"𐪀A𐪁");assert(x.size()==2);return 0;}

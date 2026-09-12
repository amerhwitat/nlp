#include "historical_scene.hpp"
#include <cassert>
#include <iostream>
int main(){
  avrs::HistoricalScene s; s.id="smoke";
  s.characters.push_back({"c1","observer",0.0,0.0,0.0,"active"});
  auto frames=s.simulate(1.0,.5); assert(frames.size()==3);
  std::cout << "AVRS C++ smoke OK\n"; return 0;
}

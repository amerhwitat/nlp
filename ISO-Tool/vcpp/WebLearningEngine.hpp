#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <unordered_map>
struct WebRecord { std::string url,title,text,sha256,sourceType; int status{}; };
class WebLearningEngine {
public:
 std::vector<WebRecord> crawl(const std::string& start,int maxPages=10,int maxDepth=1);
 double rnnScore(const std::vector<double>& seq) const { double h=0,w=0; for(double x:seq){h=std::tanh(x*.1+h*.05);w+=h*.01;} return 1.0/(1.0+std::exp(-w)); }
 std::string llmAdapter(const std::string& endpoint,const std::string& prompt) const { return "endpoint="+endpoint+"; authorizationRequired=true; prompt="+prompt; }
};

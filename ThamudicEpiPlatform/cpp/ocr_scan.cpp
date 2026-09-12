#include <curl/curl.h>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc,char**argv){if(argc<2){std::cerr<<"usage: ocr_scan image [engine]\n";return 2;}std::string path=argv[1],engine=argc>2?argv[2]:"auto";CURL* c=curl_easy_init();if(!c)return 1;curl_mime* mime=curl_mime_init(c);curl_mimepart* part=curl_mime_addpart(mime);curl_mime_name(part,"file");curl_mime_filedata(part,path.c_str());std::string url="http://127.0.0.1:8010/api/ocr/scan?engine="+engine;curl_easy_setopt(c,CURLOPT_URL,url.c_str());curl_easy_setopt(c,CURLOPT_MIMEPOST,mime);curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,[](char*p,size_t s,size_t n,void*)->size_t{std::cout.write(p,s*n);return s*n;});auto r=curl_easy_perform(c);curl_mime_free(mime);curl_easy_cleanup(c);return r==CURLE_OK?0:1;}

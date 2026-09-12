#include <curl/curl.h>
#include <iostream>
int main(){CURL* c=curl_easy_init(); if(!c)return 1; curl_easy_setopt(c,CURLOPT_URL,"http://127.0.0.1:8010/api/objects"); curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,[](char* p,size_t s,size_t n,void*)->size_t{std::cout.write(p,s*n);return s*n;}); auto r=curl_easy_perform(c); curl_easy_cleanup(c); return r==CURLE_OK?0:1;}

#define UNICODE
#define _UNICODE
#include <windows.h>
#include <string>
#include <sstream>
#include "thamudic/thamudic.hpp"
#include "thamudic/old_north_arabian.hpp"

static HWND inputBox, outputBox, detailsBox;
static std::u32string Utf16ToUtf32(const std::wstring& s){std::u32string out;for(size_t i=0;i<s.size();++i){char32_t c=s[i];if(c>=0xD800&&c<=0xDBFF&&i+1<s.size()){char32_t d=s[++i];if(d>=0xDC00&&d<=0xDFFF)c=0x10000+((c-0xD800)<<10)+(d-0xDC00);}out.push_back(c);}return out;}
static std::wstring Utf32ToUtf16(const std::u32string& s){std::wstring out;for(char32_t c:s){if(c<=0xFFFF)out.push_back(static_cast<wchar_t>(c));else{c-=0x10000;out.push_back(static_cast<wchar_t>(0xD800+(c>>10)));out.push_back(static_cast<wchar_t>(0xDC00+(c&0x3FF)));}}return out;}
static void SetText(HWND h,const std::wstring& text){SetWindowTextW(h,text.c_str());}
static void ExtractText(){int len=GetWindowTextLengthW(inputBox);std::wstring input(len,L'\0');GetWindowTextW(inputBox,input.data(),len+1);auto cps=Utf16ToUtf32(input);std::u32string ona;for(char32_t cp:cps)if(chimera::thamudic::ona::isOldNorthArabian(cp))ona.push_back(cp);SetText(outputBox,ona.empty()?L"No Old North Arabian characters detected.":Utf32ToUtf16(ona));}
static void ShowRegistry(){std::wstringstream s;s<<L"Old North Arabian / Ancient North Arabian\r\nU+10A80–U+10A9F • RTL • Dadanitic encoding basis\r\n\r\n";for(const auto& c:chimera::thamudic::ona::kAlphabet)s<<Utf32ToUtf16({c.character})<<L"  U+"<<std::hex<<c.codePoint<<L"  "<<c.name.data()<<L"  / "<<c.transliteration.data()<<L"  UTF-8 "<<c.utf8Hex.data()<<L"\r\n";SetText(detailsBox,s.str());}
LRESULT CALLBACK WndProc(HWND h,UINT m,WPARAM w,LPARAM){switch(m){case WM_CREATE:CreateWindowW(L"STATIC",L"Ancient North Arabian / Thamudic NLP Workbench",WS_CHILD|WS_VISIBLE,20,15,760,30,h,nullptr,nullptr,nullptr);inputBox=CreateWindowW(L"EDIT",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_MULTILINE|ES_AUTOVSCROLL,20,55,760,80,h,(HMENU)1,nullptr,nullptr);CreateWindowW(L"BUTTON",L"Extract Old North Arabian",WS_CHILD|WS_VISIBLE,20,145,190,34,h,(HMENU)2,nullptr,nullptr);outputBox=CreateWindowW(L"EDIT",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_MULTILINE|ES_READONLY,20,190,760,55,h,(HMENU)3,nullptr,nullptr);detailsBox=CreateWindowW(L"EDIT",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_MULTILINE|ES_READONLY|WS_VSCROLL,20,260,760,300,h,(HMENU)4,nullptr,nullptr);ShowRegistry();return 0;case WM_COMMAND:if(LOWORD(w)==2)ExtractText();return 0;case WM_DESTROY:PostQuitMessage(0);return 0;}return DefWindowProcW(h,m,w,l);}
int WINAPI wWinMain(HINSTANCE hi,HINSTANCE,PWSTR,int show){const wchar_t cls[]=L"ChimeraThamudicDesktop";WNDCLASSW wc{};wc.lpfnWndProc=WndProc;wc.hInstance=hi;wc.lpszClassName=cls;wc.hCursor=LoadCursor(nullptr,IDC_ARROW);RegisterClassW(&wc);HWND h=CreateWindowExW(0,cls,L"Chimera Thamudic — Visual C++ Desktop",WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,CW_USEDEFAULT,820,640,nullptr,nullptr,hi,nullptr);if(!h)return 1;ShowWindow(h,show);UpdateWindow(h);MSG msg{};while(GetMessageW(&msg,nullptr,0,0)>0){TranslateMessage(&msg);DispatchMessageW(&msg);}return static_cast<int>(msg.wParam);}

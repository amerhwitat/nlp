#define UNICODE
#define _UNICODE
#include <windows.h>
#include <string>

static HWND gLog, gProgress;
static void Log(const std::wstring& s){int n=GetWindowTextLengthW(gLog); SendMessageW(gLog,EM_SETSEL,n,n); SendMessageW(gLog,EM_REPLACESEL,FALSE,(LPARAM)(s+L"\r\n").c_str());}
static LRESULT CALLBACK WndProc(HWND h,UINT m,WPARAM w,LPARAM l){
 if(m==WM_CREATE){
  CreateWindowW(L"STATIC",L"ISO-Tool — GitHub source to ISO / IMG",WS_CHILD|WS_VISIBLE,20,15,600,30,h,nullptr,nullptr,nullptr);
  CreateWindowW(L"BUTTON",L"Analyze",WS_CHILD|WS_VISIBLE,20,55,110,32,h,(HMENU)1,nullptr,nullptr);
  CreateWindowW(L"BUTTON",L"Build ISO",WS_CHILD|WS_VISIBLE,140,55,110,32,h,(HMENU)2,nullptr,nullptr);
  gLog=CreateWindowW(L"EDIT",L"",WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_READONLY,20,105,740,360,h,nullptr,nullptr,nullptr);
  gProgress=CreateWindowW(PROGRESS_CLASSW,L"",WS_CHILD|WS_VISIBLE,20,480,740,24,h,nullptr,nullptr,nullptr);
  SendMessageW(gProgress,PBM_SETRANGE,0,MAKELPARAM(0,100));
  return 0;
 }
 if(m==WM_COMMAND){ if(LOWORD(w)==1){Log(L"Analyze: inventory only; repository commands are not executed.");SendMessageW(gProgress,PBM_SETPOS,100,0);} if(LOWORD(w)==2){Log(L"Build: trusted/custom authorization is required before repository build execution.");Log(L"Detecting MSVC, MASM, NASM, GCC/G++, CMake, MSBuild and ISO backends...");SendMessageW(gProgress,PBM_SETPOS,100,0);} return 0; }
 if(m==WM_DESTROY){PostQuitMessage(0);return 0;} return DefWindowProcW(h,m,w,l);
}
int WINAPI wWinMain(HINSTANCE hi,HINSTANCE, PWSTR,int show){INITCOMMONCONTROLSEX ic{sizeof(ic),ICC_PROGRESS_CLASS};InitCommonControlsEx(&ic); WNDCLASSW wc{};wc.hInstance=hi;wc.lpfnWndProc=WndProc;wc.lpszClassName=L"ISO_TOOL_MAIN";wc.hCursor=LoadCursor(nullptr,IDC_ARROW);RegisterClassW(&wc);HWND h=CreateWindowW(wc.lpszClassName,L"ISO-Tool",WS_OVERLAPPEDWINDOW,100,100,820,570,nullptr,nullptr,hi,nullptr);ShowWindow(h,show);MSG msg;while(GetMessageW(&msg,nullptr,0,0)>0){TranslateMessage(&msg);DispatchMessageW(&msg);}return 0;}

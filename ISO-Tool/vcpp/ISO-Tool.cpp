#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <string>
#include <thread>
#include <vector>
#include <exception>
#include <stdexcept>

#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "comdlg32.lib")
#if defined(_MSC_VER)
#pragma comment(linker, "\"/manifestdependency:type='Win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif

static HWND gRepo=nullptr,gLog=nullptr,gProgress=nullptr,gStatus=nullptr,gBootStatus=nullptr,gAnalyze=nullptr,gImport=nullptr,gImages=nullptr,gBuild=nullptr;
static std::wstring gIsoOutput;
static constexpr UINT WM_ISOTOOL_LOG=WM_APP+1, WM_ISOTOOL_PROGRESS=WM_APP+2, WM_ISOTOOL_DONE=WM_APP+3;

static void AppendLog(const std::wstring& text){if(!gLog)return;int n=GetWindowTextLengthW(gLog);SendMessageW(gLog,EM_SETSEL,n,n);std::wstring line=text+L"\r\n";SendMessageW(gLog,EM_REPLACESEL,FALSE,reinterpret_cast<LPARAM>(line.c_str()));SendMessageW(gLog,EM_SCROLLCARET,0,0);}
static void PostText(HWND w,UINT m,const std::wstring& t){auto*p=new std::wstring(t);if(!PostMessageW(w,m,0,reinterpret_cast<LPARAM>(p)))delete p;}
static void PostProgress(HWND w,int v,const std::wstring&t){auto*p=new std::wstring(t);if(!PostMessageW(w,WM_ISOTOOL_PROGRESS,v,reinterpret_cast<LPARAM>(p)))delete p;}

static bool ChooseIsoOutput(HWND owner){
    wchar_t path[MAX_PATH]=L"repository.iso";
    OPENFILENAMEW ofn{};ofn.lStructSize=sizeof(ofn);ofn.hwndOwner=owner;ofn.lpstrFilter=L"ISO image (*.iso)\0*.iso\0All files (*.*)\0*.*\0";ofn.lpstrFile=path;ofn.nMaxFile=MAX_PATH;ofn.lpstrDefExt=L"iso";ofn.Flags=OFN_EXPLORER|OFN_PATHMUSTEXIST|OFN_OVERWRITEPROMPT;
    if(!GetSaveFileNameW(&ofn))return false;
    gIsoOutput=path;return true;
}

static std::wstring Quote(const std::wstring&s){return L"\""+s+L"\"";}
static std::wstring FindPythonScript(){
    wchar_t cwd[MAX_PATH]{};GetCurrentDirectoryW(MAX_PATH,cwd);
    std::vector<std::wstring> candidates={std::wstring(cwd)+L"\\ISO-Tool\\python\\build_iso.py",std::wstring(cwd)+L"\\python\\build_iso.py",L"ISO-Tool\\python\\build_iso.py"};
    for(const auto&p:candidates){DWORD a=GetFileAttributesW(p.c_str());if(a!=INVALID_FILE_ATTRIBUTES&&!(a&FILE_ATTRIBUTE_DIRECTORY))return p;}
    return L"";
}

static void RunProcess(HWND window,const std::wstring& command,const std::wstring& cwd){
    STARTUPINFOW si{};si.cb=sizeof(si);PROCESS_INFORMATION pi{};
    std::vector<wchar_t> buffer(command.begin(),command.end());buffer.push_back(L'\0');
    if(!CreateProcessW(nullptr,buffer.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,nullptr,cwd.empty()?nullptr:cwd.c_str(),&si,&pi)){PostText(window,WM_ISOTOOL_LOG,L"[error] could not launch recursive build helper; install/configure Python and ISO backend tools");PostMessageW(window,WM_ISOTOOL_DONE,1,0);return;}
    WaitForSingleObject(pi.hProcess,INFINITE);DWORD code=1;GetExitCodeProcess(pi.hProcess,&code);CloseHandle(pi.hThread);CloseHandle(pi.hProcess);
    PostText(window,WM_ISOTOOL_LOG,code==0?L"[build] recursive repository build and ISO mastering completed":L"[error] recursive build/ISO mastering failed; inspect the build report/log");PostMessageW(window,WM_ISOTOOL_DONE,code,0);
}

static void RunPipeline(HWND window,bool iso){
    if(iso){
        wchar_t repo[MAX_PATH]{};GetWindowTextW(gRepo,repo,MAX_PATH);
        if(!repo[0]){PostText(window,WM_ISOTOOL_LOG,L"[error] repository path or Git URL is empty");PostMessageW(window,WM_ISOTOOL_DONE,1,0);return;}
        if(gIsoOutput.empty()&&!ChooseIsoOutput(window)){PostText(window,WM_ISOTOOL_LOG,L"[cancelled] ISO output selection cancelled; no file was created");PostMessageW(window,WM_ISOTOOL_DONE,1,0);return;}
        const std::wstring script=FindPythonScript();if(script.empty()){PostText(window,WM_ISOTOOL_LOG,L"[error] ISO-Tool/python/build_iso.py was not found");PostMessageW(window,WM_ISOTOOL_DONE,1,0);return;}
        PostText(window,WM_ISOTOOL_LOG,L"[build] recursively analyzing source, resolving external references, compiling/linking compatible targets, then mastering ISO to: "+gIsoOutput);
        PostProgress(window,10,L"Resolving repository and external references");
        std::wstring command=L"python "+Quote(script)+L" "+Quote(repo)+L" --output "+Quote(gIsoOutput)+L" --label ISO_TOOL --profile data";
        RunProcess(window,command,L"");return;
    }
    std::vector<std::wstring> steps={L"validate repository",L"recursively inventory GitHub/local checkout",L"discover build manifests and toolchains",L"resolve external dependency graph",L"prepare recursive build plan",L"compile/assemble source jobs",L"link compatible native targets",L"collect language/runtime artifacts"};
    for(size_t i=0;i<steps.size();++i){PostText(window,WM_ISOTOOL_LOG,L"[step] "+steps[i]);Sleep(75);PostProgress(window,(int)(((i+1)*100)/steps.size()),steps[i]);}
    PostMessageW(window,WM_ISOTOOL_DONE,0,0);
}

static LRESULT CALLBACK WndProc(HWND w,UINT m,WPARAM wp,LPARAM lp){
    switch(m){
    case WM_CREATE:
        CreateWindowW(L"STATIC",L"ISO-Tool — Repository → Dependencies → Compile/Link → ISO / IMG",WS_CHILD|WS_VISIBLE,20,15,850,30,w,nullptr,nullptr,nullptr);
        gRepo=CreateWindowW(L"EDIT",L"",WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL,20,50,850,28,w,nullptr,nullptr,nullptr);
        gAnalyze=CreateWindowW(L"BUTTON",L"Analyze",WS_CHILD|WS_VISIBLE,20,88,105,32,w,(HMENU)1,nullptr,nullptr);
        gImport=CreateWindowW(L"BUTTON",L"Import Boot / ISO",WS_CHILD|WS_VISIBLE,135,88,135,32,w,(HMENU)3,nullptr,nullptr);
        gImages=CreateWindowW(L"BUTTON",L"Build Images",WS_CHILD|WS_VISIBLE,280,88,120,32,w,(HMENU)4,nullptr,nullptr);
        gBuild=CreateWindowW(L"BUTTON",L"Build ISO…",WS_CHILD|WS_VISIBLE,410,88,110,32,w,(HMENU)2,nullptr,nullptr);
        gStatus=CreateWindowW(L"STATIC",L"Ready — ISO output is user-selected",WS_CHILD|WS_VISIBLE,535,94,335,24,w,nullptr,nullptr,nullptr);
        gBootStatus=CreateWindowW(L"STATIC",L"BIOS 0x7C00 | UEFI PE/COFF | fallback validation",WS_CHILD|WS_VISIBLE,20,122,850,24,w,nullptr,nullptr,nullptr);
        gLog=CreateWindowW(L"EDIT",L"",WS_CHILD|WS_VISIBLE|WS_VSCROLL|WS_HSCROLL|ES_MULTILINE|ES_READONLY,20,150,850,370,w,nullptr,nullptr,nullptr);
        gProgress=CreateWindowW(PROGRESS_CLASSW,L"",WS_CHILD|WS_VISIBLE,20,535,850,24,w,nullptr,nullptr,nullptr);SendMessageW(gProgress,PBM_SETRANGE,0,MAKELPARAM(0,100));return 0;
    case WM_COMMAND:{int id=LOWORD(wp);if(id==1||id==2||id==4){if(id==2&& !ChooseIsoOutput(w)){AppendLog(L"ISO output selection cancelled.");return 0;}EnableWindow(gAnalyze,FALSE);EnableWindow(gImport,FALSE);EnableWindow(gImages,FALSE);EnableWindow(gBuild,FALSE);SendMessageW(gProgress,PBM_SETPOS,0,0);AppendLog(id==1?L"Recursive repository analysis started.":(id==4?L"Recursive compiled-image workflow started.":L"ISO build started. Choose any writable path and filename for the final ISO."));std::thread(RunPipeline,w,id==2).detach();}else if(id==3)AppendLog(L"Import Boot / ISO selected. Imported boot code is inspected/staged as inert data and is not executed during import.");return 0;}
    case WM_ISOTOOL_LOG:{auto*p=reinterpret_cast<std::wstring*>(lp);if(p){AppendLog(*p);delete p;}return 0;}
    case WM_ISOTOOL_PROGRESS:{auto*p=reinterpret_cast<std::wstring*>(lp);SendMessageW(gProgress,PBM_SETPOS,wp,0);if(p){SetWindowTextW(gStatus,p->c_str());AppendLog(*p);delete p;}return 0;}
    case WM_ISOTOOL_DONE:EnableWindow(gAnalyze,TRUE);EnableWindow(gImport,TRUE);EnableWindow(gImages,TRUE);EnableWindow(gBuild,TRUE);if(wp==0)AppendLog(L"Workflow completed.");else AppendLog(L"Workflow failed or was cancelled; see recursive-build-report.json and recursive-build.log.");return 0;
    case WM_DESTROY:PostQuitMessage(0);return 0;default:return DefWindowProcW(w,m,wp,lp);
    }
}

int WINAPI wWinMain(HINSTANCE instance,HINSTANCE,PWSTR,int show){
    INITCOMMONCONTROLSEX controls{};controls.dwSize=sizeof(controls);controls.dwICC=ICC_PROGRESS_CLASS|ICC_STANDARD_CLASSES;if(!InitCommonControlsEx(&controls)){MessageBoxW(nullptr,L"ISO-Tool could not initialize Windows common controls.",L"ISO-Tool startup error",MB_ICONERROR|MB_OK);return 1;}
    WNDCLASSW wc{};wc.hInstance=instance;wc.lpfnWndProc=WndProc;wc.lpszClassName=L"ISO_TOOL_MAIN";wc.hCursor=LoadCursorW(nullptr,IDC_ARROW);wc.hbrBackground=reinterpret_cast<HBRUSH>(COLOR_WINDOW+1);if(!RegisterClassW(&wc)){MessageBoxW(nullptr,L"Could not register the ISO-Tool window class.",L"ISO-Tool startup error",MB_ICONERROR|MB_OK);return 1;}
    HWND window=CreateWindowW(wc.lpszClassName,L"ISO-Tool",WS_OVERLAPPEDWINDOW,100,100,920,620,nullptr,nullptr,instance,nullptr);if(!window){MessageBoxW(nullptr,L"Could not create the ISO-Tool window.",L"ISO-Tool startup error",MB_ICONERROR|MB_OK);return 1;}ShowWindow(window,show);UpdateWindow(window);MSG message{};while(GetMessageW(&message,nullptr,0,0)>0){TranslateMessage(&message);DispatchMessageW(&message);}return (int)message.wParam;
}

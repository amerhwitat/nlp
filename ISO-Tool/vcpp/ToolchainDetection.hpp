#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <cstdio>

namespace iso_tool {
struct ToolInfo { std::wstring kind, name, path, version, source; };

inline std::wstring Trim(std::wstring s) {
    const auto first=s.find_first_not_of(L" \t\r\n");
    if(first==std::wstring::npos) return L"";
    const auto last=s.find_last_not_of(L" \t\r\n");
    return s.substr(first,last-first+1);
}
inline std::wstring Basename(const std::wstring& p) {
    const auto pos=p.find_last_of(L"\\/"); return pos==std::wstring::npos?p:p.substr(pos+1);
}
inline bool FileExists(const std::wstring& p) {
    const DWORD a=GetFileAttributesW(p.c_str()); return a!=INVALID_FILE_ATTRIBUTES && !(a&FILE_ATTRIBUTE_DIRECTORY);
}
inline std::wstring SearchExecutable(const std::wstring& name) {
    wchar_t out[32768]{};
    DWORD n=SearchPathW(nullptr,name.c_str(),nullptr,static_cast<DWORD>(std::size(out)),out,nullptr);
    return n && n<std::size(out) ? std::wstring(out,n) : L"";
}
inline std::wstring RunVersion(const std::wstring& exe) {
    SECURITY_ATTRIBUTES sa{}; sa.nLength=sizeof(sa); sa.bInheritHandle=TRUE;
    HANDLE readPipe=nullptr, writePipe=nullptr;
    if(!CreatePipe(&readPipe,&writePipe,&sa,0)) return L"unknown version";
    SetHandleInformation(readPipe,HANDLE_FLAG_INHERIT,0);
    std::wstring cmd=L"\""+exe+L"\" --version";
    std::vector<wchar_t> buf(cmd.begin(),cmd.end()); buf.push_back(L'\0');
    STARTUPINFOW si{}; si.cb=sizeof(si); si.dwFlags=STARTF_USESTDHANDLES; si.hStdOutput=writePipe; si.hStdError=writePipe;
    PROCESS_INFORMATION pi{};
    bool ok=CreateProcessW(nullptr,buf.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW,nullptr,nullptr,&si,&pi)!=FALSE;
    CloseHandle(writePipe);
    if(!ok){CloseHandle(readPipe);return L"unknown version";}
    WaitForSingleObject(pi.hProcess,1500);
    DWORD code=STILL_ACTIVE; GetExitCodeProcess(pi.hProcess,&code);
    if(code==STILL_ACTIVE){TerminateProcess(pi.hProcess,1); WaitForSingleObject(pi.hProcess,250);}
    std::string bytes; char chunk[512]; DWORD got=0;
    while(ReadFile(readPipe,chunk,sizeof(chunk),&got,nullptr)&&got) bytes.append(chunk,chunk+got);
    CloseHandle(readPipe); CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    std::wstring wide(bytes.begin(),bytes.end()); wide=Trim(wide);
    const auto nl=wide.find_first_of(L"\r\n"); if(nl!=std::wstring::npos) wide.resize(nl);
    return wide.empty()?L"unknown version":wide;
}
inline void AddCandidate(std::vector<ToolInfo>& out,const std::wstring& kind,const std::wstring& name,const std::wstring& path,const std::wstring& source) {
    if(path.empty()) return;
    for(const auto& x:out) if(_wcsicmp(x.path.c_str(),path.c_str())==0) return;
    out.push_back({kind,name,path,RunVersion(path),source});
}
inline void ScanPath(std::vector<ToolInfo>& out,const std::wstring& kind,const std::vector<std::wstring>& names) {
    for(const auto& n:names) { auto p=SearchExecutable(n); if(!p.empty()) AddCandidate(out,kind,n,p,L"PATH"); }
}
inline void ScanVisualStudioRegistry(std::vector<ToolInfo>& compilers,std::vector<ToolInfo>& linkers,std::vector<ToolInfo>& assemblers) {
    HKEY key=nullptr;
    if(RegOpenKeyExW(HKEY_LOCAL_MACHINE,L"SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VS7",0,KEY_READ|KEY_WOW64_64KEY,&key)!=ERROR_SUCCESS) return;
    for(DWORD i=0;;++i){
        wchar_t value[64]{}, install[32768]{}; DWORD valueLen=static_cast<DWORD>(std::size(value)), dataLen=sizeof(install), type=0;
        if(RegEnumValueW(key,i,value,&valueLen,nullptr,&type,reinterpret_cast<LPBYTE>(install),&dataLen)!=ERROR_SUCCESS) break;
        if(type!=REG_SZ) continue;
        std::wstring root(install); std::error_code ec;
        std::filesystem::path vc=root; vc/=L"VC\\Tools\\MSVC";
        if(!std::filesystem::exists(vc,ec)) continue;
        for(auto it=std::filesystem::directory_iterator(vc,ec); !ec && it!=std::filesystem::directory_iterator(); it.increment(ec)) {
            if(!it->is_directory(ec)) continue;
            const auto bin=it->path()/L"bin";
            const auto x64=bin/L"Hostx64"/L"x64";
            AddCandidate(compilers,L"C/C++ compiler",L"MSVC cl.exe",(x64/L"cl.exe").wstring(),L"Visual Studio registry");
            AddCandidate(linkers,L"Linker",L"MSVC link.exe",(x64/L"link.exe").wstring(),L"Visual Studio registry");
            AddCandidate(assemblers,L"Assembler",L"MSVC ml64.exe",(x64/L"ml64.exe").wstring(),L"Visual Studio registry");
            AddCandidate(assemblers,L"Assembler",L"MSVC ml.exe",(x64/L"ml.exe").wstring(),L"Visual Studio registry");
        }
    }
    RegCloseKey(key);
}
inline std::vector<ToolInfo> DetectCompilers() {
    std::vector<ToolInfo> r; ScanPath(r,L"C/C++ compiler",{L"cl.exe",L"g++.exe",L"gcc.exe",L"clang++.exe",L"clang.exe",L"zig.exe"});
    return r;
}
inline std::vector<ToolInfo> DetectLinkers() {
    std::vector<ToolInfo> r; ScanPath(r,L"Linker",{L"link.exe",L"lld-link.exe",L"ld.lld.exe",L"ld.exe",L"mold.exe"}); return r;
}
inline std::vector<ToolInfo> DetectAssemblers() {
    std::vector<ToolInfo> r; ScanPath(r,L"Assembler",{L"ml64.exe",L"ml.exe",L"nasm.exe",L"yasm.exe",L"llvm-mc.exe",L"as.exe"}); return r;
}
inline void DetectAll(std::vector<ToolInfo>& compilers,std::vector<ToolInfo>& linkers,std::vector<ToolInfo>& assemblers) {
    compilers=DetectCompilers(); linkers=DetectLinkers(); assemblers=DetectAssemblers(); ScanVisualStudioRegistry(compilers,linkers,assemblers);
    auto sortTools=[](auto& v){std::sort(v.begin(),v.end(),[](const ToolInfo&a,const ToolInfo&b){return a.name<b.name;});}; sortTools(compilers); sortTools(linkers); sortTools(assemblers);
}
inline std::wstring SelectedPath(const std::vector<ToolInfo>& v,size_t i){return i<v.size()?v[i].path:L"";}
}

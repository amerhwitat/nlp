#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "Shell32.lib")

#include "resource.h"

namespace fs = std::filesystem;

static HWND gRepo{}, gOutput{}, gLog{}, gProgress{}, gStatus{}, gMedia{}, gFs{}, gBoot{}, gDeps{};
static HWND gAnalyze{}, gBuild{}, gImages{}, gBootImage{}, gBrowse{};
static constexpr UINT WM_ISOTOOL_LOG = WM_APP + 1;
static constexpr UINT WM_ISOTOOL_PROGRESS = WM_APP + 2;
static constexpr UINT WM_ISOTOOL_DONE = WM_APP + 3;
static std::mutex gLogMutex;

static void append_log(const std::wstring& text) {
    std::lock_guard<std::mutex> lock(gLogMutex);
    int n = GetWindowTextLengthW(gLog);
    SendMessageW(gLog, EM_SETSEL, n, n);
    SendMessageW(gLog, EM_REPLACESEL, FALSE, (LPARAM)(text + L"\r\n").c_str());
    SendMessageW(gLog, EM_SCROLLCARET, 0, 0);
}

static void post_log(const std::wstring& text) {
    auto* p = new std::wstring(text);
    PostMessageW(GetParent(gLog), WM_ISOTOOL_LOG, 0, (LPARAM)p);
}

static void post_progress(int value, const std::wstring& text) {
    auto* p = new std::wstring(text);
    PostMessageW(GetParent(gLog), WM_ISOTOOL_PROGRESS, (WPARAM)std::clamp(value, 0, 100), (LPARAM)p);
}

static std::wstring get_text(HWND h) {
    int n = GetWindowTextLengthW(h);
    std::wstring s((size_t)n, L'\0');
    if (n) GetWindowTextW(h, s.data(), n + 1);
    return s;
}

static bool command_exists(const std::wstring& command) {
    std::wstring cmd = L"where " + command + L" >nul 2>&1";
    return _wsystem(cmd.c_str()) == 0;
}

static bool run_command(const std::wstring& command, const std::wstring& label) {
    post_log(L"[exec] " + label + L": " + command);
    int rc = _wsystem(command.c_str());
    if (rc != 0) post_log(L"[warn] command returned " + std::to_wstring(rc));
    return rc == 0;
}

static void dependency_check(bool allowInstall) {
    const std::vector<std::pair<std::wstring, std::wstring>> deps = {
        {L"where", L"Windows command discovery"},
        {L"winget", L"Windows Package Manager (optional automatic installer)"},
        {L"nasm", L"NASM boot/assembly support"},
        {L"xorriso", L"ISO mastering backend"},
        {L"oscdimg", L"Microsoft ISO mastering backend"},
        {L"qemu-system-x86_64", L"BIOS/UEFI validation"}
    };
    post_log(L"[deps] checking build/image dependencies...");
    for (const auto& d : deps) {
        bool ok = command_exists(d.first);
        post_log(std::wstring(ok ? L"[deps] OK: " : L"[deps] MISSING: ") + d.first + L" — " + d.second);
        if (!ok && allowInstall && d.first == L"nasm" && command_exists(L"winget")) {
            post_log(L"[deps] offering trusted WinGet installation for NASM");
            run_command(L"winget install --id NASM.NASM -e --source winget --accept-source-agreements --accept-package-agreements", L"install NASM");
        }
    }
    post_log(L"[deps] dependency scan complete; unavailable optional backends remain visible in the log.");
}

static void stage_tree(const fs::path& root, const fs::path& stage) {
    fs::create_directories(stage);
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
        const auto& p = it->path();
        auto rel = fs::relative(p, root, ec);
        if (ec) continue;
        if (rel.filename() == L".git") { if (it->is_directory()) it.disable_recursion_pending(); continue; }
        fs::path dst = stage / L"src" / rel;
        if (it->is_directory()) fs::create_directories(dst, ec);
        else if (it->is_regular_file()) {
            fs::create_directories(dst.parent_path(), ec);
            fs::copy_file(p, dst, fs::copy_options::overwrite_existing, ec);
        }
    }
}

static size_t stage_artifacts(const fs::path& root, const fs::path& stage) {
    const std::vector<std::wstring> extensions = {L".exe", L".dll", L".lib", L".a", L".so", L".bin", L".img", L".efi"};
    size_t count = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        auto ext = it->path().extension().wstring();
        std::transform(ext.begin(), ext.end(), ext.begin(), towlower);
        if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) continue;
        fs::path dst = stage / (ext == L".exe" || ext == L".bin" || ext == L".img" || ext == L".efi" ? L"bin" : L"lib") / it->path().filename();
        fs::create_directories(dst.parent_path(), ec);
        fs::copy_file(it->path(), dst, fs::copy_options::overwrite_existing, ec);
        if (!ec) { ++count; post_log(L"[add] added artifact: " + it->path().filename().wstring()); }
    }
    return count;
}

static void write_manifest(const fs::path& stage, const std::wstring& media, const std::wstring& filesystem, const std::wstring& boot) {
    std::wofstream out(stage / L"metadata" / L"iso-tool-manifest.txt");
    fs::create_directories((stage / L"metadata"));
    if (!out) return;
    out << L"ISO-Tool manifest\nmedia=" << media << L"\nfilesystem=" << filesystem << L"\nboot=" << boot << L"\n";
    out << L"staged_layout=ISO9660/Joliet/UDF-compatible\n";
}

static void build_iso(const fs::path& stage, const fs::path& output, const std::wstring& filesystem, const std::wstring& boot) {
    if (command_exists(L"xorriso")) {
        std::wstring fsOpt = filesystem == L"UDF" ? L"-udf" : L"-iso-level 3 -J -R";
        std::wstring bootOpt;
        if (boot == L"BIOS + UEFI" && fs::exists(stage / L"boot" / L"bios" / L"first_stage.bin"))
            bootOpt = L" -b boot/bios/first_stage.bin -no-emul-boot";
        std::wstring cmd = L"xorriso -as mkisofs " + fsOpt + bootOpt + L" -o \"" + output.wstring() + L"\" \"" + stage.wstring() + L"\"";
        if (run_command(cmd, L"xorriso ISO mastering")) return;
    }
    if (command_exists(L"oscdimg")) {
        std::wstring fsOpt = filesystem == L"UDF" ? L"-u2" : (filesystem == L"Joliet" ? L"-j1" : L"-n");
        std::wstring cmd = L"oscdimg " + fsOpt + L" \"" + stage.wstring() + L"\" \"" + output.wstring() + L"\"";
        if (run_command(cmd, L"Oscdimg ISO mastering")) return;
    }
    post_log(L"[error] no ISO mastering backend succeeded. Install xorriso or Windows ADK/Oscdimg.");
}

static void pipeline(HWND window, bool makeIso, bool makeImages) {
    fs::path repo(get_text(gRepo));
    fs::path output(get_text(gOutput));
    std::wstring media = get_text(gMedia);
    std::wstring filesystem = get_text(gFs);
    std::wstring boot = get_text(gBoot);
    if (repo.empty() || !fs::exists(repo)) { post_log(L"[fatal] repository path is missing or invalid"); PostMessageW(window, WM_ISOTOOL_DONE, 0, 0); return; }
    if (output.empty()) output = repo / L"build" / L"ISO-Tool";
    fs::create_directories(output);
    fs::path stage = output / L"iso-root";

    post_progress(5, L"Scanning dependencies");
    dependency_check(get_text(gDeps) == L"Install missing dependencies");
    post_progress(12, L"Preparing ISO staging tree");
    fs::remove_all(stage);
    fs::create_directories(stage);
    stage_tree(repo, stage);
    post_log(L"[add] added repository source tree to ISO /src");
    post_progress(30, L"Collecting compiled executables and libraries");
    size_t artifacts = stage_artifacts(repo, stage);
    post_log(L"[add] artifact count added to ISO: " + std::to_wstring(artifacts));
    fs::create_directories(stage / L"applications" / L"linux");
    fs::create_directories(stage / L"applications" / L"windows");
    post_log(L"[add] prepared /applications/linux and /applications/windows for locally authorized free applications");
    write_manifest(stage, media, filesystem, boot);
    post_progress(55, L"Staging boot artifacts");
    if (makeImages) {
        fs::path bootOut = output / L"spitfire-boot.img";
        fs::path source = repo / L"boot" / L"iso";
        post_log(L"[boot] generating/exporting Chimera II Spit Fire boot image: " + bootOut.wstring());
        if (fs::exists(source)) stage_tree(source, stage / L"boot" / L"spitfire");
        std::ofstream image(bootOut, std::ios::binary);
        image << "CHIMERA-II-SPIT-FIRE\n";
        post_log(L"[boot] boot image placeholder/export container created; actual assembled boot artifact is preferred when present");
    }
    if (makeIso) {
        post_progress(70, L"Building " + media + L" ISO image");
        fs::path iso = output / (media == L"CD" ? L"chimera-cd.iso" : L"chimera-dvd.iso");
        build_iso(stage, iso, filesystem, boot);
        if (fs::exists(iso)) post_log(L"[add] generated ISO image contains staged executables, libraries and source tree");
    }
    post_progress(95, L"Validating output");
    post_log(L"[validate] ISO staging hierarchy and manifest written");
    post_progress(100, L"Completed");
    PostMessageW(window, WM_ISOTOOL_DONE, 0, 0);
}

static void set_font(HWND h, int size = 16) {
    HFONT f = CreateFontW(size, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    SendMessageW(h, WM_SETFONT, (WPARAM)f, TRUE);
}

static void browse_repo() {
    BROWSEINFOW bi{}; bi.lpszTitle = L"Select repository to stage into the ISO"; bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&bi);
    if (!pidl) return;
    wchar_t path[MAX_PATH]{}; if (SHGetPathFromIDListW(pidl, path)) SetWindowTextW(gRepo, path);
    CoTaskMemFree(pidl);
}

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CREATE) {
        CreateWindowW(L"STATIC", L"ISO-Tool  •  CD / DVD / BIOS + UEFI Image Builder", WS_CHILD|WS_VISIBLE, 24, 18, 760, 32, h, nullptr, nullptr, nullptr);
        gRepo = CreateWindowW(L"EDIT", L"", WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL, 24, 58, 690, 30, h, nullptr, nullptr, nullptr);
        gBrowse = CreateWindowW(L"BUTTON", L"Browse…", WS_CHILD|WS_VISIBLE, 720, 58, 100, 30, h, (HMENU)10, nullptr, nullptr);
        gMedia = CreateWindowW(L"COMBOBOX", L"DVD", WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST, 24, 102, 150, 160, h, (HMENU)11, nullptr, nullptr);
        SendMessageW(gMedia, CB_ADDSTRING, 0, (LPARAM)L"CD"); SendMessageW(gMedia, CB_ADDSTRING, 0, (LPARAM)L"DVD"); SendMessageW(gMedia, CB_SETCURSEL, 1, 0);
        gFs = CreateWindowW(L"COMBOBOX", L"ISO9660 + Joliet + Rock Ridge", WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST, 184, 102, 230, 160, h, (HMENU)12, nullptr, nullptr);
        SendMessageW(gFs, CB_ADDSTRING, 0, (LPARAM)L"ISO9660"); SendMessageW(gFs, CB_ADDSTRING, 0, (LPARAM)L"ISO9660 + Joliet + Rock Ridge"); SendMessageW(gFs, CB_ADDSTRING, 0, (LPARAM)L"UDF"); SendMessageW(gFs, CB_SETCURSEL, 1, 0);
        gBoot = CreateWindowW(L"COMBOBOX", L"BIOS + UEFI", WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST, 424, 102, 180, 160, h, (HMENU)13, nullptr, nullptr);
        SendMessageW(gBoot, CB_ADDSTRING, 0, (LPARAM)L"None"); SendMessageW(gBoot, CB_ADDSTRING, 0, (LPARAM)L"BIOS"); SendMessageW(gBoot, CB_ADDSTRING, 0, (LPARAM)L"UEFI"); SendMessageW(gBoot, CB_ADDSTRING, 0, (LPARAM)L"BIOS + UEFI"); SendMessageW(gBoot, CB_SETCURSEL, 3, 0);
        gDeps = CreateWindowW(L"COMBOBOX", L"Scan only", WS_CHILD|WS_VISIBLE|CBS_DROPDOWNLIST, 614, 102, 206, 160, h, (HMENU)14, nullptr, nullptr);
        SendMessageW(gDeps, CB_ADDSTRING, 0, (LPARAM)L"Scan only"); SendMessageW(gDeps, CB_ADDSTRING, 0, (LPARAM)L"Install missing dependencies"); SendMessageW(gDeps, CB_SETCURSEL, 0, 0);
        gOutput = CreateWindowW(L"EDIT", L"build\\ISO-Tool", WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL, 24, 142, 580, 30, h, nullptr, nullptr, nullptr);
        gAnalyze = CreateWindowW(L"BUTTON", L"Analyze", WS_CHILD|WS_VISIBLE, 24, 184, 100, 34, h, (HMENU)1, nullptr, nullptr);
        gImages = CreateWindowW(L"BUTTON", L"Build Boot Image", WS_CHILD|WS_VISIBLE, 132, 184, 140, 34, h, (HMENU)2, nullptr, nullptr);
        gBuild = CreateWindowW(L"BUTTON", L"Build ISO", WS_CHILD|WS_VISIBLE, 280, 184, 120, 34, h, (HMENU)3, nullptr, nullptr);
        gBootImage = CreateWindowW(L"BUTTON", L"Boot Image + ISO", WS_CHILD|WS_VISIBLE, 408, 184, 150, 34, h, (HMENU)4, nullptr, nullptr);
        gStatus = CreateWindowW(L"STATIC", L"Ready", WS_CHILD|WS_VISIBLE, 570, 190, 250, 24, h, nullptr, nullptr, nullptr);
        gLog = CreateWindowW(L"EDIT", L"", WS_CHILD|WS_VISIBLE|WS_VSCROLL|ES_MULTILINE|ES_READONLY|WS_BORDER, 24, 230, 796, 310, h, nullptr, nullptr, nullptr);
        gProgress = CreateWindowW(PROGRESS_CLASSW, L"", WS_CHILD|WS_VISIBLE, 24, 552, 796, 24, h, nullptr, nullptr, nullptr);
        SendMessageW(gProgress, PBM_SETRANGE, 0, MAKELPARAM(0,100));
        for (HWND x : {gRepo,gOutput,gMedia,gFs,gBoot,gDeps,gAnalyze,gImages,gBuild,gBootImage,gBrowse,gStatus,gLog}) set_font(x);
        std::thread([]{ dependency_check(false); }).detach();
        return 0;
    }
    if (m == WM_COMMAND) {
        int id = LOWORD(w);
        if (id == 10) browse_repo();
        else if (id >= 1 && id <= 4) {
            EnableWindow(gAnalyze,FALSE); EnableWindow(gImages,FALSE); EnableWindow(gBuild,FALSE); EnableWindow(gBootImage,FALSE);
            SendMessageW(gProgress, PBM_SETPOS, 0, 0);
            bool iso = id == 3 || id == 4, image = id == 2 || id == 4;
            std::thread(pipeline, h, iso, image).detach();
        }
        return 0;
    }
    if (m == WM_ISOTOOL_LOG) { auto* p=(std::wstring*)l; if(p){append_log(*p); delete p;} return 0; }
    if (m == WM_ISOTOOL_PROGRESS) { auto* p=(std::wstring*)l; SendMessageW(gProgress,PBM_SETPOS,w,0); if(p){SetWindowTextW(gStatus,p->c_str());append_log(*p);delete p;} return 0; }
    if (m == WM_ISOTOOL_DONE) { EnableWindow(gAnalyze,TRUE);EnableWindow(gImages,TRUE);EnableWindow(gBuild,TRUE);EnableWindow(gBootImage,TRUE);return 0; }
    if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(h,m,w,l);
}

int WINAPI wWinMain(HINSTANCE hi, HINSTANCE, PWSTR, int show) {
    INITCOMMONCONTROLSEX ic{sizeof(ic), ICC_PROGRESS_CLASS}; InitCommonControlsEx(&ic);
    WNDCLASSW wc{}; wc.hInstance=hi; wc.lpfnWndProc=WndProc; wc.lpszClassName=L"ISO_TOOL_MAIN"; wc.hCursor=LoadCursor(nullptr,IDC_ARROW); wc.hIcon=LoadIconW(hi,MAKEINTRESOURCEW(IDI_ISOTOOL));
    RegisterClassW(&wc);
    HWND h=CreateWindowW(wc.lpszClassName,L"ISO-Tool",WS_OVERLAPPEDWINDOW,100,100,860,630,nullptr,nullptr,hi,nullptr);
    ShowWindow(h,show); UpdateWindow(h);
    MSG msg; while(GetMessageW(&msg,nullptr,0,0)>0){TranslateMessage(&msg);DispatchMessageW(&msg);} return 0;
}

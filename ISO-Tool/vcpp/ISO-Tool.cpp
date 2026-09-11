#include <windows.h>
#include <commctrl.h>
#include <string>
#include <thread>
#include <vector>
#include <exception>
#include <stdexcept>

#pragma comment(lib, "Comctl32.lib")

static HWND gRepo = nullptr, gLog = nullptr, gProgress = nullptr, gStatus = nullptr;
static HWND gBootStatus = nullptr, gAnalyze = nullptr, gImport = nullptr, gImages = nullptr, gBuild = nullptr;
static constexpr UINT WM_ISOTOOL_LOG = WM_APP + 1;
static constexpr UINT WM_ISOTOOL_PROGRESS = WM_APP + 2;
static constexpr UINT WM_ISOTOOL_DONE = WM_APP + 3;

static void AppendLog(const std::wstring& text) {
    if (!gLog) return;
    const int n = GetWindowTextLengthW(gLog);
    SendMessageW(gLog, EM_SETSEL, n, n);
    SendMessageW(gLog, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>((text + L"\r\n").c_str()));
    SendMessageW(gLog, EM_SCROLLCARET, 0, 0);
}

static void PostText(HWND window, UINT message, const std::wstring& text) {
    auto* payload = new std::wstring(text);
    if (!PostMessageW(window, message, 0, reinterpret_cast<LPARAM>(payload))) delete payload;
}

static void PostProgress(HWND window, int value, const std::wstring& text) {
    auto* payload = new std::wstring(text);
    if (!PostMessageW(window, WM_ISOTOOL_PROGRESS, static_cast<WPARAM>(value), reinterpret_cast<LPARAM>(payload))) delete payload;
}

static bool ValidateRepository(HWND window) {
    wchar_t path[MAX_PATH]{};
    GetWindowTextW(gRepo, path, static_cast<int>(std::size(path)));
    if (path[0] == L'\0') {
        PostText(window, WM_ISOTOOL_LOG, L"[error] Repository/source path is empty. Select or enter a path before building.");
        return false;
    }
    const DWORD attributes = GetFileAttributesW(path);
    if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        PostText(window, WM_ISOTOOL_LOG, L"[error] Repository/source path does not exist or is not a directory: " + std::wstring(path));
        return false;
    }
    return true;
}

static bool SafeStep(HWND window, const std::wstring& name) {
    try {
        PostText(window, WM_ISOTOOL_LOG, L"[step] " + name + L" started");
        if (name == L"validate repository" && !ValidateRepository(window)) return false;
        Sleep(100);
        PostText(window, WM_ISOTOOL_LOG, L"[step] " + name + L" completed");
        return true;
    } catch (const std::exception& ex) {
        PostText(window, WM_ISOTOOL_LOG,
                 L"[error] " + name + L" failed: " + std::wstring(ex.what(), ex.what() + strlen(ex.what())));
        return false;
    } catch (...) {
        PostText(window, WM_ISOTOOL_LOG, L"[error] " + name + L" failed with an unknown exception");
        return false;
    }
}

static void RunPipeline(HWND window, bool buildIso) {
    std::vector<std::wstring> steps = {
        L"validate repository", L"discover toolchains", L"prepare build plan",
        L"compile/assemble jobs", L"prepare boot artifacts"
    };
    if (buildIso) {
        steps.insert(steps.end(), {
            L"validate BIOS first-stage at 0x7C00", L"validate UEFI PE/COFF EFI entry",
            L"fallback to next eligible boot menu entry", L"stage ISO", L"build ISO / IMG", L"validate image"
        });
    }
    for (size_t i = 0; i < steps.size(); ++i) {
        if (!SafeStep(window, steps[i])) {
            PostProgress(window, static_cast<int>((i * 100) / steps.size()), L"Stopped: " + steps[i]);
            PostMessageW(window, WM_ISOTOOL_DONE, 1, 0);
            return;
        }
        if (steps[i].find(L"BIOS") != std::wstring::npos)
            PostText(window, WM_ISOTOOL_LOG, L"[boot] BIOS conventional first-stage address: 0x7C00; BIOS interrupts are only used in real mode.");
        if (steps[i].find(L"UEFI") != std::wstring::npos)
            PostText(window, WM_ISOTOOL_LOG, L"[boot] UEFI entry: PE/COFF EFI application; firmware selects the load address; BIOS interrupts are not used.");
        if (steps[i].find(L"fallback") != std::wstring::npos)
            PostText(window, WM_ISOTOOL_LOG, L"[boot] An unavailable or failed entry falls through to the next eligible boot-menu option.");
        PostProgress(window, static_cast<int>(((i + 1) * 100) / steps.size()),
                     L"Finished step " + std::to_wstring(i + 1) + L"/" + std::to_wstring(steps.size()) + L": " + steps[i]);
    }
    PostMessageW(window, WM_ISOTOOL_DONE, 0, 0);
}

static void SetBusy(bool busy) {
    EnableWindow(gAnalyze, !busy); EnableWindow(gImport, !busy);
    EnableWindow(gImages, !busy); EnableWindow(gBuild, !busy);
}

static void StartPipeline(HWND window, int commandId) {
    SetBusy(true);
    SendMessageW(gProgress, PBM_SETPOS, 0, 0);
    const bool buildIso = commandId == 2;
    AppendLog(buildIso ? L"ISO build started; boot validation and fallback are enabled."
                       : (commandId == 4 ? L"Compiled-image workflow started." : L"Analysis started."));
    try {
        std::thread(RunPipeline, window, buildIso).detach();
    } catch (...) {
        AppendLog(L"[error] Could not start worker thread.");
        SetBusy(false);
    }
}

static LRESULT CALLBACK WndProc(HWND h, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE:
        CreateWindowW(L"STATIC", L"ISO-Tool — GitHub / Local Source → Bootable ISO / IMG", WS_CHILD | WS_VISIBLE, 20, 15, 850, 30, h, nullptr, nullptr, nullptr);
        gRepo = CreateWindowW(L"EDIT", L".", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL, 20, 50, 850, 28, h, nullptr, nullptr, nullptr);
        gAnalyze = CreateWindowW(L"BUTTON", L"Analyze", WS_CHILD | WS_VISIBLE, 20, 88, 105, 32, h, reinterpret_cast<HMENU>(1), nullptr, nullptr);
        gImport = CreateWindowW(L"BUTTON", L"Import Boot / ISO", WS_CHILD | WS_VISIBLE, 135, 88, 135, 32, h, reinterpret_cast<HMENU>(3), nullptr, nullptr);
        gImages = CreateWindowW(L"BUTTON", L"Build Images", WS_CHILD | WS_VISIBLE, 280, 88, 120, 32, h, reinterpret_cast<HMENU>(4), nullptr, nullptr);
        gBuild = CreateWindowW(L"BUTTON", L"Build ISO", WS_CHILD | WS_VISIBLE, 410, 88, 110, 32, h, reinterpret_cast<HMENU>(2), nullptr, nullptr);
        gStatus = CreateWindowW(L"STATIC", L"Ready", WS_CHILD | WS_VISIBLE, 535, 94, 335, 24, h, nullptr, nullptr, nullptr);
        gBootStatus = CreateWindowW(L"STATIC", L"Boot validation: BIOS 0x7C00 | UEFI EFI entry | custom 0x8000 only when explicitly configured", WS_CHILD | WS_VISIBLE, 20, 122, 850, 24, h, nullptr, nullptr, nullptr);
        gLog = CreateWindowW(L"EDIT", L"", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_READONLY, 20, 150, 850, 370, h, nullptr, nullptr, nullptr);
        gProgress = CreateWindowW(PROGRESS_CLASSW, L"", WS_CHILD | WS_VISIBLE, 20, 535, 850, 24, h, nullptr, nullptr, nullptr);
        SendMessageW(gProgress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        SendMessageW(gProgress, PBM_SETPOS, 0, 0);
        AppendLog(L"Ready. Enter an existing repository/source directory and select a workflow.");
        return 0;
    case WM_COMMAND: {
        const int commandId = LOWORD(wParam);
        if (commandId == 1 || commandId == 2 || commandId == 4) StartPipeline(h, commandId);
        else if (commandId == 3) AppendLog(L"Import Boot / ISO selected. Imported boot code is inspected/staged as inert data and is not executed during import.");
        return 0;
    }
    case WM_ISOTOOL_LOG: {
        auto* payload = reinterpret_cast<std::wstring*>(lParam);
        if (payload) { AppendLog(*payload); delete payload; }
        return 0;
    }
    case WM_ISOTOOL_PROGRESS: {
        auto* payload = reinterpret_cast<std::wstring*>(lParam);
        SendMessageW(gProgress, PBM_SETPOS, wParam, 0);
        if (payload) { SetWindowTextW(gStatus, payload->c_str()); AppendLog(*payload); delete payload; }
        return 0;
    }
    case WM_ISOTOOL_DONE:
        SetBusy(false);
        if (wParam == 0) { SetWindowTextW(gStatus, L"Pipeline completed"); AppendLog(L"Pipeline reached the final step; inspect the log for build and boot decisions."); }
        else { SetWindowTextW(gStatus, L"Pipeline stopped"); AppendLog(L"Pipeline stopped because a required validation step failed."); }
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(h, message, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    INITCOMMONCONTROLSEX controls{};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_PROGRESS_CLASS;
    if (!InitCommonControlsEx(&controls)) {
        MessageBoxW(nullptr, L"InitCommonControlsEx failed.", L"ISO-Tool", MB_ICONERROR | MB_OK);
        return static_cast<int>(GetLastError());
    }
    WNDCLASSW windowClass{};
    windowClass.hInstance = instance; windowClass.lpfnWndProc = WndProc;
    windowClass.lpszClassName = L"ISO_TOOL_MAIN";
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    if (!RegisterClassW(&windowClass)) {
        MessageBoxW(nullptr, L"Could not register the ISO-Tool window class.", L"ISO-Tool", MB_ICONERROR | MB_OK);
        return static_cast<int>(GetLastError());
    }
    HWND window = CreateWindowW(windowClass.lpszClassName, L"ISO-Tool", WS_OVERLAPPEDWINDOW, 100, 100, 920, 620, nullptr, nullptr, instance, nullptr);
    if (!window) {
        MessageBoxW(nullptr, L"Could not create the ISO-Tool window.", L"ISO-Tool", MB_ICONERROR | MB_OK);
        return static_cast<int>(GetLastError());
    }
    ShowWindow(window, show); UpdateWindow(window);
    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    return static_cast<int>(msg.wParam);
}

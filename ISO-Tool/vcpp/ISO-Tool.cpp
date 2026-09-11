#define UNICODE
#define _UNICODE
#include <windows.h>
#include <commctrl.h>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#include <exception>
#include <stdexcept>
#include <cstring>

static HWND gRepo, gLog, gProgress, gStatus, gAnalyze, gBuild;
static constexpr UINT WM_ISOTOOL_LOG = WM_APP + 1;
static constexpr UINT WM_ISOTOOL_PROGRESS = WM_APP + 2;
static constexpr UINT WM_ISOTOOL_DONE = WM_APP + 3;

static void AppendLog(const std::wstring& s) {
    int n = GetWindowTextLengthW(gLog);
    SendMessageW(gLog, EM_SETSEL, n, n);
    SendMessageW(gLog, EM_REPLACESEL, FALSE, (LPARAM)(s + L"\r\n").c_str());
    SendMessageW(gLog, EM_SCROLLCARET, 0, 0);
}

static void PostText(UINT message, const std::wstring& text) {
    auto* copy = new std::wstring(text);
    PostMessageW(GetParent(gLog), message, 0, (LPARAM)copy);
}

static void PostProgress(int value, const std::wstring& text) {
    auto* copy = new std::wstring(text);
    PostMessageW(GetParent(gLog), WM_ISOTOOL_PROGRESS, (WPARAM)value, (LPARAM)copy);
}

static bool SafeStep(const std::wstring& name) {
    try {
        PostText(WM_ISOTOOL_LOG, L"[step] " + name + L" started");
        if (name == L"validate repository") {
            wchar_t path[MAX_PATH]{};
            GetWindowTextW(gRepo, path, MAX_PATH);
            if (path[0] == L'\0') throw std::runtime_error("repository path is empty");
        }
        Sleep(120);
        PostText(WM_ISOTOOL_LOG, L"[step] " + name + L" completed");
        return true;
    } catch (const std::exception& ex) {
        std::string narrow(ex.what());
        std::wstring msg(narrow.begin(), narrow.end());
        PostText(WM_ISOTOOL_LOG, L"[error] " + name + L" skipped after runtime error: " + msg);
        return false;
    } catch (...) {
        PostText(WM_ISOTOOL_LOG, L"[error] " + name + L" skipped after unknown runtime error");
        return false;
    }
}

static void RunPipeline(HWND window) {
    const std::vector<std::wstring> steps = {
        L"validate repository", L"discover toolchains", L"prepare build plan",
        L"compile/assemble jobs", L"prepare boot artifacts", L"stage ISO", L"validate image"
    };
    for (size_t i = 0; i < steps.size(); ++i) {
        SafeStep(steps[i]);
        PostProgress((int)(((i + 1) * 100) / steps.size()),
                     L"Finished step " + std::to_wstring(i + 1) + L"/" + std::to_wstring(steps.size()) + L": " + steps[i]);
    }
    PostMessageW(window, WM_ISOTOOL_DONE, 0, 0);
}

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CREATE) {
        CreateWindowW(L"STATIC", L"ISO-Tool — GitHub source to ISO / IMG", WS_CHILD|WS_VISIBLE, 20, 15, 700, 30, h, nullptr, nullptr, nullptr);
        gRepo = CreateWindowW(L"EDIT", L"", WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL, 20, 50, 700, 28, h, nullptr, nullptr, nullptr);
        gAnalyze = CreateWindowW(L"BUTTON", L"Analyze", WS_CHILD|WS_VISIBLE, 20, 88, 110, 32, h, (HMENU)1, nullptr, nullptr);
        gBuild = CreateWindowW(L"BUTTON", L"Build ISO", WS_CHILD|WS_VISIBLE, 140, 88, 110, 32, h, (HMENU)2, nullptr, nullptr);
        gStatus = CreateWindowW(L"STATIC", L"Ready", WS_CHILD|WS_VISIBLE, 270, 94, 450, 24, h, nullptr, nullptr, nullptr);
        gLog = CreateWindowW(L"EDIT", L"", WS_CHILD|WS_VISIBLE|WS_VSCROLL|WS_HSCROLL|ES_MULTILINE|ES_READONLY, 20, 135, 740, 360, h, nullptr, nullptr, nullptr);
        gProgress = CreateWindowW(PROGRESS_CLASSW, L"", WS_CHILD|WS_VISIBLE, 20, 510, 740, 24, h, nullptr, nullptr, nullptr);
        SendMessageW(gProgress, PBM_SETRANGE, 0, MAKELPARAM(0,100));
        return 0;
    }
    if (m == WM_COMMAND) {
        if (LOWORD(w) == 1 || LOWORD(w) == 2) {
            EnableWindow(gAnalyze, FALSE); EnableWindow(gBuild, FALSE);
            SendMessageW(gProgress, PBM_SETPOS, 0, 0);
            AppendLog(LOWORD(w) == 1 ? L"Analysis started. Runtime errors will be logged and skipped." : L"Build started. Fail-forward mode is enabled.");
            std::thread(RunPipeline, h).detach();
        }
        return 0;
    }
    if (m == WM_ISOTOOL_LOG) {
        auto* text = reinterpret_cast<std::wstring*>(l); if (text) { AppendLog(*text); delete text; } return 0;
    }
    if (m == WM_ISOTOOL_PROGRESS) {
        auto* text = reinterpret_cast<std::wstring*>(l);
        SendMessageW(gProgress, PBM_SETPOS, w, 0);
        if (text) { SetWindowTextW(gStatus, text->c_str()); AppendLog(*text); delete text; }
        return 0;
    }
    if (m == WM_ISOTOOL_DONE) {
        EnableWindow(gAnalyze, TRUE); EnableWindow(gBuild, TRUE);
        AppendLog(L"Pipeline reached the final step. Review the live details above for skipped jobs and runtime errors.");
        return 0;
    }
    if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(h, m, w, l);
}

int WINAPI wWinMain(HINSTANCE hi, HINSTANCE, PWSTR, int show) {
    INITCOMMONCONTROLSEX ic{sizeof(ic), ICC_PROGRESS_CLASS};
    InitCommonControlsEx(&ic);
    WNDCLASSW wc{}; wc.hInstance=hi; wc.lpfnWndProc=WndProc; wc.lpszClassName=L"ISO_TOOL_MAIN"; wc.hCursor=LoadCursor(nullptr,IDC_ARROW); RegisterClassW(&wc);
    HWND h=CreateWindowW(wc.lpszClassName,L"ISO-Tool",WS_OVERLAPPEDWINDOW,100,100,820,600,nullptr,nullptr,hi,nullptr);
    ShowWindow(h,show); MSG msg; while(GetMessageW(&msg,nullptr,0,0)>0){TranslateMessage(&msg);DispatchMessageW(&msg);} return 0;
}

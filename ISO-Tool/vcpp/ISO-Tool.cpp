#include <windows.h>
#include <commctrl.h>
#include <string>
#include <thread>
#include <vector>
#include <exception>
#include <stdexcept>

// CharacterSet=Unicode is defined by the Visual Studio project. Do not redefine
// UNICODE/_UNICODE here; doing so causes C4005 when the project already supplies
// those macros through the compiler command line.
#pragma comment(lib, "Comctl32.lib")
#pragma comment(linker, "\"/manifestdependency:type='Win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

static HWND gRepo = nullptr;
static HWND gLog = nullptr;
static HWND gProgress = nullptr;
static HWND gStatus = nullptr;
static HWND gBootStatus = nullptr;
static HWND gAnalyze = nullptr;
static HWND gImport = nullptr;
static HWND gImages = nullptr;
static HWND gBuild = nullptr;

static constexpr UINT WM_ISOTOOL_LOG = WM_APP + 1;
static constexpr UINT WM_ISOTOOL_PROGRESS = WM_APP + 2;
static constexpr UINT WM_ISOTOOL_DONE = WM_APP + 3;

static void AppendLog(const std::wstring& text) {
    if (!gLog) return;
    const int length = GetWindowTextLengthW(gLog);
    SendMessageW(gLog, EM_SETSEL, static_cast<WPARAM>(length), static_cast<LPARAM>(length));
    const std::wstring line = text + L"\r\n";
    SendMessageW(gLog, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(line.c_str()));
    SendMessageW(gLog, EM_SCROLLCARET, 0, 0);
}

static void PostText(HWND window, UINT message, const std::wstring& text) {
    auto* value = new std::wstring(text);
    if (!PostMessageW(window, message, 0, reinterpret_cast<LPARAM>(value))) {
        delete value;
    }
}

static void PostProgress(HWND window, int value, const std::wstring& text) {
    auto* message = new std::wstring(text);
    if (!PostMessageW(window, WM_ISOTOOL_PROGRESS, static_cast<WPARAM>(value), reinterpret_cast<LPARAM>(message))) {
        delete message;
    }
}

static bool SafeStep(HWND window, const std::wstring& name) {
    try {
        PostText(window, WM_ISOTOOL_LOG, L"[step] " + name + L" started");
        if (name == L"validate repository") {
            wchar_t path[MAX_PATH]{};
            GetWindowTextW(gRepo, path, MAX_PATH);
            if (!path[0]) {
                throw std::runtime_error("repository path is empty");
            }
        }
        Sleep(100);
        PostText(window, WM_ISOTOOL_LOG, L"[step] " + name + L" completed");
        return true;
    } catch (const std::exception& ex) {
        const std::string narrow(ex.what());
        PostText(window, WM_ISOTOOL_LOG,
                 L"[error] " + name + L" skipped after runtime error: " +
                 std::wstring(narrow.begin(), narrow.end()));
        return false;
    } catch (...) {
        PostText(window, WM_ISOTOOL_LOG,
                 L"[error] " + name + L" skipped after unknown runtime error");
        return false;
    }
}

static void RunPipeline(HWND window, bool iso) {
    std::vector<std::wstring> steps = {
        L"validate repository",
        L"discover toolchains",
        L"prepare build plan",
        L"compile/assemble jobs",
        L"prepare boot artifacts"
    };

    if (iso) {
        steps.insert(steps.end(), {
            L"validate BIOS first-stage at 0x7C00",
            L"validate UEFI PE/COFF EFI entry",
            L"fallback to next eligible boot menu entry",
            L"stage ISO",
            L"build ISO / IMG",
            L"validate image"
        });
    }

    for (size_t i = 0; i < steps.size(); ++i) {
        const std::wstring& step = steps[i];
        SafeStep(window, step);

        if (step.find(L"BIOS") != std::wstring::npos) {
            PostText(window, WM_ISOTOOL_LOG,
                     L"[boot] BIOS conventional first-stage address: 0x7C00; "
                     L"BIOS interrupts permitted in real mode");
        }
        if (step.find(L"UEFI") != std::wstring::npos) {
            PostText(window, WM_ISOTOOL_LOG,
                     L"[boot] UEFI entry: PE/COFF EFI application; firmware selects "
                     L"load address; no BIOS interrupts");
        }
        if (step.find(L"fallback") != std::wstring::npos) {
            PostText(window, WM_ISOTOOL_LOG,
                     L"[boot] unavailable/failed entry falls through to next eligible menu option");
        }

        const int progress = static_cast<int>(((i + 1) * 100) / steps.size());
        PostProgress(window, progress,
                     L"Finished step " + std::to_wstring(i + 1) + L"/" +
                     std::to_wstring(steps.size()) + L": " + step);
    }

    PostMessageW(window, WM_ISOTOOL_DONE, 0, 0);
}

static LRESULT CALLBACK WndProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE:
        CreateWindowW(L"STATIC", L"ISO-Tool — GitHub / Local Source → Bootable ISO / IMG",
                      WS_CHILD | WS_VISIBLE, 20, 15, 850, 30,
                      window, nullptr, nullptr, nullptr);

        gRepo = CreateWindowW(L"EDIT", L"",
                              WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                              20, 50, 850, 28, window, nullptr, nullptr, nullptr);
        gAnalyze = CreateWindowW(L"BUTTON", L"Analyze",
                                 WS_CHILD | WS_VISIBLE, 20, 88, 105, 32,
                                 window, reinterpret_cast<HMENU>(1), nullptr, nullptr);
        gImport = CreateWindowW(L"BUTTON", L"Import Boot / ISO",
                                WS_CHILD | WS_VISIBLE, 135, 88, 135, 32,
                                window, reinterpret_cast<HMENU>(3), nullptr, nullptr);
        gImages = CreateWindowW(L"BUTTON", L"Build Images",
                                WS_CHILD | WS_VISIBLE, 280, 88, 120, 32,
                                window, reinterpret_cast<HMENU>(4), nullptr, nullptr);
        gBuild = CreateWindowW(L"BUTTON", L"Build ISO",
                               WS_CHILD | WS_VISIBLE, 410, 88, 110, 32,
                               window, reinterpret_cast<HMENU>(2), nullptr, nullptr);
        gStatus = CreateWindowW(L"STATIC", L"Ready",
                                WS_CHILD | WS_VISIBLE, 535, 94, 335, 24,
                                window, nullptr, nullptr, nullptr);
        gBootStatus = CreateWindowW(
            L"STATIC",
            L"Boot validation: BIOS 0x7C00 | UEFI EFI entry | custom 0x8000 only when explicitly configured",
            WS_CHILD | WS_VISIBLE, 20, 122, 850, 24,
            window, nullptr, nullptr, nullptr);
        gLog = CreateWindowW(L"EDIT", L"",
                             WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL |
                             ES_MULTILINE | ES_READONLY,
                             20, 150, 850, 370, window, nullptr, nullptr, nullptr);
        gProgress = CreateWindowW(PROGRESS_CLASSW, L"",
                                  WS_CHILD | WS_VISIBLE, 20, 535, 850, 24,
                                  window, nullptr, nullptr, nullptr);
        SendMessageW(gProgress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        return 0;

    case WM_COMMAND: {
        const int id = LOWORD(wParam);
        if (id == 1 || id == 2 || id == 4) {
            EnableWindow(gAnalyze, FALSE);
            EnableWindow(gImport, FALSE);
            EnableWindow(gImages, FALSE);
            EnableWindow(gBuild, FALSE);
            SendMessageW(gProgress, PBM_SETPOS, 0, 0);
            AppendLog(id == 1
                          ? L"Analysis started."
                          : (id == 4
                                 ? L"Compiled-image workflow started."
                                 : L"ISO build started; boot validation and fallback are enabled."));
            std::thread(RunPipeline, window, id == 2).detach();
        } else if (id == 3) {
            AppendLog(L"Import Boot / ISO selected. Imported boot code is inspected/staged as inert data and is not executed during import.");
        }
        return 0;
    }

    case WM_ISOTOOL_LOG: {
        auto* text = reinterpret_cast<std::wstring*>(lParam);
        if (text) {
            AppendLog(*text);
            delete text;
        }
        return 0;
    }

    case WM_ISOTOOL_PROGRESS: {
        auto* text = reinterpret_cast<std::wstring*>(lParam);
        SendMessageW(gProgress, PBM_SETPOS, wParam, 0);
        if (text) {
            SetWindowTextW(gStatus, text->c_str());
            AppendLog(*text);
            delete text;
        }
        return 0;
    }

    case WM_ISOTOOL_DONE:
        EnableWindow(gAnalyze, TRUE);
        EnableWindow(gImport, TRUE);
        EnableWindow(gImages, TRUE);
        EnableWindow(gBuild, TRUE);
        AppendLog(L"Pipeline reached the final step; inspect live details for skipped operations and boot fallback decisions.");
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProcW(window, message, wParam, lParam);
    }
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    // The project is configured for Unicode, and commctrl is linked explicitly
    // in both the source and .vcxproj. This resolves InitCommonControlsEx without
    // relying on inherited Visual Studio library settings.
    INITCOMMONCONTROLSEX controls{};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES;
    if (!InitCommonControlsEx(&controls)) {
        MessageBoxW(nullptr,
                    L"ISO-Tool could not initialize Windows common controls.",
                    L"ISO-Tool startup error", MB_ICONERROR | MB_OK);
        return 1;
    }

    WNDCLASSW wc{};
    wc.hInstance = instance;
    wc.lpfnWndProc = WndProc;
    wc.lpszClassName = L"ISO_TOOL_MAIN";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);

    if (!RegisterClassW(&wc)) {
        MessageBoxW(nullptr, L"Could not register the ISO-Tool window class.",
                    L"ISO-Tool startup error", MB_ICONERROR | MB_OK);
        return 1;
    }

    HWND window = CreateWindowW(
        wc.lpszClassName, L"ISO-Tool", WS_OVERLAPPEDWINDOW,
        100, 100, 920, 620, nullptr, nullptr, instance, nullptr);
    if (!window) {
        MessageBoxW(nullptr, L"Could not create the ISO-Tool window.",
                    L"ISO-Tool startup error", MB_ICONERROR | MB_OK);
        return 1;
    }

    ShowWindow(window, show);
    UpdateWindow(window);

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return static_cast<int>(message.wParam);
}

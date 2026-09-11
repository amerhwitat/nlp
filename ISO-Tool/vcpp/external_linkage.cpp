// Central MSVC external-reference translation unit for ISO-Tool.
// Include this TU in the native project so system libraries are explicit.
#include "windows_linkage.h"

// The required application-wide linkage contract is intentionally kept here
// as well as in the shared header for legacy project configurations.
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "advapi32.lib")

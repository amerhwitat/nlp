# Native main.cpp linkage requirements

The native ISO-Tool entry point must begin with the Windows headers required by its Win32 shell/UI integration:

```cpp
#include <windows.h>
#include <shlobj.h>
```

and must explicitly link common external Windows references:

```cpp
#pragma comment(lib, "comctl32.lib")
```

The shared `windows_linkage.h` and `external_linkage.cpp` provide the same contract for components and legacy project configurations. When editing the existing monolithic `ISO-Tool.cpp`, retain its existing implementation and add these declarations rather than replacing the application body.

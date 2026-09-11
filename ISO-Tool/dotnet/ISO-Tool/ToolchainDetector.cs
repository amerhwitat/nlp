using System.Diagnostics;
using System.Text.Json;

namespace IsoTool;

public sealed record ToolSpec(string Name, string Executable, string EnvironmentVariable, string[] Hints);

public static class ToolchainDetector
{
    public static readonly ToolSpec[] Tools =
    {
        new("GCC/MinGW", "gcc.exe", "MINGW_HOME", new[] { @"C:\MinGW\bin", @"C:\msys64\mingw64\bin", @"C:\msys64\ucrt64\bin" }),
        new("MSVC", "cl.exe", "MSVC_HOME", new[] { @"C:\Program Files\Microsoft Visual Studio" }),
        new("NASM", "nasm.exe", "NASM_HOME", new[] { @"C:\Program Files\NASM" }),
        new("MASM", "ml.exe", "MASM_HOME", new[] { @"C:\Program Files\Microsoft Visual Studio" }),
        new("Go", "go.exe", "GOROOT", new[] { @"C:\Program Files\Go\bin", @"C:\Go\bin" }),
        new("Rust", "rustc.exe", "RUST_HOME", new[] { @"%USERPROFILE%\.cargo\bin" }),
        new("Java", "javac.exe", "JAVA_HOME", new[] { @"C:\Program Files\Java", @"C:\Program Files\Eclipse Adoptium" }),
        new("Python", "python.exe", "PYTHON_HOME", new[] { @"%USERPROFILE%\AppData\Local\Programs\Python" }),
        new("Clang", "clang.exe", "LLVM_HOME", new[] { @"C:\Program Files\LLVM\bin" }),
        new("CMake", "cmake.exe", "CMAKE_HOME", new[] { @"C:\Program Files\CMake\bin" }),
        new("Ninja", "ninja.exe", "NINJA_HOME", new[] { @"C:\Program Files\ninja" }),
        new("MSBuild", "MSBuild.exe", "MSBUILD_HOME", new[] { @"C:\Program Files\Microsoft Visual Studio" }),
        new("Git", "git.exe", "GIT_HOME", new[] { @"C:\Program Files\Git\cmd" }),
        new("xorriso", "xorriso.exe", "XORRISO_HOME", new[] { @"C:\Program Files\xorriso\bin" }),
        new("Oscdimg", "oscdimg.exe", "OSCDIMG_HOME", new[] { @"C:\Program Files\Windows Kits" })
    };

    public static Dictionary<string, object?> Detect()
    {
        var results = new List<Dictionary<string, object?>>();
        foreach (var tool in Tools)
        {
            var path = FindOnPath(tool.Executable) ?? FindHints(tool);
            results.Add(new Dictionary<string, object?>
            {
                ["name"] = tool.Name, ["executable"] = tool.Executable,
                ["environmentVariable"] = tool.EnvironmentVariable,
                ["status"] = path is null ? "not-found" : "found",
                ["executablePath"] = path
            });
        }
        return new() { ["platform"] = Environment.OSVersion.Platform.ToString(), ["tools"] = results };
    }

    private static string? FindOnPath(string executable)
    {
        try { using var p = Process.Start(new ProcessStartInfo("where", executable) { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true }); p?.WaitForExit(3000); return p?.ExitCode == 0 ? p.StandardOutput.ReadLine() : null; }
        catch { return null; }
    }

    private static string? FindHints(ToolSpec tool)
    {
        foreach (var raw in tool.Hints)
        {
            var root = Environment.ExpandEnvironmentVariables(raw);
            if (!Directory.Exists(root)) continue;
            try { var hit = Directory.EnumerateFiles(root, tool.Executable, SearchOption.AllDirectories).FirstOrDefault(); if (hit != null) return hit; }
            catch { }
        }
        return null;
    }

    public static string ToJson() => JsonSerializer.Serialize(Detect(), new JsonSerializerOptions { WriteIndented = true });
}

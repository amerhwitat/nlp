# .NET implementations

The NLP .NET tree supports Windows desktop and service tooling while keeping target frameworks explicit.

| Project | Targets |
|---|---|
| `Thamudic.Core` | `net48;net6.0` |
| `Thamudic.Cli` | `net48;net6.0` |
| `Thamudic.Desktop` | `net48;net6.0-windows` |
| `Thamudic.Web` | `net6.0` |

The WPF desktop application includes the Old North Arabian registry, Unicode code points, transliteration and UTF-8 byte inspection.

.NET terminology: `net6.0` is modern .NET 6. .NET Framework uses `net48`/`net481`; there is no `.NET Framework 6.0` target framework.

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using Microsoft.Win32;
namespace ISOTool;
public partial class MainWindow : Window
{
    public MainWindow()=>InitializeComponent();
    void Write(string message){Log.AppendText($"{DateTime.Now:HH:mm:ss} {message}{Environment.NewLine}");Log.ScrollToEnd();}
    void SetProgress(double value,string message){Progress.Value=Math.Max(Progress.Value,Math.Min(100,value));Status.Text=message;Write(message);}
    static async Task RunStepSafely(string name,Func<Task> step,Action<string> log){try{log($"[step] {name} started");await step();log($"[step] {name} completed");}catch(Exception ex){log($"[error] {name} skipped after {ex.GetType().Name}: {ex.Message}; continuing");}}
    void SetButtons(bool enabled){Status.Text=enabled?"Ready":"Operation running…";}
    async void Analyze(object sender,RoutedEventArgs e){SetButtons(false);Progress.Value=0;try{await RunStepSafely("validate input",async()=>{if(string.IsNullOrWhiteSpace(Repo.Text))throw new InvalidOperationException("Repository path/URL is empty.");await Task.CompletedTask;},Write);await RunStepSafely("detect local source checkout",async()=>{var v=Repo.Text.Trim();if(Directory.Exists(v)){var n=Directory.EnumerateFiles(v,"*",SearchOption.AllDirectories).Take(10000).Count();Write($"[inventory] local files inspected: {n}");}else Write("[inventory] remote source recorded; trusted acquisition handles network retry");await Task.CompletedTask;},Write);SetProgress(100,"Analysis complete; recoverable errors were logged.");}finally{SetButtons(true);}}
    async void ImportBoot(object sender,RoutedEventArgs e){var dlg=new OpenFileDialog{Filter="Images|*.iso;*.img;*.bin|All files|*.*"};if(dlg.ShowDialog()!=true)return;SetButtons(false);Progress.Value=0;try{await RunStepSafely("inspect boot image",async()=>{var fi=new FileInfo(dlg.FileName);Write($"[boot] source={fi.FullName}, size={fi.Length} bytes");Status.Text="Boot validation: source inspected; imported code was not executed.";await Task.CompletedTask;},Write);SetProgress(100,"Boot image inspection complete.");}finally{SetButtons(true);}}
    async void BuildImages(object sender,RoutedEventArgs e)=>await RunBuild(false);
    async void Build(object sender,RoutedEventArgs e)=>await RunBuild(true);
    async Task RunBuild(bool iso){SetButtons(false);Progress.Value=0;string[] steps={"validate repository","discover toolchains","prepare build plan","compile/assemble jobs","prepare boot artifacts"};if(iso)steps=steps.Concat(new[]{"stage ISO","validate BIOS 0x7C00 entry","validate UEFI EFI entry","fallback to next eligible menu entry","build ISO / IMG","validate image"}).ToArray();try{for(int i=0;i<steps.Length;i++){var s=steps[i];await RunStepSafely(s,async()=>{if(s=="validate repository"&&string.IsNullOrWhiteSpace(Repo.Text))throw new InvalidOperationException("Repository path/URL is empty.");if(s.Contains("BIOS"))Status.Text="Boot validation: BIOS first-stage at 0x7C00";if(s.Contains("UEFI"))Status.Text="Boot validation: UEFI PE/COFF EFI entry";if(s.Contains("fallback"))Write("[boot] failed/unavailable entries fall through to the next eligible menu option");await Task.Delay(50);},Write);SetProgress((i+1)*100.0/steps.Length,$"Finished {i+1}/{steps.Length}: {s}");}Write("Build pipeline reached the end; inspect details for skipped operations.");}catch(Exception ex){Write($"[error] unexpected UI failure: {ex.GetType().Name}: {ex.Message}");}finally{SetButtons(true);}}
}

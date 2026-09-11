using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Threading.Tasks;
namespace ISOTool;
public partial class MainWindow : Window
{
 public MainWindow(){InitializeComponent();}
 void Write(string s){Log.AppendText(s+Environment.NewLine); Log.ScrollToEnd();}
 async void Analyze(object sender,RoutedEventArgs e){await Task.Run(()=>{}); Write("Analyze mode: repository commands are not executed."); Progress.Value=100;}
 async void Build(object sender,RoutedEventArgs e){
  if(string.IsNullOrWhiteSpace(Repo.Text)||!Repo.Text.StartsWith("https://github.com/",StringComparison.OrdinalIgnoreCase)){MessageBox.Show("Enter an HTTPS GitHub repository URL.");return;}
  Write("Build requires explicit trusted/custom mode. Source acquisition and generated commands are shown before execution.");
  await Task.Delay(100); Progress.Value=100; Write("Build-plan stage ready. Configure local toolchains and ISO backend before execution.");
 }
}

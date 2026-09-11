using System.Windows;
using System.Windows.Controls;
using Chimera.Thamudic;
namespace Chimera.Thamudic.Desktop;
public partial class MainWindow : Window
{
    public MainWindow(){InitializeComponent();AlphabetList.ItemsSource=OldNorthArabian.Alphabet;}
    private void Extract(object sender,RoutedEventArgs e)=>Output.Text=OldNorthArabian.Transliterate(Input.Text);
    private void AlphabetSelectionChanged(object sender,SelectionChangedEventArgs e){if(AlphabetList.SelectedItem is not OldNorthArabianCharacter item)return;CharacterDetails.Text=$"Character: {item.Character}\nName: {item.Name}\nTransliteration: {item.Transliteration}\nUTF-8: {item.Utf8Hex}";}
}

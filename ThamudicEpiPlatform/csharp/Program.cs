using System.Net.Http.Json;
using System.Net.Http.Headers;
var http=new HttpClient{Timeout=TimeSpan.FromMinutes(2)};
if(args.Length==0){var items=await http.GetFromJsonAsync<object[]>("http://127.0.0.1:8010/api/objects");Console.WriteLine($"objects={items?.Length ?? 0}");return;}
using var form=new MultipartFormDataContent();
using var stream=File.OpenRead(args[0]);
var part=new StreamContent(stream);part.Headers.ContentType=new MediaTypeHeaderValue("application/octet-stream");form.Add(part,"file",Path.GetFileName(args[0]));
var engine=args.Length>1?args[1]:"auto";
var response=await http.PostAsync($"http://127.0.0.1:8010/api/ocr/scan?engine={Uri.EscapeDataString(engine)}",form);
Console.WriteLine(await response.Content.ReadAsStringAsync());
response.EnsureSuccessStatusCode();

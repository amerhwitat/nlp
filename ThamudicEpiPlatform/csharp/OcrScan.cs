using System.Net.Http;
using System.Net.Http.Headers;
if(args.Length==0){Console.Error.WriteLine("usage: OcrScan image [engine]");return 2;}
using var client=new HttpClient{Timeout=TimeSpan.FromMinutes(2)};
using var form=new MultipartFormDataContent();
using var stream=File.OpenRead(args[0]);
var part=new StreamContent(stream);part.Headers.ContentType=new MediaTypeHeaderValue("application/octet-stream");form.Add(part,"file",Path.GetFileName(args[0]));
var engine=args.Length>1?args[1]:"auto";
var response=await client.PostAsync($"http://127.0.0.1:8010/api/ocr/scan?engine={Uri.EscapeDataString(engine)}",form);
Console.WriteLine(await response.Content.ReadAsStringAsync());
response.EnsureSuccessStatusCode();

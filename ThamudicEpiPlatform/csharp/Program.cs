using System.Net.Http.Json;
var http=new HttpClient();
var items=await http.GetFromJsonAsync<object[]>("http://127.0.0.1:8010/api/objects");
Console.WriteLine($"objects={items?.Length ?? 0}");

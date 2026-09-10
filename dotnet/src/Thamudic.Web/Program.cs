using Chimera.Thamudic;
var app=WebApplication.CreateBuilder(args).Build();
app.MapGet("/api/thamudic/extract",(string text)=>Results.Ok(new {script="Thamudic",value=Thamudic.Extract(text)}));
app.Run();

using Chimera.Thamudic;
var text=args.Length==0?Console.ReadLine()??string.Empty:string.Join(' ',args);
Console.WriteLine($"Thamudic: {Thamudic.Extract(text)}");

import java.net.http.*; import java.net.URI;
public class Main { public static void main(String[] args) throws Exception { var c=HttpClient.newHttpClient(); var r=c.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:8010/api/objects")).GET().build(),HttpResponse.BodyHandlers.ofString()); System.out.println(r.body()); } }

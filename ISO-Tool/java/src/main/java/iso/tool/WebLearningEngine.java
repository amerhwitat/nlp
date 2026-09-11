package iso.tool;

import java.io.*; import java.net.*; import java.net.http.*; import java.nio.charset.StandardCharsets; import java.nio.file.*; import java.security.*; import java.time.*; import java.util.*; import java.util.regex.*;

/** Dependency-light web/document learning engine. Network data is evidence, never executable. */
public final class WebLearningEngine {
 public record Record(String url,String title,String text,String retrievedAt,String sha256,String sourceType,int status){}
 private final HttpClient client=HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
 public List<Record> crawl(URI start,int maxPages,int maxDepth) throws Exception {
  if(!Set.of("http","https").contains(start.getScheme())) throw new IllegalArgumentException("HTTP(S) only");
  ArrayDeque<Map.Entry<URI,Integer>> q=new ArrayDeque<>(); q.add(Map.entry(start,0)); Set<String> seen=new HashSet<>(); List<Record> out=new ArrayList<>();
  while(!q.isEmpty()&&out.size()<maxPages){var e=q.remove(); URI u=e.getKey(); int d=e.getValue(); if(!seen.add(u.toString())||d>maxDepth||!allowed(u))continue;
   HttpResponse<String> r=client.send(HttpRequest.newBuilder(u).header("User-Agent","ISO-Tool-Crawler/1.0").GET().build(),HttpResponse.BodyHandlers.ofString()); String body=r.body();
   String text=body.replaceAll("(?is)<script.*?</script>|<style.*?</style>"," ").replaceAll("<[^>]+>"," ").replaceAll("\\s+"," ").trim(); String title=match(body,"(?is)<title[^>]*>(.*?)</title>");
   out.add(new Record(u.toString(),title,text,Instant.now().toString(),sha256(body),"web",r.statusCode()));
   if(d<maxDepth) for(String href:links(body)){try{URI n=u.resolve(href.split("#")[0]); if(Objects.equals(n.getHost(),u.getHost()))q.add(Map.entry(n,d+1));}catch(Exception ignored){}}
  } return out;
 }
 private boolean allowed(URI u){try{URI r=new URI(u.getScheme(),u.getAuthority(),"/robots.txt",null,null); HttpResponse<String> x=client.send(HttpRequest.newBuilder(r).header("User-Agent","ISO-Tool-Crawler/1.0").GET().build(),HttpResponse.BodyHandlers.ofString()); if(x.statusCode()>=500)return false; for(String line:x.body().split("\\R")){String s=line.trim().toLowerCase(); if(s.startsWith("disallow:")&&s.substring(9).trim().equals("/"))return false;} return true;}catch(Exception e){return false;}}
 private static String match(String s,String rx){Matcher m=Pattern.compile(rx).matcher(s);return m.find()?m.group(1).replaceAll("\\s+"," ").trim():"";}
 private static List<String> links(String s){List<String> r=new ArrayList<>(); Matcher m=Pattern.compile("(?is)href\\s*=\\s*[\\\"']([^\\\"']+)").matcher(s); while(m.find())r.add(m.group(1)); return r;}
 private static String sha256(String s)throws Exception{byte[] b=MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8));StringBuilder x=new StringBuilder();for(byte v:b)x.append(String.format("%02x",v));return x.toString();}
 public Map<String,Object> trainRnn(List<double[]> sequences){double h=0,w=0;for(double[] seq:sequences)for(double x:seq){h=Math.tanh(x*.1+h*.05);w+=h*.01;}return Map.of("backend","builtin-rnn","samples",sequences.size(),"weight",w,"confidence",1.0/(1.0+Math.exp(-w)));}
 public Map<String,Object> knowledgeFromPath(Path root) throws IOException {long n=Files.walk(root).filter(Files::isRegularFile).filter(p->p.toString().matches(".*\\.(md|txt|rst|adoc|json|xml|ya?ml|toml|java|cpp|h|cs|py|sh)$")).count();return Map.of("repository",root.toString(),"documents",n,"evidence","repository");}
 public Map<String,Object> llmAdapter(String endpoint,String prompt){return Map.of("endpoint",endpoint,"prompt",prompt,"authorizationRequired",true,"mode","adapter-only");}
}

package org.chimera.thamudic;
import java.util.*;
public final class Thamudic {
  public static final int FIRST=0x10A80,LAST=0x10A9F;
  private Thamudic(){}
  public static boolean isThamudic(int cp){return cp>=FIRST&&cp<=LAST;}
  public static String extract(String text){StringBuilder b=new StringBuilder();text.codePoints().filter(Thamudic::isThamudic).forEach(b::appendCodePoint);return b.toString();}
  public static String transliterate(String text, Map<Integer,String> map){StringBuilder b=new StringBuilder();text.codePoints().forEach(cp->{if(map.containsKey(cp))b.append(map.get(cp));else if(isThamudic(cp))b.append('?');else if(Character.isWhitespace(cp))b.append(' ');});return b.toString();}
  public record Box(int x1,int y1,int x2,int y2){}
  public static List<Box> connectedComponents(byte[] pixels,int width,int height,int minArea){if(width<=0||height<=0||pixels.length!=width*height)throw new IllegalArgumentException("invalid image");boolean[] seen=new boolean[pixels.length];List<Box> out=new ArrayList<>();int[] dx={1,-1,0,0},dy={0,0,1,-1};for(int y=0;y<height;y++)for(int x=0;x<width;x++){int s=y*width+x;if(seen[s]||pixels[s]==0)continue;ArrayDeque<Integer> q=new ArrayDeque<>();q.add(s);seen[s]=true;int x1=x,x2=x,y1=y,y2=y,n=0;while(!q.isEmpty()){int p=q.remove(),cx=p%width,cy=p/width;n++;x1=Math.min(x1,cx);x2=Math.max(x2,cx);y1=Math.min(y1,cy);y2=Math.max(y2,cy);for(int k=0;k<4;k++){int nx=cx+dx[k],ny=cy+dy[k];if(nx>=0&&ny>=0&&nx<width&&ny<height){int z=ny*width+nx;if(!seen[z]&&pixels[z]!=0){seen[z]=true;q.add(z);}}}}if(n>=minArea)out.add(new Box(x1,y1,x2,y2));}return out;}
}

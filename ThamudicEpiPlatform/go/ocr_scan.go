package main

import("bytes";"fmt";"io";"mime/multipart";"net/http";"os")

func scan(path,engine string) error{f,e:=os.Open(path);if e!=nil{return e};defer f.Close();var b bytes.Buffer;w:=multipart.NewWriter(&b);p,e:=w.CreateFormFile("file",path);if e!=nil{return e};if _,e=io.Copy(p,f);e!=nil{return e};w.Close();r,e:=http.Post("http://127.0.0.1:8010/api/ocr/scan?engine="+engine,w.FormDataContentType(),&b);if e!=nil{return e};defer r.Body.Close();out,_:=io.ReadAll(r.Body);fmt.Println(string(out));if r.StatusCode>=300{return fmt.Errorf("ocr HTTP %d",r.StatusCode)};return nil}
func main(){if len(os.Args)<2{fmt.Println("usage: ocr_scan image [engine]");os.Exit(2)};engine:="auto";if len(os.Args)>2{engine=os.Args[2]};if e:=scan(os.Args[1],engine);e!=nil{panic(e)}}

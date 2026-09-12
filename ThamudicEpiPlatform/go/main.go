package main
import("fmt";"io";"net/http")
func main(){r,e:=http.Get("http://127.0.0.1:8010/api/objects");if e!=nil{panic(e)};defer r.Body.Close();b,_:=io.ReadAll(r.Body);fmt.Println(string(b))}

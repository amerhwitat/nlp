use reqwest::blocking::{Client, multipart};
use std::{env,fs};
fn main(){let a:Vec<String>=env::args().collect();if a.len()<2{eprintln!("usage: ocr_scan image [engine]");std::process::exit(2)}let engine=a.get(2).cloned().unwrap_or_else(||"auto".into());let data=fs::read(&a[1]).unwrap();let part=multipart::Part::bytes(data).file_name(a[1].clone()).mime_str("application/octet-stream").unwrap();let form=multipart::Form::new().part("file",part);let r=Client::new().post(format!("http://127.0.0.1:8010/api/ocr/scan?engine={engine}")).multipart(form).send().unwrap();println!("{}",r.text().unwrap());}

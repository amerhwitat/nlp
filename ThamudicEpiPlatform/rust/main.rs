fn main(){let body=reqwest::blocking::get("http://127.0.0.1:8010/api/objects").unwrap().text().unwrap();println!("{}",body);}

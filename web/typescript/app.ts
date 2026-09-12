export type TextMetrics={characters:number;words:number;lines:number};
export function metrics(text:string):TextMetrics{return {characters:[...text].length,words:text.trim()?text.trim().split(/\s+/u).length:0,lines:text?text.split(/\r?\n/u).length:0};}

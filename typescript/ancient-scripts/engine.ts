export interface Reading {
  script: string; text: string; transliteration?: string; unicodeText?: string;
  confidence: number; alternates: string[]; damage: string[]; provenance: string[];
}
export interface TranslationCandidate { text: string; confidence: number; evidence: string[]; }
export function rank(candidates: TranslationCandidate[]): TranslationCandidate[] {
  return [...candidates].sort((a,b) => b.confidence - a.confidence);
}
export function requiresReview(r: Reading): boolean { return r.confidence < 0.85 || r.damage.length > 0; }

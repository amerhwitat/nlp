#[derive(Clone, Debug)]
pub struct Reading {
    pub script: String,
    pub text: String,
    pub transliteration: Option<String>,
    pub confidence: f32,
    pub damage: Vec<String>,
    pub alternates: Vec<String>,
    pub provenance: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct TranslationCandidate { pub text: String, pub confidence: f32 }

pub fn rank(candidates: &mut [TranslationCandidate]) {
    candidates.sort_by(|a,b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
}

pub fn requires_review(r: &Reading) -> bool { r.confidence < 0.85 || !r.damage.is_empty() }

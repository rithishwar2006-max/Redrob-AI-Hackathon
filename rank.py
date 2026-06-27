"""
Redrob Hackathon — Intelligent Candidate Ranking System
Implements the 6-stage framework for Senior AI Engineer JD.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Constraints: CPU only, <5 min, <16 GB RAM, no network.
"""

import argparse
import csv
import gzip
import json
import math
import re
import sys
from datetime import date, datetime, timedelta
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

REFERENCE_DATE = date(2026, 6, 15)

APPROVED_HUBS = {
    "pune", "noida", "delhi ncr", "delhi", "new delhi", "gurgaon",
    "gurugram", "faridabad", "hyderabad", "mumbai", "bangalore", "bengaluru"
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcltech", "tech mahindra", "mphasis", "mindtree", "l&t infotech",
    "ltimindtree", "hexaware", "niit technologies", "zensar", "persistent",
    "cyient", "kpit", "birlasoft"
}

# Stage 0: Core IR/ML taxonomy
IR_ML_KEYWORDS = {
    "retrieval", "ranking", "embeddings", "embedding", "vector search",
    "matching systems", "recommendation", "recommendations", "recommend",
    "milvus", "pinecone", "qdrant", "weaviate", "faiss", "elasticsearch",
    "opensearch", "ndcg", "mrr", "map", "learning-to-rank", "ltr",
    "ann", "approximate nearest neighbor", "hybrid retrieval",
    "bm25", "dense retrieval", "sparse retrieval", "reranking", "re-ranking",
    "rerank", "semantic search", "vector database", "vector db", "vectordb",
    "information retrieval", "search engine", "search quality",
    "candidate matching", "job matching", "talent matching",
    "matching engine", "entity matching", "profile matching",
    "owned ranking", "ranking strategy", "ranking roadmap",
    "ranking quality", "retrieval quality", "ranking owner",
    # A human recruiter instantly recognizes these as core relevance/
    # retrieval/matching work, but none were previously covered:
    # "candidate generation" is standard recsys/search terminology for
    # the retrieval stage that produces a candidate set before ranking;
    # "relevance optimization" is a direct restatement of the JD's core
    # ask; "marketplace matching" is the two-sided-marketplace framing
    # of exactly the matching systems the JD describes.
    "candidate generation", "relevance optimization", "relevance pipeline",
    "marketplace matching", "matching marketplace platform",
    "sentence-transformers", "sentence transformers", "bi-encoder",
    "cross-encoder", "colbert", "dpr", "ance", "splade",
    "recall optimization", "precision recall", "search relevance",
    "query understanding", "query expansion", "bge", "e5 model",
    "chroma", "pgvector"
}

# Stage 2: Deep Domain Taxonomies
RECOMMENDER_TERMS = {
    "recommendation engine", "recommender", "collaborative filtering",
    "matrix factorization", "personalization", "feed ranking",
    "content ranking", "candidate recommendation", "item ranking",
    "personalized feed", "matching marketplace", "candidate discovery",
    "content recommendations", "user-item scoring"
}

LTR_TERMS = {
    "learning-to-rank", "learning to rank", "ltr", "lambdamart",
    "lightgbm ranker", "ranknet", "listnet", "xgboost ranking",
    "xgboost ranker", "catboost ranking", "gradient boosted ranking"
}

# Inject advanced terms into base keywords to pass Stage 0
IR_ML_KEYWORDS.update(RECOMMENDER_TERMS)
IR_ML_KEYWORDS.update(LTR_TERMS)

# Stage 0: Verb Taxonomy
BUILDER_VERBS = {
    "shipped", "built", "build", "designed", "implemented", "deployed",
    "architected", "productionized", "created", "developed", "wrote",
    "established", "launched", "delivered", "authored", "constructed",
    "engineered", "coded", "programmed", "released", "published"
}
LEADER_VERBS = {
    "led", "drove", "owned", "managed", "directed", "oversaw",
    "supervised", "spearheaded", "championed", "initiated", "coordinated",
    "mentored", "scaled"
}
PASSIVE_VERBS = {
    "contributed", "supported", "assisted", "researched", "investigated",
    "helped", "participated", "involved", "worked on", "collaborated"
}

# Stage 4: Speciality Penalties/Keywords
GENAI_KEYWORDS = {
    "langchain", "llamaindex", "llama_index", "openai api", "chatgpt api",
    "gpt-4 api", "gpt4 api", "rag pipeline", "rag application",
    "langsmith", "flowise", "haystack", "promptflow"
}

BUSINESS_METRICS = {
    "ctr", "click-through", "click through", "conversion",
    "engagement lift", "search success", "revenue", "retention",
    "bounce rate", "active users"
}

EMBEDDING_OPS = {
    "drift", "index refresh", "re-indexing", "reindexing",
    "retrieval quality", "retrieval regression", "embedding monitoring",
    "vector refresh", "index rebuild"
}

# Candidates sometimes explicitly disclaim ownership of exactly the thing
# a bare keyword match would otherwise credit them with (e.g. "production
# deployment was handled by the platform team", "my own modeling work was
# secondary"). Originally this lived only inside apply_hard_gates' rescue-
# substance check; promoted to module level so extract_best_evidence can
# also respect it when deciding whether to label a citation "in
# production" -- a hardcoded "production" claim in an evidence bucket
# label is just as much a fabrication risk as a keyword match that
# ignores context, and was found doing exactly that (CAND_0010541,
# CAND_0064130, CAND_0067717 all disclaim production ownership in the
# very same sentence the "built a production recommendation/re-ranking
# system" label was citing).
OWNERSHIP_DISCLAIMER_PATTERNS = [
    "was handled by", "handled by the platform team", "handled by another team",
    "deployment was handled", "someone else deployed", "platform team deployed",
    "not my responsibility", "wasn't my role", "was not my role",
    "secondary", "my role was more on the modeling side",
    "my own modeling work was secondary",
    # Team-participation phrasing that signals the candidate was NOT the
    # primary owner/builder, even though it often appears alongside a
    # genuine keyword match (e.g. "worked with a team that built a
    # recommendation system" contains "recommendation system" but is
    # explicitly NOT a first-person ownership claim).
    "worked with a team that", "as part of a team that", "alongside a team that",
    "team that built", "team that designed", "team that owned",
]

# Used by llm_wrapper_check to distinguish a genuine GenAI-wrapper candidate
# (LangChain/OpenAI API calls with no real retrieval infra or eval rigor
# behind them) from someone who built a real RAG/search system on top of a
# vector DB with proper evaluation -- the latter should NOT be penalized.
VECTORDB_EVAL_KEYWORDS = {
    "milvus", "pinecone", "qdrant", "weaviate", "faiss", "chroma",
    "opensearch", "elasticsearch", "pgvector", "ndcg", "mrr", "map",
    "a/b test", "ab test", "offline eval", "online eval",
    "evaluation framework", "ranking evaluation", "retrieval eval"
}

# Stage 2: Feature weights (Technical 70, Culture 20, Hireability 10)
TECH_WEIGHTS = {
    "relevance_ownership": 20,
    "production_ml": 15,
    "evaluation_rigor": 10,
    "product_ownership": 10,
    "retrieval_architecture": 7,
    "career_environment": 5,
    "production_scale": 2,
    "search_infra": 1,
}

CULTURE_WEIGHTS = {
    "founding_team_fit": 10,
    "python_depth": 5,
    "distributed_systems": 3,
    "external_validation": 2,
}

HIRE_WEIGHTS = {
    "availability_timeline": 6,
    "pipeline_friction": 4,
}

# ── Stage 0: Extraction Primitives ───────────────────────────────────────────

def extract_text(candidate: dict) -> str:
    """Concatenate all text fields for keyword scanning."""
    parts = []
    p = candidate.get("profile", {})
    parts.append(p.get("summary", ""))
    parts.append(p.get("headline", ""))
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
    for sk in candidate.get("skills", []):
        parts.append(sk.get("name", ""))
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))
    return " ".join(parts).lower()


def extract_career_text(candidate: dict) -> str:
    """Only career history text (source weight 1.0)."""
    parts = []
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
    return " ".join(parts).lower()


def extract_skills_text(candidate: dict) -> str:
    """Only skills section text (source weight 0.5)."""
    return " ".join(sk.get("name", "") for sk in candidate.get("skills", [])).lower()


def contains_exact_match(keyword: str, text: str) -> bool:
    """
    Universal word-boundary-safe matching helper to prevent substring
    traps. A bare short term checked via raw `in` can match as a
    fragment inside unrelated words -- e.g. "ann" in "planning", "map"
    in "roadmap", "bert" in "robert", "led" in "failed"/"handled"/
    "scaled"/"knowledge", "ctr" in "doctrine"/"electric". This was
    previously only fixed for IR_ML_KEYWORDS (via text_contains_ir_keyword)
    but the same vulnerability existed unprotected in BUSINESS_METRICS,
    arch_terms, eval_terms, pre_llm_terms, and ownership_verbs across
    multiple scoring functions, silently inflating scores for candidates
    whose resumes happened to contain these completely unrelated words.
    Multi-word/hyphenated phrases are left on substring matching since
    they're inherently specific enough that false-positive risk is
    negligible.
    """
    if " " in keyword or "-" in keyword or keyword.startswith("<"):
        return keyword in text
    return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))


def text_contains_ir_keyword(text: str) -> bool:
    """
    Shared, word-boundary-aware check for whether IR_ML_KEYWORDS appear
    in a piece of text. Short, single-word terms (e.g. "ranking",
    "ance") use word-boundary regex matching rather than bare substring
    checks -- a bare substring match on a single word is the highest-risk
    case for false positives. The worst real example found: "ance" (a
    real retrieval technique acronym) was matching as a substring inside
    extremely common, unrelated words like "relevance", "performance",
    "governance", "maintenance", "compliance" -- meaning a candidate
    description like "Built relevance optimization platform" would
    falsely register as containing IR evidence purely from the word
    "relevance" itself, not anything about retrieval.
    Multi-word phrases (e.g. "ranking system", "learning-to-rank") are
    left on substring matching since they're inherently specific enough
    that false-positive risk is negligible.
    This is the SINGLE shared implementation -- it must be the only
    place this matching logic lives, since duplicating it (as
    find_ir_evidence_job previously did with its own unprotected
    "kw in desc" check) silently reintroduces the same bug in a second
    location even after the first occurrence was fixed.
    """
    text_lower = text.lower()
    for kw in IR_ML_KEYWORDS:
        if " " in kw or "-" in kw:
            if kw in text_lower:
                return True
        else:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return True
    return False


def has_ir_ml_evidence(candidate: dict) -> bool:
    """Stage 0: has_IR_ML_Search_evidence = True if any IR keyword found anywhere."""
    full_text = extract_text(candidate)
    return text_contains_ir_keyword(full_text)


def ir_evidence_in_career(candidate: dict) -> bool:
    """
    Whether IR evidence exists specifically in career history, AND under a
    job title that's consistent with that evidence (guards against the
    dataset's scrambled title/description trap -- e.g. a 'Computer Vision
    Engineer' entry whose description is actually about recommendation
    systems, which is not reliable evidence of real IR ownership).
    """
    ir_job, is_consistent = find_ir_evidence_job(candidate)
    return ir_job is not None and is_consistent


def has_python(candidate: dict) -> bool:
    """Check Python evidence in career history or skills."""
    career_text = extract_career_text(candidate)
    if "python" in career_text or "pyspark" in career_text or "pytest" in career_text:
        return True
    for sk in candidate.get("skills", []):
        if "python" in sk.get("name", "").lower():
            return True
    return False


def has_recent_code(candidate: dict, months: int = 18) -> bool:
    software_eng_titles = {
        "software engineer", "software developer", "ml engineer",
        "machine learning engineer", "ai engineer", "data engineer",
        "data scientist", "backend engineer", "frontend engineer",
        "full stack", "fullstack", "platform engineer", "sde", "swe",
        "research engineer", "applied scientist", "nlp engineer",
        "computer vision", "deep learning", "ai researcher",
        "site reliability", "sre", "cloud engineer", "recommendation",
        "search engineer", "ranking engineer", "retrieval engineer",
        "distributed systems", "mlops"
    }
    
    lead_titles = {"lead", "manager", "architect", "head", "director", "vp", "principal", "staff"}

    def is_coding_role(title: str, desc: str) -> bool:
        t = title.lower()
        if any(phrase in t for phrase in software_eng_titles):
            return True
        if any(phrase in t for phrase in lead_titles):
            return any(
                re.search(r'\b' + re.escape(v) + r'\b', desc.lower())
                for v in BUILDER_VERBS
            )
        return False

    # Calculate exactly 18 months (~540 days) ago
    cutoff = REFERENCE_DATE - timedelta(days=months * 30)

    for job in candidate.get("career_history", []):
        start = job.get("start_date", "")
        end = job.get("end_date", None)
        
        if start:
            try:
                start_d = date.fromisoformat(start[:10]) # ensure YYYY-MM-DD
            except ValueError:
                continue
            
            # If the job has an end date, check if it falls inside our 18-month window
            if end:
                try:
                    end_d = date.fromisoformat(end[:10])
                except ValueError:
                    end_d = REFERENCE_DATE
                
                if end_d >= cutoff and start_d <= REFERENCE_DATE:
                    if is_coding_role(job.get("title", ""), job.get("description", "")):
                        return True
            # If the job is ongoing, just ensure it started before our reference date
            else:
                if start_d <= REFERENCE_DATE:
                    if is_coding_role(job.get("title", ""), job.get("description", "")):
                        return True

    return False

def extract_verbs(candidate: dict):
    """Count Builder / Leader / Passive verbs, supporting multi-word phrases."""
    career_text = extract_career_text(candidate)
    
    def count_occurrences(verb_set):
        count = 0
        for v in verb_set:
            # \b ensures we match exact whole words/phrases, not substrings
            count += len(re.findall(r'\b' + re.escape(v) + r'\b', career_text))
        return count

    builder = count_occurrences(BUILDER_VERBS)
    leader = count_occurrences(LEADER_VERBS)
    passive = count_occurrences(PASSIVE_VERBS)
    total = builder + leader + passive
    
    return builder, leader, passive, total


def shipping_velocity(candidate: dict) -> float:
    builder, leader, passive, total = extract_verbs(candidate)
    if total == 0:
        return 0.0
    return (1.0 * builder + 0.5 * leader) / total


def compute_title_tiers(candidate: dict) -> list:
    """Map each career entry title to tier integer."""
    tier_map = {
        1: {"junior", "associate", "intern", "trainee"},
        2: {"engineer", "developer", "data scientist", "analyst", "designer",
            "programmer", "consultant", "specialist"},
        3: {"senior", "lead", "tech lead", "team lead", "principal engineer"},
        4: {"staff", "principal", "architect", "distinguished"},
        5: {"director", "head", "vp", "vice president", "cto", "cpo"}
    }

    def classify(title: str) -> int:
        t = title.lower()
        for tier in [5, 4, 3, 2, 1]:
            for keyword in tier_map[tier]:
                if keyword in t:
                    return tier
        return 2  # default

    tiers = []
    for job in sorted(
        candidate.get("career_history", []),
        key=lambda j: j.get("start_date", "")
    ):
        tiers.append(classify(job.get("title", "")))
    return tiers


def title_chaser_check(candidate: dict) -> bool:
    """Returns True if Title-Chaser penalty should apply."""
    jobs = candidate.get("career_history", [])
    if len(jobs) < 2:
        return False

    # Average tenure check
    tenures = [j.get("duration_months", 0) for j in jobs if not j.get("is_current", False)]
    if not tenures:
        return False
    avg_tenure = sum(tenures) / len(tenures)
    if avg_tenure >= 18:
        return False  # tenure OK, no penalty

    # Check title inflation
    tiers = compute_title_tiers(candidate)
    for i in range(1, len(tiers)):
        if tiers[i] > tiers[i - 1]:
            return True
    return False


def is_consulting_only(candidate: dict) -> bool:
    """Returns True if entire career is at known consulting firms."""
    jobs = candidate.get("career_history", [])
    if not jobs:
        return False
    for job in jobs:
        company = job.get("company", "").lower()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            return False
    return True


def has_product_ownership_verbs(candidate: dict) -> bool:
    """Check for architect/drove/owned/scaled in career history."""
    ownership_verbs = {"architected", "drove", "owned", "scaled", "spearheaded",
                       "designed", "established", "launched", "led"}
    career_text = extract_career_text(candidate)
    words = set(re.findall(r'\b\w+\b', career_text))
    return bool(words & ownership_verbs)


def llm_wrapper_check(candidate: dict) -> bool:
    """Returns True if LLM wrapper penalty should apply (all 3 conditions)."""
    # Condition 1: First ML role after Jan 2023
    ml_titles = {"machine learning", "ai ", "nlp", "data scientist",
                 "deep learning", "llm", "gen ai", "generative"}

    def text_has_ml_title_signal(text: str) -> bool:
        if re.search(r'\bml\b', text):
            return True
        return any(kw in text for kw in ml_titles)

    jobs_sorted = sorted(
        candidate.get("career_history", []),
        key=lambda j: j.get("start_date", "")
    )
    first_ml_date = None
    for job in jobs_sorted:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        if text_has_ml_title_signal(title) or text_has_ml_title_signal(desc):
            first_ml_date = job.get("start_date", "")
            break

    if first_ml_date is None:
        return False
    if first_ml_date <= "2023-01-01":
        return False  # Had ML before Jan 2023 — not a wrapper

    # Condition 2: No pre-LLM IR evidence
    if ir_evidence_in_career(candidate):
        return False

    # Condition 3: GenAI keyword count >= 3 AND no VectorDB/Eval evidence
    full_text = extract_text(candidate)
    genai_count = sum(1 for kw in GENAI_KEYWORDS if kw in full_text)
    if genai_count < 3:
        return False
    has_vectordb_eval = any(kw in full_text for kw in VECTORDB_EVAL_KEYWORDS)
    if has_vectordb_eval:
        return False  # Has real eval/vectordb — not a pure wrapper

    return True


def research_only_check(candidate: dict) -> bool:
    """Returns True if all experience is academic/research with no production."""
    research_indicators = {"research", "lab", "laboratory", "academia",
                          "phd candidate", "postdoc", "university", "institute"}
    production_indicators = {"production", "deployed", "shipped", "users",
                             "product", "startup", "company", "clients"}
    career_text = extract_career_text(candidate)
    has_research = any(kw in career_text for kw in research_indicators)
    has_production = any(kw in career_text for kw in production_indicators)
    return has_research and not has_production


def compute_skill_inflation_penalty(candidate: dict) -> float:
    """Stage 4: Compute total skill inflation penalty strictly per-skill."""
    proficiency_to_expected = {
        "beginner": 40,
        "intermediate": 65,
        "advanced": 80,
        "expert": 90
    }
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if not assessment_scores:
        return 0.0

    total_penalty = 0.0

    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "intermediate")
        if name not in assessment_scores:
            continue
            
        E = proficiency_to_expected.get(proficiency, 65)
        A = assessment_scores[name]
        
        # Only accumulate penalties. A high score in Skill B does not excuse lying about Skill A.
        if A < (E - 15):
            penalty = ((E - A - 15) / E) * 0.15
            total_penalty += penalty

    return min(1.0, total_penalty)  # Cap absolute penalty at 100%

def compute_tenure_stability(candidate: dict) -> tuple:
    """Returns (avg_tenure_months, is_unstable)."""
    jobs = candidate.get("career_history", [])
    tenures = [j.get("duration_months", 0) for j in jobs if not j.get("is_current", False)]
    if not tenures:
        return 0, False
    avg = sum(tenures) / len(tenures)
    return avg, avg < 18


# ── Stage 1: Hard Gates ───────────────────────────────────────────────────────

def apply_hard_gates(candidate: dict) -> tuple:
    """
    Returns (passed: bool, route: str, flags: list, continue_scoring: bool)
    route: 'pass', 'hard_reject', 'alternate_pipeline', 'manual_review'
    """
    flags = []
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    country = profile.get("country", "").strip()
    city = profile.get("location", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    open_to_work = signals.get("open_to_work_flag", False)

    # Check last active
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = date.fromisoformat(last_active_str)
        days_inactive = (REFERENCE_DATE - last_active).days
    except Exception:
        days_inactive = 999

    # Logistics: International
    if country.lower() not in ("india", "in"):
        flags.append(f"Requires Visa Review ({profile.get('location', '')}, {country})")
        # Continue scoring but flag

    # Logistics: Passive/Ghost (non-halting)
    if not open_to_work and days_inactive > 180:
        flags.append("Passive/Unavailable — Low Pipeline Priority")

    # Logistics: Domestic Relocation Block
    if country.lower() in ("india", "in"):
        city_lower = city.lower()
        in_hub = any(hub in city_lower for hub in APPROVED_HUBS)
        if not in_hub and not willing_to_relocate:
            return False, "hard_reject", flags, False

    # Domain: Absolute Zero
    if not has_python(candidate):
        return False, "hard_reject", flags, False
    if not has_recent_code(candidate):
        return False, "hard_reject", flags, False

    # Domain: IR Absence
    if not has_ir_ml_evidence(candidate):
        return False, "alternate_pipeline", flags, False

    # Domain: Weak Signal Trap
    cri = compute_core_relevance_index(candidate)
    relevance_score = score_relevance_ownership(candidate)

    career_text = extract_career_text(candidate)
    has_recommender = any(term in career_text for term in RECOMMENDER_TERMS)
    has_deep_search = any(term in career_text for term in [
        "ranking", "retrieval", "search engine",
        # The JD explicitly lists "matching systems" as a core
        # requirement alongside ranking/retrieval -- the rescue path
        # previously had no way to recognize genuine matching-system
        # evidence (e.g. "candidate matching", "dual-tower architecture")
        # even when score_relevance_ownership now credits it.
        "matching system", "candidate matching", "job matching",
        "talent matching", "matching engine", "entity matching", "profile matching",
        "dual-tower", "dual tower", "two-tower", "two tower",
        "candidate generation", "relevance optimization", "relevance pipeline",
        "marketplace matching"
    ])

    # The rescue path exists to avoid over-rejecting candidates whose CRI
    # is dragged down by the (eval rigor + retrieval architecture) terms
    # in the CRI formula even though they have real recommender/search
    # ownership. But relevance_score alone can be high purely from
    # keyword DENSITY (repeating "recommendation", "recommender",
    # "personalization" etc.) with zero production evidence and zero
    # quantified outcome -- exactly the keyword-stuffing trap the JD
    # warns about. Require genuine substance beyond keyword presence:
    # either production deployment language or a quantified outcome
    # (the same signals score_product_ownership and score_production_ml
    # already treat as markers of real, not just claimed, experience.
    # Detecting the bare word "production"/"deployed" is not enough --
    # candidates sometimes explicitly disclaim ownership of exactly this
    # (e.g. "production deployment was handled by the platform team",
    # "someone else deployed it"). A disclaimed mention should NOT count
    # as substance; it's the opposite signal. Check for ownership-negating
    # phrases co-occurring with the substance keyword and exclude those.
    has_ownership_disclaimer = any(p in career_text for p in OWNERSHIP_DISCLAIMER_PATTERNS)

    has_deploy_evidence = (
        any(t in career_text for t in ["production", "deployed", "inference", "shipped"])
        and not has_ownership_disclaimer
    )
    has_quantified_outcome = bool(
        re.search(r'\d+\s*%|\d+x\s|improved|reduced|increased', career_text)
    )
    has_substance = has_deploy_evidence or has_quantified_outcome

    # Only hard-fail candidates below 35 IF they lack core terms AND substantial relevance
    if cri < 35:
        if cri < 25:
            return False, "alternate_pipeline", flags, True
        # Require keyword presence AND a baseline relevance score AND
        # real substance (production/quantified evidence) to rescue --
        # closes the keyword-density-only loophole.
        elif not ((has_recommender or has_deep_search) and relevance_score > 40
                  and has_substance):
            return False, "alternate_pipeline", flags, True
        else:
            flags.append("Rescued: Foundational Recommender/Search Evidence")

    return True, "pass", flags, True

def get_recency_multiplier(job: dict) -> float:
    """Returns a multiplier based on the recency of the job."""
    if job.get("is_current"):
        return 1.0
        
    end_date_str = job.get("end_date", "")
    if not end_date_str:
        return 1.0 # Assume current if no end date
        
    try:
        end_date = date.fromisoformat(end_date_str[:10])
        days_ago = (REFERENCE_DATE - end_date).days
        years_ago = days_ago / 365.25
        
        if years_ago <= 3:
            return 0.9
        elif years_ago <= 6:
            return 0.7
        else:
            return 0.5
    except ValueError:
        return 0.7 # Default to medium if date parsing fails

# ── Stage 2 & 3: Raw Fit Score ────────────────────────────────────────────────

def score_relevance_ownership(candidate: dict) -> float:
    """
    Built ranking/search/recommendation — heavily rewards end-to-end
    ownership & LTR. Scans aggregated text across ALL title-consistent
    career_history entries (not just one), so evidence spread across
    multiple legitimate roles (e.g. eval rigor in the current role,
    ranking architecture in a past role) isn't artificially discarded.
    Still requires at least one trustworthy (non-contradicted) job to
    exist at all, so the CV/scrambled-title trap remains caught.
    """
    ir_job, is_consistent = find_ir_evidence_job(candidate)
    if ir_job is None or not is_consistent:
        return 0.0

    career_text = aggregate_trustworthy_career_text(candidate)
    score = 0.0

    strong_terms = {
        "ranking system", "search system", "recommendation system",
        "matching system", "retrieval system", "ranking model",
        "semantic search", "candidate ranking", "search relevance",
        # The JD's own language is "ranking, retrieval, and matching
        # systems" -- "candidate matching" and dual/two-tower
        # architectures are standard real-world terms for exactly this
        # kind of system but were previously absent, causing genuine
        # matching-system candidates (the JD's explicit use case) to
        # score poorly purely from a keyword-list gap, not weak evidence.
        "candidate matching", "job matching", "talent matching",
        "matching engine", "entity matching", "profile matching",
        "dual-tower", "dual tower", "two-tower", "two tower",
        # The JD's central framing is "own the ranking layer" -- these
        # phrases directly signal ranking OWNERSHIP (the JD's strongest
        # emphasis) but previously scored zero despite being some of the
        # most natural ways to phrase exactly that ownership on a resume.
        "owned ranking", "ranking strategy", "ranking roadmap",
        "ranking quality", "retrieval quality", "ranking owner",
        "owns ranking", "own the ranking",
        # Same vocabulary gap as in IR_ML_KEYWORDS above -- these are
        # natural, common phrasings for retrieval/ranking/matching work
        # that previously scored zero purely from absence, not weak
        # evidence (e.g. "Designed candidate generation layer", "Built
        # relevance optimization platform").
        "candidate generation", "relevance optimization", "relevance pipeline",
        "marketplace matching", "matching marketplace platform",
        "owned retrieval", "retrieval strategy", "retrieval owner",
        "owns retrieval", "own the retrieval",
        # "ranking layer" alone is one of the most common GENUINE phrasings
        # in this entire dataset -- it appears in dozens of candidates'
        # career history (e.g. "owned the ranking layer", "designed the
        # ranking layer") but was never itself added to strong_terms;
        # only more specific suffixed variants (ranking strategy/roadmap/
        # quality) were. A candidate who writes natural prose ("Designed
        # the ranking layer for the company's flagship product... I owned
        # all three") rather than keyword-stuffed bullet points was
        # scoring near zero on relevance ownership despite unambiguous,
        # senior-level ranking ownership -- exactly the false-negative
        # risk a human recruiter would never make.
        "ranking layer", "search and discovery", "discovery experience",
        "owned the search", "designed the search",
    }
    
    # 1. Base Feature Extraction (Unique matched terms to prevent gaming)
    strong_matched = {t for t in strong_terms if t in career_text}
    recommender_matched = {t for t in RECOMMENDER_TERMS if t in career_text}
    ltr_matched = {t for t in LTR_TERMS if t in career_text}
    medium_matched = {t for t in ["embeddings", "vector search", "relevance"] if t in career_text}

    # 2. Base keyword-evidence score, CAPPED at 35.
    # A single unsubstantiated sentence (e.g. "Implemented LambdaMART
    # ranking model") was previously able to score 55+ purely from
    # keyword presence across two buckets, before any deployment, eval,
    # ownership, or outcome evidence was checked -- more than half the
    # maximum score for one bare technical noun phrase with zero
    # demonstrated substance. Capping the keyword-only base at 35 means
    # a candidate needs at least one substance signal (below) to clear
    # the halfway mark, while genuine full-stack ownership can still
    # reach 80-100 via the substance bonuses.
    keyword_base = (min(70, len(strong_matched) * 25)
                     + min(50, len(recommender_matched) * 20)
                     + min(50, len(ltr_matched) * 30)
                     + min(20, len(medium_matched) * 10))
    has_participation_disclaimer = any(p in career_text for p in OWNERSHIP_DISCLAIMER_PATTERNS)
    capped_keyword_base = min(35, keyword_base)
    if has_participation_disclaimer:
        # "Worked with a team that built X" still demonstrates SOME
        # exposure to X, but is explicitly not a first-person ownership
        # claim -- discount the keyword-only base rather than zero it out.
        capped_keyword_base *= 0.5
    score += capped_keyword_base

    # 3. End-to-End Ownership & Specific Domain Ownership -- substance
    # bonuses. These were already gated behind co-occurring evidence
    # (deploy+eval together, or an ownership verb), so they're left as
    # the path to scores above the 35-point keyword-only ceiling.
    has_retrieval = (len(strong_matched) > 0 or len(recommender_matched) > 0 or len(ltr_matched) > 0)
    has_deploy = any(t in career_text for t in ["production", "deployed", "inference"])
    has_eval = any(t in career_text for t in ["ndcg", "mrr", "a/b test", "ab test", "offline eval"])
    # Require the outcome verb to co-occur with an actual percentage/
    # multiplier OR an IR-relevant metric name -- the bare verb alone
    # ("improved documentation", "improved onboarding") was previously
    # enough to earn the outcome-substance bonus regardless of relevance.
    # Two grammatical orders are covered: verb-first ("improved CTR by
    # 5%") and metric-first ("CTR improved from 0.12 to 0.17"), plus
    # percentage-point notation ("+4.2pp") which a bare \d+\s*% pattern
    # doesn't catch.
    relevant_metric_names = (
        r'ctr|click-through|ndcg|mrr|map|relevance|engagement|revenue|'
        r'retention|conversion|latency|recall|precision|search|ranking|'
        r'retrieval|recommendation'
    )
    has_outcome = bool(re.search(
        r'\d+\s*%|\d+x\s|[+-]?\d+(\.\d+)?\s*pp\b|'
        rf'(improved|reduced|increased|decreased|boosted|grew|lifted|cut)\s+'
        rf'[\w\-\s]{{0,30}}?({relevant_metric_names})|'
        rf'({relevant_metric_names})[\w\-\s]{{0,30}}?'
        rf'(improved|reduced|increased|decreased|boosted|grew|lifted|cut|'
        rf'went from|from \d)',
        career_text
    ))

    if has_retrieval and has_deploy:
        score += 20  # Deployment substance bonus
    if has_retrieval and has_eval:
        score += 20  # Evaluation substance bonus
    if has_retrieval and has_outcome:
        score += 10  # Quantified outcome substance bonus

    if has_retrieval and has_deploy and has_eval:
        score += 10  # Full-stack AI ownership bonus (on top of the individual bonuses above)

    ownership_verbs = {"architected", "drove", "owned", "designed", "led", "shipped"}
    has_ownership_verb = any(v in career_text for v in ownership_verbs)

    if has_retrieval and has_ownership_verb:
        score += 15  # Explicitly owned the retrieval system

    # ... (rest of score_relevance_ownership) ...
    
    # 4. Apply Recency Multiplier
    recency_multiplier = get_recency_multiplier(ir_job)
    final_relevance_score = score * recency_multiplier

    return min(100.0, final_relevance_score)

def score_production_ml(candidate: dict) -> float:
    """Model deployment, pre-2023 ML exposure (immune to non-technical title traps)."""
    score = 0.0
    prod_count = 0
    pre_llm_count = 0
    pre_2023_ml = False

    production_terms = [
        "production", "deployed", "deployment", "inference", "serving",
        "latency", "throughput", "drift", "monitoring", "model drift",
        "a/b test", "canary", "shadow mode"
    ]
    pre_llm_terms = [
        "xgboost", "lightgbm", "sklearn", "scikit-learn", "random forest",
        "gradient boosting", "svm", "logistic regression",
        "collaborative filtering", "matrix factorization", "word2vec",
        "fasttext", "bert", "roberta", "sentence-bert"
    ]
    
    non_technical_skip = {
        "hr manager", "accountant", "graphic designer", "content writer",
        "sales executive", "business analyst", "project manager",
        "operations manager", "marketing manager", "civil engineer",
        "mechanical engineer", "customer support", "customer success",
        "recruiter", "talent acquisition", "office manager", "executive assistant"
    }

    # Production terms alone are too generic -- DevOps/SRE/infra roles
    # legitimately use "production", "deployment", "latency", "monitoring"
    # constantly with zero connection to ML models, ranking, or retrieval.
    # Require co-occurrence with BOTH ML/model context AND a builder/
    # ownership verb in the SAME job description before counting
    # production-term hits from that job. ML-context alone wasn't
    # enough: "Production ML model with Docker, Kubernetes, Airflow,
    # monitoring, CI/CD, MLOps, canary, feature store, and inference all
    # handled" satisfies has_ml_context via the bare word "model" and
    # then racks up 7+ distinct production-term hits with zero verb
    # demonstrating the candidate actually built or owned any of it.
    ml_context_terms = [
        "model", "ml", "machine learning", "neural", "prediction",
        "ranking", "retrieval", "recommendation", "embedding",
        "classifier", "regression", "training", "inference pipeline"
    ]
    builder_verb_terms = [
        "built", "build", "shipped", "deployed", "owned", "designed",
        "architected", "implemented", "developed", "created",
        "launched", "established"
    ]

    for job in candidate.get("career_history", []):
        title = job.get("title", "").lower()
        
        # Skip jobs where the title makes ML claims inherently suspicious (the dataset trap)
        # but allow CV/Audio titles to claim valid production ML deployment experience.
        if any(kw in title for kw in non_technical_skip):
            continue
            
        desc = job.get("description", "").lower()

        has_ml_context = (
            re.search(r'\bml\b', desc) is not None
            or any(t in desc for t in ml_context_terms if t != "ml")
        )
        has_builder_verb = any(t in desc for t in builder_verb_terms)
        if has_ml_context and has_builder_verb:
            # Cap distinct production-term credit per job at 4, so a
            # single builder verb doesn't unlock full credit for an
            # arbitrarily long buzzword list in the same sentence (e.g.
            # "Built a production ML model with Docker, Kubernetes,
            # Airflow, monitoring, CI/CD, MLOps, canary, feature store,
            # inference" -- one verb, nine nouns -- should not score 9x
            # higher than someone who genuinely mentions 2-3 in context).
            job_prod_terms = sum(1 for t in production_terms if contains_exact_match(t, desc))
            prod_count += min(4, job_prod_terms)
        pre_llm_count += sum(1 for t in pre_llm_terms if contains_exact_match(t, desc))

        if job.get("start_date", "") <= "2023-01-01":
            has_ml_word = re.search(r'\bml\b', desc) is not None
            if has_ml_word or any(t in desc for t in ["model", "machine learning", "neural", "prediction"]):
                pre_2023_ml = True

    score += min(50, prod_count * 12)
    score += min(30, pre_llm_count * 15)
    if pre_2023_ml:
        score += 20
        
    return min(100.0, score)

def score_evaluation_rigor(candidate: dict) -> float:
    """
    Offline/online eval loops: NDCG, MAP, MRR, A/B testing.
    Restricted primarily to career history (source weight 1.0), with
    skills-section-only mentions counted at a heavily reduced weight
    (0.3x) rather than full credit. Previously this scanned extract_text()
    (career history + skills + headline + summary combined) with no
    source distinction, so a candidate listing 'NDCG', 'MRR', 'MAP' as
    bare skills with zero career-history substantiation could score
    3 * 25 = 75/100 -- exactly the keyword-stuffing-in-skills pattern the
    JD explicitly warns about ("all the AI keywords listed as skills...
    is not a fit"). This brings evaluation_rigor in line with the same
    career-history-first principle already applied to relevance_ownership.
    """
    career_text = aggregate_trustworthy_career_text(candidate)
    skills_text = extract_skills_text(candidate)

    eval_terms = [
        "ndcg", "mrr", "map", "mean average precision",
        "a/b test", "ab test", "online experiment", "offline eval",
        "evaluation framework", "ranking eval", "precision@", "recall@",
        "click-through rate", "ctr", "engagement metric",
        "relevance judgment", "human eval", "annotation",
        # The JD explicitly names "offline-to-online correlation" as a
        # required evaluation skill. A candidate can describe this exact
        # rigor in plain prose ("offline metrics that correlated with
        # online engagement") without using any of the terms above --
        # that's not weaker evidence, it's the same evidence in
        # different words, and was previously scoring as if it barely
        # existed.
        "offline-online correlation", "offline to online correlation",
        "offline metrics", "correlated with online", "offline-to-online",
        "evaluation methodology", "eval methodology",
    ]
    career_count = sum(1 for t in eval_terms if contains_exact_match(t, career_text))
    # Only count a skills-section term if it ISN'T already counted from
    # career history, to avoid double-counting the same evidence twice.
    skills_only_count = sum(
        1 for t in eval_terms
        if contains_exact_match(t, skills_text) and not contains_exact_match(t, career_text)
    )
    score = career_count * 25.0 + skills_only_count * 25.0 * 0.3
    return min(100.0, score)


def score_product_ownership(candidate: dict) -> float:
    """Architected/Drove × Platform Scope × Quantified Outcomes × Business Metrics."""
    career_text = extract_career_text(candidate)
    score = 0.0

    ownership_verbs = ["architected", "drove", "owned", "designed", "led", "shipped"]
    platform_terms = ["platform", "system", "infrastructure", "pipeline"]
    
    has_quantified = bool(re.search(r'\d+\s*%|\d+x\s|improved|reduced|increased', career_text))
    metric_count = sum(1 for m in BUSINESS_METRICS if contains_exact_match(m, career_text))

    verb_count = sum(1 for v in ownership_verbs if contains_exact_match(v, career_text))
    platform_count = sum(1 for p in platform_terms if contains_exact_match(p, career_text))

    score += min(30, verb_count * 8)
    score += min(20, platform_count * 8)
    
    if has_quantified:
        score += 20
        
    # Massive boost for explicitly mentioning business/engagement metrics
    if metric_count > 0:
        score += min(30, metric_count * 15)

    return min(100.0, score)

def score_retrieval_architecture(candidate: dict) -> float:
    """
    ANN design, hybrid retrieval, recall optimization.
    Restricted to career history (not skills section -- "familiar with
    ANN, BM25, HNSW" as a bare skills listing shouldn't score the same
    as someone who designed retrieval architecture). Also requires an
    ownership verb or production context to reach the upper half of the
    range: five architecture nouns with zero design/ownership language
    previously maxed out at 100/100, indistinguishable from genuine
    architecture ownership.
    """
    career_text = aggregate_trustworthy_career_text(candidate)
    arch_terms = [
        "ann", "approximate nearest neighbor", "hnsw", "ivf",
        "hybrid retrieval", "hybrid search", "dense-sparse", "bm25",
        "recall optimization", "inverted index", "sharding",
        "query latency", "p99", "vector index"
    ]
    matched = {t for t in arch_terms if contains_exact_match(t, career_text)}
    count = len(matched)

    base = min(50, count * 15)

    ownership_verbs = {"architected", "designed", "built", "owned",
                       "implemented", "optimized", "tuned"}
    has_ownership = any(v in career_text for v in ownership_verbs)
    has_production_context = any(t in career_text for t in
                                  ["production", "deployed", "serving", "scale"])

    substance_bonus = 0
    if count > 0 and has_ownership:
        substance_bonus += 25
    if count > 0 and has_production_context:
        substance_bonus += 25

    return min(100.0, base + substance_bonus)


def score_career_environment(candidate: dict) -> float:
    """Product ML > Product+Startup > Product+Consulting > Consulting Only."""
    jobs = candidate.get("career_history", [])
    if not jobs:
        return 20.0

    startup_sizes = {"1-10", "11-50", "51-200"}
    product_industries = {
        "software", "saas", "fintech", "edtech", "healthtech",
        "ecommerce", "marketplace", "ai", "transportation", "food delivery",
        "gaming", "media", "entertainment", "telecom"
    }
    consulting_companies = CONSULTING_FIRMS

    has_product = False
    has_startup = False
    is_consulting = True

    for job in jobs:
        industry = job.get("industry", "").lower()
        company = job.get("company", "").lower()
        size = job.get("company_size", "")

        if any(firm in company for firm in consulting_companies):
            continue
        else:
            is_consulting = False

        if any(ind in industry for ind in product_industries):
            has_product = True
        if size in startup_sizes:
            has_startup = True

    if is_consulting:
        return 20.0
    if has_product and has_startup:
        return 95.0
    if has_product:
        return 80.0
    return 60.0


def score_production_scale(candidate: dict) -> float:
    """TB-scale data, strict latency <100ms, high QPS."""
    career_text = extract_career_text(candidate)
    scale_terms = [
        "tb", "terabyte", "petabyte", "billion", "million users",
        "100ms", "<100", "p99", "latency", "qps", "queries per second",
        "high throughput", "large scale", "at scale"
    ]
    count = sum(1 for t in scale_terms if t in career_text)
    return min(100.0, count * 20.0)


def score_search_infra(candidate: dict) -> float:
    """Distinguishes basic VectorDB usage from advanced lifecycle (drift, indexing)."""
    career_text = extract_career_text(candidate)
    score = 0.0
    
    vector_db_terms = ["milvus", "pinecone", "qdrant", "weaviate", "faiss", "pgvector"]
    
    # Basic usage
    db_count = sum(1 for t in vector_db_terms if contains_exact_match(t, career_text))
    score += min(40, db_count * 15)
    
    # Advanced lifecycle/infrastructure ops (What Redrob actually wants)
    ops_count = sum(1 for op in EMBEDDING_OPS if contains_exact_match(op, career_text))
    score += min(60, ops_count * 30) # Double weight for operational realities
    
    return min(100.0, score)

def score_founding_team_fit(candidate: dict) -> float:
    """0-to-1 muscle: ambiguity, MVP, Seed/Series A/B experience."""
    career_text = extract_career_text(candidate)
    founding_terms = [
        "seed", "series a", "series b", "startup", "0 to 1", "zero to one",
        "founding", "ground up", "greenfield", "mvp", "prototype",
        "ambiguity", "fast-paced", "early stage", "from scratch"
    ]
    # Startup size signals
    startup_count = sum(
        1 for job in candidate.get("career_history", [])
        if job.get("company_size", "") in {"1-10", "11-50"}
    )
    term_count = sum(1 for t in founding_terms if t in career_text)
    score = min(60, term_count * 15) + min(40, startup_count * 20)
    return min(100.0, score)


def score_python_depth(candidate: dict) -> float:
    """Backend and system architecture design in Python."""
    full_text = extract_text(candidate)
    python_terms = [
        "python", "pyspark", "fastapi", "flask", "django",
        "asyncio", "multiprocessing", "pytest", "poetry", "pipenv"
    ]
    depth_terms = [
        "system design", "architecture", "microservice", "backend",
        "api design", "performance optimization", "profiling", "celery"
    ]
    py_count = sum(1 for t in python_terms if t in full_text)
    depth_count = sum(1 for t in depth_terms if t in full_text)
    return min(100.0, py_count * 15 + depth_count * 10)


def score_distributed_systems(candidate: dict) -> float:
    """Low-latency inference, concurrency."""
    full_text = extract_text(candidate)
    dist_terms = [
        "distributed", "kafka", "spark", "flink", "kubernetes", "k8s",
        "docker", "redis", "concurrent", "async", "parallelism",
        "load balancing", "sharding", "microservices", "grpc"
    ]
    count = sum(1 for t in dist_terms if t in full_text)
    return min(100.0, count * 12.0)


def score_external_validation(candidate: dict) -> float:
    """GitHub, conference talks, patents, papers."""
    signals = candidate.get("redrob_signals", {})
    gh_score = signals.get("github_activity_score", -1)
    full_text = extract_text(candidate)
    external_terms = ["paper", "publication", "patent", "conference", "talk",
                      "open source", "arxiv", "neurips", "icml", "acl", "emnlp"]

    score = 0.0
    if gh_score >= 50:
        score += 60
    elif gh_score >= 20:
        score += 30
    elif gh_score >= 0:
        score += 10

    ext_count = sum(1 for t in external_terms if t in full_text)
    score += min(40, ext_count * 15)
    return min(100.0, score)


def score_availability_timeline(candidate: dict) -> float:
    """Notice period tier × open_to_work_flag."""
    signals = candidate.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 90)
    open_to_work = signals.get("open_to_work_flag", False)

    if notice <= 30:
        tier_score = 100
    elif notice <= 60:
        tier_score = 60
    elif notice <= 90:
        tier_score = 30
    else:
        tier_score = 10

    availability = tier_score * (1 if open_to_work else 0)
    return float(availability)


def score_pipeline_friction(candidate: dict) -> float:
    """recruiter_response_rate × interview_completion_rate."""
    signals = candidate.get("redrob_signals", {})
    rr = signals.get("recruiter_response_rate", 0.0)
    icr = signals.get("interview_completion_rate", 0.0)
    return (rr * 50 + icr * 50)


def compute_raw_fit_score(candidate: dict) -> tuple:
    """Returns (raw_score, breakdown_dict)."""
    # Technical Fit (70%)
    tech_scores = {
        "relevance_ownership": score_relevance_ownership(candidate),
        "production_ml": score_production_ml(candidate),
        "evaluation_rigor": score_evaluation_rigor(candidate),
        "product_ownership": score_product_ownership(candidate),
        "retrieval_architecture": score_retrieval_architecture(candidate),
        "career_environment": score_career_environment(candidate),
        "production_scale": score_production_scale(candidate),
        "search_infra": score_search_infra(candidate),
    }
    tech_weighted = sum(
        tech_scores[k] * TECH_WEIGHTS[k] / 100
        for k in TECH_WEIGHTS
    )
    tech_total_weight = sum(TECH_WEIGHTS.values())  # 70
    technical_fit = tech_weighted / tech_total_weight * 70  # scale to 70 points

    # Culture (20%)
    culture_scores = {
        "founding_team_fit": score_founding_team_fit(candidate),
        "python_depth": score_python_depth(candidate),
        "distributed_systems": score_distributed_systems(candidate),
        "external_validation": score_external_validation(candidate),
    }
    culture_weighted = sum(
        culture_scores[k] * CULTURE_WEIGHTS[k] / 100
        for k in CULTURE_WEIGHTS
    )
    culture_total_weight = sum(CULTURE_WEIGHTS.values())  # 20
    culture_fit = culture_weighted / culture_total_weight * 20

    # Hireability (10%)
    hire_scores = {
        "availability_timeline": score_availability_timeline(candidate),
        "pipeline_friction": score_pipeline_friction(candidate),
    }
    hire_weighted = sum(
        hire_scores[k] * HIRE_WEIGHTS[k] / 100
        for k in HIRE_WEIGHTS
    )
    hire_total_weight = sum(HIRE_WEIGHTS.values())  # 10
    hire_fit = hire_weighted / hire_total_weight * 10

    raw_score = technical_fit + culture_fit + hire_fit

    breakdown = {
        "technical_fit": round(technical_fit, 1),
        "engineering_culture": round(culture_fit, 1),
        "hireability": round(hire_fit, 1),
        "tech_sub_scores": {k: round(v, 1) for k, v in tech_scores.items()},
    }
    return round(raw_score, 2), breakdown


# ── Stage 3: Core Relevance Index & Ceiling ───────────────────────────────────

def score_recommendation_ownership(candidate: dict) -> float:
    """
    Dedicated recommendation-systems scoring component for CRI. The JD
    explicitly lists "ranking, retrieval, and matching systems" and
    "Recommendation Systems" experience as a core requirement, but the
    original CRI formula only had Relevance/Eval-Rigor/Retrieval-
    Architecture, none of which a pure recommender engineer (no BM25,
    no NDCG, no ANN/HNSW language) has much reason to use. That left a
    genuine recommender engineer -- with real production deployment and
    a quantified outcome -- scoring as low as CRI~25, well below the
    Weak Signal threshold, despite being exactly the JD's stated profile.
    This component gives recommendation-specific evidence its own,
    title-consistency-respecting scoring path within CRI.
    """
    ir_job, is_consistent = find_ir_evidence_job(candidate)
    if ir_job is None or not is_consistent:
        return 0.0

    desc = aggregate_trustworthy_career_text(candidate)
    recommender_matched = {t for t in RECOMMENDER_TERMS if t in desc}
    if not recommender_matched:
        return 0.0

    score = min(60, len(recommender_matched) * 20)

    has_deploy = any(t in desc for t in ["production", "deployed", "shipped", "inference"])
    relevant_metric_names = (
        r'ctr|click-through|ndcg|mrr|map|relevance|engagement|revenue|'
        r'retention|conversion|latency|recall|precision|search|ranking|'
        r'retrieval|recommendation'
    )
    has_outcome = bool(re.search(
        r'\d+\s*%|\d+x\s|[+-]?\d+(\.\d+)?\s*pp\b|'
        rf'(improved|reduced|increased|decreased|boosted|grew|lifted|cut)\s+'
        rf'[\w\-\s]{{0,30}}?({relevant_metric_names})|'
        rf'({relevant_metric_names})[\w\-\s]{{0,30}}?'
        rf'(improved|reduced|increased|decreased|boosted|grew|lifted|cut|'
        rf'went from|from \d)',
        desc
    ))
    if has_deploy:
        score += 20
    if has_outcome:
        score += 20

    return min(100.0, score)


def compute_core_relevance_index(candidate: dict) -> float:
    """
    CRI takes the MAX of two valid evidence paths, rather than always
    linearly blending all four components:

      Path A (ranking/retrieval-centric):
        0.50(Relevance) + 0.30(Evaluation Rigor) + 0.20(Retrieval Arch)
      Path B (recommendation-centric):
        0.55(Relevance) + 0.45(Recommendation Ownership)

    Why: a pure recommender engineer (real production deployment,
    genuine recommendation/feed-ranking language, no BM25/NDCG/ANN
    vocabulary -- which they have no reason to use) was structurally
    capped near CRI~25 under a single linear blend, because Evaluation
    Rigor and Retrieval Architecture together carried 65% of the weight
    and are largely unreachable for that archetype even with strong,
    genuine recommendation evidence. The JD explicitly lists
    "Recommendation Systems" as a core requirement alongside ranking/
    retrieval, so a candidate shouldn't need evidence in ALL FOUR areas
    to score well -- excelling via either the ranking/retrieval path OR
    the recommendation path should each be sufficient on their own.
    Taking the max (rather than, say, summing or further reweighting)
    avoids double-counting a candidate who happens to have evidence on
    both paths, while still letting either path alone carry them to a
    fair score.
    """
    relevance = score_relevance_ownership(candidate)
    eval_rigor = score_evaluation_rigor(candidate)
    retrieval_arch = score_retrieval_architecture(candidate)
    recommender = score_recommendation_ownership(candidate)

    path_a = 0.50 * relevance + 0.30 * eval_rigor + 0.20 * retrieval_arch
    path_b = 0.55 * relevance + 0.45 * recommender

    cri = max(path_a, path_b)
    return round(cri, 1)


def apply_cri_ceiling(raw_score: float, cri: float) -> float:
    """Apply non-linear ceiling based on CRI."""
    if cri < 40:
        ceiling = 50
    elif cri < 60:
        ceiling = 70
    elif cri < 80:
        ceiling = 90
    else:
        ceiling = 100
    return min(raw_score, ceiling)


# ── Stage 4: Calibration Layer ────────────────────────────────────────────────

def apply_calibration(candidate: dict, raw_score: float) -> tuple:
    """Apply all penalties/bonuses. Returns (adjusted_score, penalty_flags)."""
    score = raw_score
    penalty_flags = []

    # Title-Chaser Penalty
    if title_chaser_check(candidate):
        score *= 0.90
        penalty_flags.append("Culture Risk: Title-Chaser")

    # Consulting Tax
    if is_consulting_only(candidate) and not has_product_ownership_verbs(candidate):
        score *= 0.85
        penalty_flags.append("Conditional Consulting Tax Applied")

    # LLM Wrapper Penalty
    if llm_wrapper_check(candidate):
        score *= 0.90
        penalty_flags.append("LLM Wrapper Penalty")

    # Skill Inflation Penalty
    inflation_penalty = compute_skill_inflation_penalty(candidate)
    if inflation_penalty > 0:
        score *= (1 - inflation_penalty)
        if inflation_penalty > 0.05:
            penalty_flags.append(f"Skill Inflation Penalty: -{inflation_penalty:.1%}")

    # Research Only Penalty
    if research_only_check(candidate):
        score *= 0.80
        penalty_flags.append("Research-Only Penalty")

    # Tenure Stability Penalty (only if title chaser not already applied)
    avg_tenure, is_unstable = compute_tenure_stability(candidate)
    if is_unstable and "Culture Risk: Title-Chaser" not in penalty_flags:
        score *= 0.95
        penalty_flags.append("Tenure Stability Penalty")

    # Product-Company Trajectory Bonus, capped at +5% maximum. A 10%
    # multiplicative bonus on this signal could outrank a candidate with
    # a meaningfully stronger CRI/technical-fit score, even though
    # career_environment itself is one of the LOWEST-weighted technical
    # sub-components (5/70 in TECH_WEIGHTS) -- the bonus shouldn't be
    # able to swing the final ranking more than the underlying signal's
    # intended weight justifies. Penalties (Title-Chaser, Consulting Tax,
    # LLM Wrapper, Research-Only, etc.) are left uncapped since
    # over-penalizing a genuinely weak signal is a safer failure mode
    # than over-rewarding a generic one.
    env_score = score_career_environment(candidate)
    if env_score >= 95:
        score *= 1.05
    elif env_score >= 80:
        score *= 1.03

    return round(score, 2), penalty_flags


# ── Stage 5: Confidence Engine ────────────────────────────────────────────────

def compute_confidence(candidate: dict) -> float:
    """Compute confidence score (0–100) based on Coverage, Reliability, Consistency."""
    signals = candidate.get("redrob_signals", {})
    career_text = extract_career_text(candidate)
    full_text = extract_text(candidate)

    # Safely compute recent activity. The previous inline expression
    # called date.fromisoformat() with no guard against malformed or
    # blank date strings (e.g. "15-06-2026" instead of YYYY-MM-DD, or a
    # whitespace-only string) -- such input would raise ValueError and
    # crash the entire pipeline mid-run, the same class of risk
    # apply_hard_gates was already protected against but this function
    # was not.
    try:
        last_active_str = signals.get("last_active_date", "")
        if not last_active_str.strip():
            raise ValueError("empty last_active_date")
        last_active = date.fromisoformat(last_active_str[:10])
        recent_active = (REFERENCE_DATE - last_active).days <= 90
    except ValueError:
        recent_active = False

    # Coverage: 7 hardcoded signals
    coverage_checks = [
        has_python(candidate),                                          # 1. Python
        any(t in career_text for t in ["production", "deployed",       # 2. Production scale
                                        "users", "scale", "million"]),
        ir_evidence_in_career(candidate),                               # 3. IR architecture
        any(contains_exact_match(t, full_text) for t in ["ndcg", "mrr", "a/b test",  # 4. Eval frameworks
                                      "map", "precision", "recall"]),
        recent_active,                                                  # 5. Recent activity
        signals.get("verified_email", False) or                         # 6. Verified contact
        signals.get("verified_phone", False),
        bool(re.search(r'\d+\s*%|\d+x\s|improved|reduced|increased',  # 7. Quantified outcome
                       career_text))
    ]
    coverage = sum(1 for c in coverage_checks if c) / 7

    # Reliability: weighted by source
    # For each of the 7 signals, determine the best source found
    source_weights = {
        "career": 1.0, "external": 0.9, "skills": 0.5, "endorsements": 0.2
    }
    signal_reliabilities = []
    skills_text = extract_skills_text(candidate)

    # Python: career=1.0 or skills=0.5
    if "python" in career_text:
        signal_reliabilities.append(1.0)
    elif "python" in skills_text:
        signal_reliabilities.append(0.5)

    # Production scale in career
    if any(t in career_text for t in ["production", "deployed", "scale"]):
        signal_reliabilities.append(1.0)

    # IR in career vs skills only
    if ir_evidence_in_career(candidate):
        signal_reliabilities.append(1.0)
    elif has_ir_ml_evidence(candidate):
        signal_reliabilities.append(0.5)

    # Eval in career
    eval_in_career = any(t in career_text for t in ["ndcg", "mrr", "a/b", "evaluation"])
    if eval_in_career:
        signal_reliabilities.append(1.0)
    elif any(t in skills_text for t in ["ndcg", "mrr"]):
        signal_reliabilities.append(0.5)

    # GitHub for external validation
    gh = signals.get("github_activity_score", -1)
    if gh > 20:
        signal_reliabilities.append(0.9)

    # Verified contact
    if signals.get("verified_email") or signals.get("verified_phone"):
        signal_reliabilities.append(1.0)

    # Quantified outcomes in career
    if re.search(r'\d+\s*%|\d+x\s', career_text):
        signal_reliabilities.append(1.0)

    reliability = (sum(signal_reliabilities) / len(signal_reliabilities)
                   if signal_reliabilities else 0.3)

    # Consistency: penalize for contradictory signals
    # e.g. claims IR skills but only ETL in career history
    ir_in_skills = has_ir_ml_evidence(candidate) and not ir_evidence_in_career(candidate)
    consistency = 1.0
    if ir_in_skills:
        consistency -= 0.3
    # Skill inflation
    inflation = compute_skill_inflation_penalty(candidate)
    if inflation > 0.05:
        consistency -= 0.2

    consistency = max(0.0, consistency)

    # Sparse profile protection
    _, _, _, total_verbs = extract_verbs(candidate)

    confidence = 0.4 * coverage + 0.4 * reliability + 0.2 * consistency
    confidence_score = round(confidence * 100, 1)

    # Sparse profile floor
    if total_verbs < 10:
        confidence_score = max(0, confidence_score - 10)

    return confidence_score


# ── Stage 6: Alternate Pipeline Routing ──────────────────────────────────────

def determine_alternate_pipeline(candidate: dict) -> str:
    """Deterministic routing based on signal pattern."""
    full_text = extract_text(candidate)
    career_text = extract_career_text(candidate)

    data_infra_terms = ["spark", "kafka", "airflow", "dbt", "snowflake",
                        "hadoop", "pipeline", "etl", "data warehouse", "flink"]
    cv_audio_terms = ["computer vision", "image classification", "object detection",
                      "cnn", "yolo", "tts", "speech", "audio", "gans", "resnet"]

    data_score = sum(1 for t in data_infra_terms if t in full_text)
    cv_score = sum(1 for t in cv_audio_terms if t in full_text)

    if llm_wrapper_check(candidate):
        return "Junior AI / Prompt Engineering"
    if research_only_check(candidate):
        return "Research Collaborator / Advisor Track"
    if is_consulting_only(candidate) and not has_product_ownership_verbs(candidate):
        return "Solutions Architecture / Implementation"
    if cv_score > data_score and cv_score >= 2:
        return "Computer Vision / Multimodal ML"
    if data_score >= 2:
        return "Data Engineering / MLOps"
    return "General Engineering Pipeline"


# ── Honeypot Detection ────────────────────────────────────────────────────────

def honeypot_score(candidate: dict) -> float:
    """
    Returns a penalty multiplier (0.0–1.0).
    0.0 = very likely honeypot. 1.0 = clean.
    """
    penalties = 0.0
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    # Check 1: Experience vs company age (impossible tenure)
    yoe = profile.get("years_of_experience", 0)
    for job in candidate.get("career_history", []):
        start = job.get("start_date", "")
        if start:
            try:
                start_yr = int(start[:4])
                company_implied_age = REFERENCE_DATE.year - start_yr
                if job.get("duration_months", 0) > (company_implied_age * 12 + 6):
                    penalties += 0.3
            except Exception:
                pass

    # Check 2: Expert proficiency in many skills with 0 months used
    expert_zero_dur = sum(
        1 for sk in candidate.get("skills", [])
        if sk.get("proficiency") in ("advanced", "expert")
        and sk.get("duration_months", 0) == 0
        and sk.get("endorsements", 0) == 0
    )
    if expert_zero_dur >= 5:
        penalties += 0.4

    # Check 3: salary_range min > max (inverted)
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal.get("min", 0) > sal.get("max", 0) and sal.get("max", 0) > 0:
        penalties += 0.15

    # Check 4: Signup date after last_active_date (impossible)
    signup = signals.get("signup_date", "")
    last_active = signals.get("last_active_date", "")
    if signup and last_active and signup > last_active:
        penalties += 0.25

    # Check 5: Title/description mismatch (title says one thing, desc says another)
    title = profile.get("current_title", "").lower()
    summary = profile.get("summary", "").lower()
    # If title is "Marketing Manager" but has heavy AI claims
    non_tech_titles = {"marketing manager", "accountant", "civil engineer",
                       "mechanical engineer", "hr manager", "operations manager",
                       "customer support", "project manager", "content writer",
                       "graphic designer", "sales executive", "business analyst"}
    if any(nt in title for nt in non_tech_titles):
        # Check if skills section is loaded with AI keywords despite non-tech title
        skills_text = extract_skills_text(candidate)
        ai_skill_count = sum(1 for kw in IR_ML_KEYWORDS
                              if (kw in skills_text if (" " in kw or "-" in kw)
                                  else re.search(r'\b' + re.escape(kw) + r'\b', skills_text)))
        if ai_skill_count >= 4:
            penalties += 0.2

    return max(0.0, 1.0 - penalties)


# ── Main Scoring Pipeline ─────────────────────────────────────────────────────

def count_ai_core_skills(candidate: dict) -> int:
    """Count how many of the candidate's listed skills are IR/ML/AI-core terms."""
    count = 0
    for sk in candidate.get("skills", []):
        name = sk.get("name", "").lower()
        if text_contains_ir_keyword(name):
            count += 1
        elif any(term in name for term in [
            "nlp", "llm", "fine-tun", "gan", "cnn", "transformer",
            "deep learning", "neural", "computer vision", "speech",
            "prompt engineering", "tts", "mlops", "pytorch", "tensorflow"
        ]):
            count += 1
    return count


def title_matches_ir_role(title: str) -> bool:
    """Does this job title itself suggest a ranking/search/retrieval-relevant role?"""
    t = title.lower()
    ir_relevant_titles = [
        "search", "ranking", "retrieval", "recommendation", "relevance",
        "ml engineer", "machine learning engineer", "applied scientist",
        "nlp engineer", "ai engineer", "data scientist"
    ]
    return any(kw in t for kw in ir_relevant_titles)


def is_cv_only_title(title: str) -> bool:
    """Is this job title CV/speech/robotics-specific (not IR/NLP)?"""
    t = title.lower()
    cv_terms = ["computer vision", "cv engineer", "vision engineer",
                "speech engineer", "robotics engineer", "image processing"]
    return any(kw in t for kw in cv_terms)


def title_contradicts_ir_evidence(title: str) -> bool:
    """
    Does this job title actively SUGGEST something inconsistent with IR/
    ranking work -- as opposed to merely being a generic engineering title
    that says nothing either way? Only the former should discount evidence;
    a title like 'Senior Software Engineer' is uninformative, not
    contradictory, and should not zero out genuinely strong description
    evidence just because it lacks a domain-specific keyword.
    """
    t = title.lower()
    if is_cv_only_title(t):
        return True
    # Non-technical / clearly unrelated-domain titles actively contradict
    # IR evidence appearing under them (this is the dataset's scrambling
    # trap: e.g. 'HR Manager' title with a ranking-system description).
    non_technical_titles = [
        "hr manager", "hr specialist", "accountant", "graphic designer",
        "content writer", "sales executive", "business analyst",
        "project manager", "operations manager", "marketing manager",
        "civil engineer", "mechanical engineer", "customer support",
        "customer success", "recruiter", "talent acquisition",
        "office manager", "executive assistant"
    ]
    return any(kw in t for kw in non_technical_titles)


def find_ir_evidence_job(candidate: dict):
    """
    Find the specific career_history entry that contains IR/ranking
    evidence, distinguishing three cases:
      1. Title actively supports it (ML Engineer, Search Engineer, etc.)
      2. Title is generic/uninformative (Senior Software Engineer,
         Engineering Lead) -- doesn't contradict, so evidence still counts
      3. Title actively contradicts it (CV-only title, or a clearly
         non-technical title like 'HR Manager') -- the dataset's
         scrambled title/description trap; evidence is discounted
    Returns (job_dict, is_evidence_trustworthy) or (None, False).
    """
    jobs = candidate.get("career_history", [])
    best_trustworthy = None
    best_contradicted = None

    for job in jobs:
        desc = job.get("description", "").lower()
        title = job.get("title", "")
        has_ir = text_contains_ir_keyword(desc)
        if not has_ir:
            continue

        if title_contradicts_ir_evidence(title):
            # Title actively contradicts the description -- scrambled pair
            if best_contradicted is None:
                best_contradicted = job
        else:
            # Title either supports it directly, or is merely generic --
            # in both cases the description evidence is trustworthy
            if best_trustworthy is None or job.get("is_current"):
                best_trustworthy = job

    if best_trustworthy is not None:
        return best_trustworthy, True
    if best_contradicted is not None:
        return best_contradicted, False
    return None, False


def aggregate_trustworthy_career_text(candidate: dict) -> str:
    """
    Concatenate description text from EVERY career_history entry whose own
    title does not contradict its content (per title_contradicts_ir_evidence),
    rather than restricting to a single best-matching job.

    score_relevance_ownership previously scanned only the single job
    returned by find_ir_evidence_job, which silently discarded legitimate
    evidence sitting in a candidate's OTHER title-consistent roles --
    e.g. a candidate whose current role has strong evaluation-rigor
    language ("offline-online correlation... predicted A/B test outcomes")
    and whose PAST role has strong ranking-architecture language
    ("learning-to-rank model... improved revenue-per-search by 12%") would
    only get credit for whichever single job was picked, even though both
    are genuine, title-consistent evidence. This mirrors the breadth that
    score_evaluation_rigor and score_retrieval_architecture already use
    (they scan extract_text() across all jobs), so relevance_ownership
    isn't artificially narrower than the other two CRI components.

    Jobs whose title actively contradicts their description (the dataset's
    scrambled title/description trap -- e.g. 'HR Manager' title sitting on
    a ranking-system description, or a CV-only title sitting on
    recommendation-system text) are still excluded, exactly as
    find_ir_evidence_job excludes them from "best_trustworthy".
    """
    jobs = candidate.get("career_history", [])
    trustworthy_descs = []
    for job in jobs:
        title = job.get("title", "")
        desc = job.get("description", "")
        if not title_contradicts_ir_evidence(title):
            trustworthy_descs.append(desc)
    return " ".join(trustworthy_descs).lower()


def extract_best_evidence(candidate: dict) -> str:
    """
    Find the single strongest piece of concrete evidence in the candidate's
    career history and describe it in plain language (paraphrased, not
    quoted verbatim from the profile). Title-aware: does not cite IR
    evidence sitting under a job whose own title contradicts it (the
    dataset's scrambled title/description trap), and flags CV-only
    candidates per the JD's explicit warning about CV/speech/robotics
    backgrounds without real NLP/IR exposure.
    """
    current_title = ""
    for job in candidate.get("career_history", []):
        if job.get("is_current"):
            current_title = job.get("title", "")
            break

    ir_job, is_consistent = find_ir_evidence_job(candidate)

    if ir_job is not None and is_consistent:
        # Use aggregated text across ALL title-consistent jobs, not just
        # the single job find_ir_evidence_job returns -- otherwise a
        # candidate whose strongest, most specific evidence (e.g. "BM25 +
        # dense retrieval... NDCG, MRR, recall@K... learning-to-rank") sits
        # in a DIFFERENT job than the one picked (e.g. their current role,
        # which might be about fine-tuning/inference-cost work instead)
        # would get the generic fallback message instead of citing their
        # actual strongest evidence. This mirrors the same fix already
        # applied to score_relevance_ownership.
        desc = aggregate_trustworthy_career_text(candidate)

        # Cite the SPECIFIC matched phrase, not a generic bucket label --
        # e.g. "evolving a hand-tuned scorer into a learning-to-rank
        # model" instead of always saying "owned a ranking/search-
        # relevance system in production". Checking phrases in priority
        # order (most specific/rare first) and returning the literal
        # phrase context avoids dozens of distinct candidates collapsing
        # into one identical templated sentence, which is exactly what
        # the submission spec's Stage 4 review flags as a red flag
        # ("All-identical reasoning strings", "lack of variation").
        ranking_phrase_map = [
            ("learning-to-rank", "evolved scoring into a learning-to-rank model"),
            ("learning to rank", "evolved scoring into a learning-to-rank model"),
            ("ltr model", "built and shipped an LTR model"),
            ("ranking layer", "owned the ranking layer end-to-end"),
            ("relevance labeling", "designed the relevance-labeling pipeline feeding the ranker"),
            ("relevance layer", "owned the relevance layer in production"),
            ("ranking system", "built a ranking system in production"),
            ("ranking model", "shipped a ranking model in production"),
            ("search relevance", "owned search relevance for the product"),
            ("search quality", "owned search quality for the product"),
            ("ranker", "shipped a ranker in production"),
        ]
        # Some keywords appear inside COMPARATIVE or DOWNPLAYING language
        # rather than a genuine first-person claim -- e.g. "lighter weight
        # than ranking systems at FAANG" contains the literal substring
        # "ranking system" but is explicitly saying the candidate's work
        # is NOT a full ranking system, by way of unfavorable comparison.
        # A bare substring match would fabricate a positive citation
        # ("built a ranking system in production") that directly
        # contradicts what the candidate's own words say. This is a
        # hallucination risk, not just a differentiation gap -- it must
        # be excluded, not just deprioritized.
        comparative_disclaimer_window = 25  # chars of context to inspect before the match
        comparative_markers = [
            "lighter weight than", "less sophisticated than", "simpler than",
            "not as", "nowhere near", "far from a full", "a simplified version of",
            "more basic than", "smaller scale than",
        ]

        def is_comparative_mention(matched_keyword: str, text: str) -> bool:
            idx = text.find(matched_keyword)
            if idx == -1:
                return False
            window_start = max(0, idx - 60)
            preceding_text = text[window_start:idx]
            return any(marker in preceding_text for marker in comparative_markers)

        matched_phrase_text = None
        for keyword, phrase_description in ranking_phrase_map:
            if keyword in desc and not is_comparative_mention(keyword, desc):
                matched_phrase_text = phrase_description
                break

        if matched_phrase_text:
            # Append a distinguishing secondary detail when present, so
            # two candidates who both "built a ranking system" still read
            # differently if one has scale/eval evidence and the other
            # doesn't. Critically: find ALL quantified outcomes across the
            # candidate's full aggregated text (not just the first regex
            # hit) and surface the LARGEST one.
            #
            # This dataset reuses near-identical templated career-history
            # sentences across many candidates (the same "evolving it from
            # a hand-tuned scoring function to a learning-to-rank model...
            # improved revenue-per-search by 12%" sentence appears
            # verbatim for several different candidates). Taking only the
            # first match in the aggregated text means whichever job
            # happens to come first wins, even when a candidate has a
            # genuinely more impressive, non-templated outcome in a
            # DIFFERENT job (e.g. a 35% search-relevance improvement, or
            # a recommendation system serving 10M+ users) that gets
            # silently ignored. Scanning for the maximum percentage across
            # all jobs surfaces the candidate's actual best evidence and
            # naturally differentiates candidates who share one templated
            # sentence but differ everywhere else in their history.
            extra = []
            # Two grammatical patterns appear in this dataset's
            # descriptions: verb-form ("improved revenue-per-search by
            # 12%") and noun-form ("reported search-relevance improvement
            # of 35%", "saw a 6% lift in retention"). Catching only the
            # verb-form misses genuinely larger, more impressive outcomes
            # phrased as nouns -- which is exactly what was happening
            # here: a candidate's best evidence (a 35% search-relevance
            # improvement) was being silently passed over in favor of a
            # smaller, verb-phrased 12% figure from a different, more
            # templated job entry.
            verb_form = re.findall(
                r'(improved|reduced|increased|decreased|boosted|grew|lifted|cut)\s+'
                r'([\w\-\s]{0,40}?)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
                desc
            )
            noun_form = re.findall(
                r'([\w\-\s]{0,40}?)\s*(?:improvement|increase|reduction|lift|gain|drop)\s+'
                r'(?:of|in)?\s*(\d+(?:\.\d+)?)\s*%',
                desc
            )
            # Normalize noun_form to the same (verb, metric, pct) shape.
            # Strip any leading reporting/filler verb the lookback window
            # may have captured (e.g. "reported search-relevance" should
            # yield metric="search-relevance", not duplicate "reported").
            leading_filler_re = re.compile(
                r'^(reported|saw|achieved|recorded|measured|observed|delivered|drove|the|a|an)\s+',
                re.IGNORECASE
            )
            def clean_metric_phrase(raw_phrase: str) -> str:
                cleaned = raw_phrase.strip()
                # Strip repeatedly in case of multiple filler words stacked
                for _ in range(3):
                    new_cleaned = leading_filler_re.sub('', cleaned).strip()
                    if new_cleaned == cleaned:
                        break
                    cleaned = new_cleaned
                return cleaned or "a key metric"

            normalized_outcomes = [(v, m, p) for v, m, p in verb_form]
            normalized_outcomes += [("reported", clean_metric_phrase(m), p)
                                     for m, p in noun_form]

            all_multipliers = re.findall(r'(\d+(?:\.\d+)?)\s*x\s', desc)

            if normalized_outcomes:
                # Pick the outcome with the largest percentage -- the
                # candidate's single most impressive quantified result,
                # regardless of which job entry or grammatical form it
                # came from.
                best_outcome = max(normalized_outcomes, key=lambda o: float(o[2]))
                verb, metric_phrase, pct = best_outcome
                metric_clean = metric_phrase.strip() or "a key metric"
                extra.append(f"{verb} {metric_clean} by {pct}%")
            elif all_multipliers:
                best_multiplier = max(all_multipliers, key=float)
                extra.append(f"a {best_multiplier}x improvement")
            elif any(t in desc for t in ["ndcg", "mrr", "a/b test", "ab test"]):
                extra.append("backed by formal eval metrics")
            elif any(t in desc for t in ["million", "billion", "qps", "queries per"]):
                extra.append("at meaningful production scale")
            suffix = f" ({extra[0]})" if extra else ""
            return matched_phrase_text + suffix

        has_disclaimer = any(p in desc for p in OWNERSHIP_DISCLAIMER_PATTERNS)

        if any(t in desc for t in ["hybrid retrieval", "dense retrieval",
                                    "bm25", "semantic search"]):
            if has_disclaimer:
                return "has hybrid/dense retrieval exposure, though deployment was owned by a different team"
            return "shipped hybrid/dense retrieval in production"
        if any(t in desc for t in ["recommendation system", "recommender",
                                    "collaborative filtering", "re-ranking",
                                    "matrix factorization", "discovery feed",
                                    "ranking models for"]):
            if has_disclaimer:
                return "built the modeling side of a recommendation/re-ranking system, though production deployment was owned by a different team"
            return "built a production recommendation/re-ranking system"
        if any(t in desc for t in ["ndcg", "mrr", "a/b test", "ab test",
                                    "offline-online correlation",
                                    "offline to online correlation",
                                    "click-through", "relevance judgments",
                                    "human judgments"]):
            return "set up offline/online ranking evaluation rigor (eval framework, A/B correlation)"
        if any(t in desc for t in ["embeddings", "vector search", "vector database"]):
            if has_disclaimer:
                return "has embeddings/vector-search modeling exposure, though production deployment was owned by a different team"
            return "has embeddings/vector-search production experience"
        if any(t in desc for t in ["matching system", "candidate matching",
                                    "job matching", "talent matching"]):
            return "built a matching system directly analogous to this role"
        return "has confirmed, title-consistent IR/ranking evidence in their current or past role"

    if ir_job is not None and not is_consistent:
        if is_cv_only_title(current_title):
            return ("primary background is CV/vision-specific; IR/ranking "
                    "language in profile doesn't match the role titles -- "
                    "treat with caution per JD's CV-background warning")
        return ("IR/ranking keywords appear in career history but under a "
                "job title that doesn't match -- evidence is inconsistent, "
                "verify directly with candidate")

    if is_cv_only_title(current_title):
        return "primary background is CV/vision-specific with no NLP/IR exposure -- per JD, likely needs to re-learn fundamentals"

    career_text = extract_career_text(candidate)
    full_text = extract_text(candidate)
    # Only lead with career-environment as positive evidence when there's
    # at least SOME ML/technical signal alongside it -- for a candidate
    # with zero IR evidence and a CRI floor score, citing "founding-team
    # fit" as the headline evidence is technically true but misleading,
    # since it presents a culture/logistics signal as if it were
    # relevance evidence for someone who has none.
    has_any_ml_signal_raw = (
        re.search(r'\bml\b', career_text) is not None
        or any(t in career_text for t in [
            "model", "machine learning", "neural", "data pipeline",
            "etl", "feature", "prediction"
        ])
    )
    # A bare ML-keyword match alone is too permissive: a candidate who
    # explicitly minimizes their own ML involvement ("I wouldn't call
    # myself an ML specialist", "70% on data infrastructure") still
    # satisfies the raw keyword check, but citing "founding-team fit"
    # alongside that ML mention misleadingly implies more relevant
    # technical depth than the candidate themselves claims to have.
    # Reuse the same disclaimer detection already proven elsewhere in
    # this file rather than inventing a second, shallower check.
    has_ml_disclaimer = any(p in career_text for p in OWNERSHIP_DISCLAIMER_PATTERNS) or any(
        p in career_text for p in [
            "wouldn't call myself", "would not call myself", "not an ml specialist",
            "not a specialist", "lightweight ml", "comfortable with the modeling work but"
        ]
    )
    has_any_ml_signal = has_any_ml_signal_raw and not has_ml_disclaimer
    if score_career_environment(candidate) >= 95 and has_any_ml_signal:
        return "comes from a product+startup background (founding-team fit)"
    has_generic_disclaimer = any(p in career_text for p in OWNERSHIP_DISCLAIMER_PATTERNS)
    if any(t in career_text for t in ["deployed", "production", "inference"]):
        if has_generic_disclaimer:
            return "has ML modeling exposure, though production ownership in their history was generally handled by other teams"
        return "has hands-on production ML deployment experience"
    if any(t in career_text for t in ["data pipeline", "etl", "spark", "airflow", "kafka"]):
        return "has strong data-infrastructure experience, but it's adjacent to IR/ranking, not core to it"
    return "has limited direct evidence of ranking/retrieval/search ownership in their career history"


def generate_reasoning(candidate: dict, gate_result: tuple, scores: dict) -> str:
    """
    Generate reasoning in the format:
    '{Title} with {X} yrs; {N} AI core skills; response rate {Y}. {Evidence clause}. {Concern clause if any}'
    For alternate_pipeline (below-cutoff) candidates, produces a clear filler
    explanation matching the spec's own sample format so Stage 4 reviewers
    understand exactly why this candidate appears despite not passing the gate.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    passed, route, flags, _ = gate_result

    title = profile.get("current_title", "Candidate")
    yoe = profile.get("years_of_experience", 0)
    ai_skill_count = count_ai_core_skills(candidate)
    response_rate = signals.get("recruiter_response_rate", 0.0)

    base = f"{title} with {yoe:.1f} yrs; {ai_skill_count} AI core skills; response rate {response_rate:.2f}."

    # Below-cutoff candidates get a distinct, honest filler explanation.
    # The spec's own sample uses: "Adjacent skills only — likely below cutoff
    # but included as final filler given experience and engagement signals."
    # We extend this with the specific reason they failed the gate, so a
    # Stage 4 reviewer sampling from the tail understands immediately.
    if route == "alternate_pipeline":
        cri = scores.get("cri", 0)
        if cri < 10:
            reason = "no direct IR/ranking/retrieval evidence found in career history"
        elif cri < 25:
            reason = f"CRI {cri:.0f}/100 - insufficient IR/ranking ownership for this role"
        else:
            reason = f"CRI {cri:.0f}/100 - adjacent experience without core ranking/retrieval ownership"
        filler_note = (
            f"Below cutoff - included as mandatory filler to reach 100 candidates "
            f"({reason}). Not a true match for this ranking/retrieval JD."
        )
        return f"{base} {filler_note}"

    evidence = extract_best_evidence(candidate)
    evidence_clause = f" Evidence: {evidence}."

    # Collect concerns in two tiers: technical-fit concerns (which bear
    # directly on whether this candidate can do the job) and logistics
    # concerns (which bear on whether they're practically hireable). When
    # multiple concerns exist, a technical-fit concern must surface first --
    # otherwise a borderline-CRI candidate whose relocation flag happens to
    # get appended first would have their thin IR depth silently hidden
    # behind a logistics note, which misleads a recruiter judging fit.
    technical_concerns = []
    logistics_concerns = []

    cri = scores.get("cri", 0)
    if 0 < cri < 60:
        technical_concerns.append(f"CRI only {cri:.0f}/100 -- thinner IR depth than ideal")
    inflation_flags = scores.get("penalty_flags", []) or []
    if any("Skill Inflation" in f for f in inflation_flags):
        technical_concerns.append("skill assessment scores below claimed proficiency")

    notice = signals.get("notice_period_days", 90)
    if notice > 60:
        logistics_concerns.append(f"{notice}-day notice period")
    if not signals.get("willing_to_relocate", True) and "pune" not in profile.get("location", "").lower() \
            and "noida" not in profile.get("location", "").lower():
        logistics_concerns.append("not willing to relocate")

    concerns = technical_concerns + logistics_concerns

    concern_clause = f" Concern: {concerns[0]}." if concerns else ""

    return base + evidence_clause + concern_clause


def score_candidate(candidate: dict) -> dict:
    """Full 6-stage pipeline for one candidate."""
    gate_result = apply_hard_gates(candidate)
    passed, route, flags, continue_scoring = gate_result

    candidate_id = candidate.get("candidate_id", "")
    profile = candidate.get("profile", {})

    cri = compute_core_relevance_index(candidate)

    hp_multiplier = honeypot_score(candidate)

    if not continue_scoring:
        return {
            "candidate_id": candidate_id,
            "passed_gates": passed,
            "route": route,
            "flags": flags,
            "final_score": 0.0,
            "raw_fit_score": 0.0,
            "cri": cri,
            "shipping_velocity": 0.0,
            "confidence": 0.0,
            "alternate_pipeline": determine_alternate_pipeline(candidate),
            "breakdown": {},
            "penalty_flags": [],
            "honeypot_multiplier": hp_multiplier,
        }

    raw_score, breakdown = compute_raw_fit_score(candidate)

    cri_capped = apply_cri_ceiling(raw_score, cri)

    calibrated, penalty_flags = apply_calibration(candidate, cri_capped)

    final_score = calibrated * hp_multiplier

    if route == "alternate_pipeline":
        final_score = min(final_score, 30.0)

    confidence = compute_confidence(candidate)

    # Smooth confidence multiplier instead of a step function. The
    # previous version created discontinuities at confidence=25 and
    # confidence=40 (e.g. confidence=39 -> 0.95x but confidence=40 ->
    # 1.00x, an arbitrary one-point cliff with no principled basis).
    # confidence_factor = 0.80 + 0.20*(confidence/100) ranges smoothly
    # from 0.80 (confidence=0) to 1.00 (confidence=100), so a one-point
    # change in confidence now always produces a proportionally small,
    # continuous change in final_score rather than a sudden jump.
    confidence_factor = 0.80 + 0.20 * (confidence / 100.0)
    final_score *= confidence_factor

    sv = shipping_velocity(candidate)

    return {
        "candidate_id": candidate_id,
        "passed_gates": passed,
        "route": route,
        "flags": flags,
        "final_score": round(final_score, 2),
        "raw_fit_score": round(raw_score, 2),
        "cri": cri,
        "shipping_velocity": round(sv, 3),
        "confidence": confidence,
        "alternate_pipeline": determine_alternate_pipeline(candidate) if route != "pass" else "None",
        "breakdown": breakdown,
        "penalty_flags": penalty_flags,
        "honeypot_multiplier": round(hp_multiplier, 3),
    }


def load_candidates(path: str) -> list:
    """Load candidates from .jsonl or .jsonl.gz."""
    candidates = []
    if path.endswith(".gz"):
        opener = gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = open(path, "r", encoding="utf-8")

    with opener as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output")
    parser.add_argument("--debug", action="store_true", help="Print top-20 scores to stderr")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.", file=sys.stderr)

    print("Scoring candidates...", file=sys.stderr)
    scored = []
    for i, c in enumerate(candidates):
        result = score_candidate(c)
        scored.append(result)
        if (i + 1) % 10000 == 0:
            print(f"  Scored {i+1}/{len(candidates)}...", file=sys.stderr)

    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    top_n = scored[:args.top_n]

    for i in range(len(top_n) - 1):
        if top_n[i]["final_score"] < top_n[i + 1]["final_score"]:
            raise AssertionError(
                f"Non-monotonic score at rank {i+1} ({top_n[i]['final_score']}) "
                f"< rank {i+2} ({top_n[i+1]['final_score']}). "
                f"This should be impossible after the global sort -- investigate."
            )

    if args.debug:
        print("\nTop 20 candidates:", file=sys.stderr)
        for i, s in enumerate(top_n[:20]):
            print(f"  {i+1}. {s['candidate_id']} | score={s['final_score']:.1f} | "
                  f"raw={s['raw_fit_score']:.1f} | CRI={s['cri']:.1f} | "
                  f"route={s['route']} | flags={s['flags']}", file=sys.stderr)

    cand_by_id = {c["candidate_id"]: c for c in candidates}

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, result in enumerate(top_n, 1):
            cid = result["candidate_id"]
            cand = cand_by_id.get(cid, {})

            norm_score = result["final_score"] / 100.0
            formatted_score = f"{norm_score:.4f}"

            scores_for_reasoning = {
                "final_score": result["final_score"],
                "raw_fit": result["raw_fit_score"],
                "cri": result["cri"],
                "alternate_pipeline": result["alternate_pipeline"],
                "penalty_flags": result.get("penalty_flags", []),
            }
            gate_result = (
                result["passed_gates"],
                result["route"],
                result["flags"],
                True
            )
            reasoning = generate_reasoning(cand, gate_result, scores_for_reasoning)

            writer.writerow([cid, rank, formatted_score, reasoning])

    print(f"\nSubmission written to {args.out}", file=sys.stderr)
    print(f"Total candidates scored: {len(scored)}", file=sys.stderr)
    print(f"Candidates passing gates: {sum(1 for s in scored if s['route'] == 'pass')}", file=sys.stderr)
    print(f"Top-{args.top_n} output rows: {len(top_n)}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
Entity Extractor: NLP Pipeline for Market Question Analysis

This module extracts structured information from market question text:
1. Named entities (people, teams, locations, organizations)
2. Numerical thresholds and ranges (margins, scores, seat counts)
3. Logical operators (win, lose, by at least, more than)
4. Temporal references (dates, "through 2024", "by end of")

Why this matters for dependency detection:
- Two markets sharing the same entities + same event date are candidate dependents
- Threshold containment (e.g., "wins" contains "wins by 2+") signals dependency
- Opposite framings of the same event (D wins vs R wins) are complementary

The paper's LLM approach failed on entity disambiguation (popular vote vs electoral
college, Senate vs House). Structured extraction catches these distinctions explicitly.

Example:
    Input:  "Will the Democratic candidate win Georgia by 2.0%-3.0%?"
    Output: {
        "entities": {"parties": ["Democratic"], "locations": ["Georgia"]},
        "event_type": "election_margin",
        "thresholds": {"min": 2.0, "max": 3.0, "unit": "percent"},
        "direction": "win",
        "subject": "Democratic candidate",
        "object": "Georgia",
    }
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# EXTRACTED STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class MarketEntities:
    """Structured extraction from a market question."""
    # Named entities
    persons: list = field(default_factory=list)      # ["Donald Trump", "Kamala Harris"]
    locations: list = field(default_factory=list)     # ["Georgia", "New York"]
    organizations: list = field(default_factory=list) # ["Republican", "Democratic"]
    teams: list = field(default_factory=list)         # ["Boston Celtics", "Lakers"]

    # Event classification
    event_type: str = ""          # "election_winner", "election_margin", "sports_winner",
                                  # "sports_score", "price_target", "yes_no_binary"
    event_subtype: str = ""       # "presidential", "senate", "house", "governor", etc.

    # Logical structure
    direction: str = ""           # "win", "lose", "exceed", "remain", "drop"
    subject: str = ""             # Who/what the question is about
    object: str = ""              # What they're winning/losing (state, game, etc.)

    # Numerical thresholds
    threshold_min: Optional[float] = None   # Lower bound (inclusive)
    threshold_max: Optional[float] = None   # Upper bound (inclusive)
    threshold_unit: str = ""      # "percent", "points", "goals", "seats", "dollars"
    threshold_exact: Optional[float] = None  # Exact value if specified

    # Temporal
    time_reference: str = ""      # "2024", "through 2024", "by November"

    # Raw tokens for fallback matching
    key_tokens: list = field(default_factory=list)


# ─────────────────────────────────────────────
# EXTRACTION PATTERNS
# ─────────────────────────────────────────────

# Political entities
POLITICAL_PARTIES = {
    "democrat": "Democratic", "democratic": "Democratic", "dem": "Democratic",
    "republican": "Republican", "gop": "Republican", "rep": "Republican",
    "third-party": "Third-Party", "third party": "Third-Party",
    "independent": "Independent", "libertarian": "Libertarian",
    "green": "Green Party",
}

# Known political figures (extend as needed — or use a proper NER model)
KNOWN_POLITICIANS = [
    "Donald Trump", "Kamala Harris", "Joe Biden", "JD Vance", "Tim Walz",
    "Ron DeSantis", "Nikki Haley", "Vivek Ramaswamy", "Robert Kennedy",
    "RFK Jr", "Gavin Newsom", "Mike Pence", "Marco Rubio",
    "Elizabeth Warren", "Bernie Sanders", "Pete Buttigieg",
]

# US States (for election markets)
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]

# Election subtypes
ELECTION_SUBTYPES = {
    "presidential": ["president", "presidential", "presidency", "white house", "electoral college"],
    "senate": ["senate", "senator"],
    "house": ["house", "representative", "congress"],
    "governor": ["governor", "gubernatorial"],
    "popular_vote": ["popular vote"],
    "electoral_college": ["electoral college", "electoral vote"],
}

# Threshold patterns
THRESHOLD_PATTERNS = [
    # "by 2.0%-3.0%" → range
    re.compile(r'by\s+(\d+\.?\d*)%?\s*[-–]\s*(\d+\.?\d*)%', re.I),
    # "by at least 2 goals" → min only
    re.compile(r'by\s+(?:at\s+least|more\s+than|over)\s+(\d+\.?\d*)\s*(\w+)?', re.I),
    # "by less than 2" → max only
    re.compile(r'by\s+(?:less\s+than|under|fewer\s+than)\s+(\d+\.?\d*)\s*(\w+)?', re.I),
    # "win by 2+" → min with plus
    re.compile(r'by\s+(\d+\.?\d*)\+', re.I),
    # "have 56 or more seats" → min threshold
    re.compile(r'(\d+)\s+or\s+more\s+(\w+)', re.I),
    # "have 51 seats" → exact
    re.compile(r'have\s+(\d+)\s+(\w+)', re.I),
    # "215+" → min with plus
    re.compile(r'(\d+)\+', re.I),
    # General range: "1-4" with context
    re.compile(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)', re.I),
]

# Direction keywords
WIN_WORDS = {"win", "wins", "won", "victory", "defeat", "beat", "beats"}
LOSE_WORDS = {"lose", "loses", "lost", "eliminated", "drops"}
EXCEED_WORDS = {"exceed", "exceeds", "surpass", "above", "over", "more than"}
REMAIN_WORDS = {"remain", "stay", "keep", "hold", "retain"}


# ─────────────────────────────────────────────
# EXTRACTOR
# ─────────────────────────────────────────────

class EntityExtractor:
    """
    Extracts structured information from market question text.

    This uses rule-based extraction optimized for Polymarket's question formats.
    For production, you'd layer spaCy NER on top for better generalization.

    Usage:
        extractor = EntityExtractor()
        entities = extractor.extract("Will the Democratic candidate win Georgia by 2.0%-3.0%?")
        print(entities.locations)       # ["Georgia"]
        print(entities.threshold_min)   # 2.0
        print(entities.threshold_max)   # 3.0
    """

    def __init__(self, use_spacy: bool = False):
        """
        Args:
            use_spacy: If True, use spaCy NER model for entity extraction.
                       Falls back to rule-based if spaCy is not available.
        """
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Using spaCy NER model")
            except (ImportError, OSError):
                logger.warning("spaCy not available, falling back to rule-based extraction")

    def extract(self, question: str, description: str = "") -> MarketEntities:
        """
        Extract structured entities from a market question.

        Args:
            question: The market question text
            description: Optional full description for additional context

        Returns:
            MarketEntities with all extracted fields populated
        """
        entities = MarketEntities()
        text = question.strip()
        text_lower = text.lower()
        full_text = f"{question} {description}".strip()

        # 1. Extract named entities
        entities.persons = self._extract_persons(full_text)
        entities.locations = self._extract_locations(text)
        entities.organizations = self._extract_organizations(text_lower)
        entities.teams = self._extract_teams(text)

        # 2. Classify event type
        entities.event_type = self._classify_event(text_lower)
        entities.event_subtype = self._classify_election_subtype(text_lower)

        # 3. Extract direction
        entities.direction = self._extract_direction(text_lower)

        # 4. Extract thresholds
        self._extract_thresholds(text, entities)

        # 5. Extract temporal references
        entities.time_reference = self._extract_time(text)

        # 6. Extract subject and object
        entities.subject = self._extract_subject(text, entities)
        entities.object = self._extract_object(text, entities)

        # 7. Key tokens for fallback matching
        entities.key_tokens = self._tokenize_key_words(text_lower)

        return entities

    def _extract_persons(self, text: str) -> list:
        """Extract person names from text."""
        found = []

        # Rule-based: check known politicians
        for name in KNOWN_POLITICIANS:
            if name.lower() in text.lower():
                found.append(name)

        # spaCy NER if available
        if self.nlp and not found:
            doc = self.nlp(text)
            found = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        return list(set(found))

    def _extract_locations(self, text: str) -> list:
        """Extract location names, especially US states."""
        found = []
        text_lower = text.lower()

        for state in US_STATES:
            if state.lower() in text_lower:
                found.append(state)

        # Also check for country names, cities etc. via spaCy
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC") and ent.text not in found:
                    found.append(ent.text)

        return list(set(found))

    def _extract_organizations(self, text_lower: str) -> list:
        """Extract political parties and organizations."""
        found = []
        for keyword, party in POLITICAL_PARTIES.items():
            if keyword in text_lower:
                found.append(party)
        return list(set(found))

    def _extract_teams(self, text: str) -> list:
        """
        Extract sports team names.
        In production, use a sports entity database. Here we use simple heuristics.
        """
        teams = []
        # Pattern: "Team A vs Team B" or "Team A defeats Team B"
        vs_match = re.search(r'(.+?)\s+(?:vs\.?|versus|v\.)\s+(.+?)(?:\?|$)', text, re.I)
        if vs_match:
            teams = [vs_match.group(1).strip(), vs_match.group(2).strip()]

        return teams

    def _classify_event(self, text_lower: str) -> str:
        """Classify the type of event the market is about."""
        if any(w in text_lower for w in ["election", "vote", "ballot", "nominee", "candidate"]):
            if any(w in text_lower for w in ["margin", "by ", "percentage", "points"]):
                return "election_margin"
            return "election_winner"
        if any(w in text_lower for w in ["game", "match", "championship", "tournament", "bowl"]):
            if any(w in text_lower for w in ["score", "points", "goals", "margin"]):
                return "sports_score"
            return "sports_winner"
        if any(w in text_lower for w in ["price", "bitcoin", "ethereum", "btc", "eth", "$"]):
            return "price_target"
        if any(w in text_lower for w in ["balance of power", "senate", "house"]):
            return "balance_of_power"
        return "yes_no_binary"

    def _classify_election_subtype(self, text_lower: str) -> str:
        """Distinguish presidential, senate, house, etc."""
        for subtype, keywords in ELECTION_SUBTYPES.items():
            if any(kw in text_lower for kw in keywords):
                return subtype
        return ""

    def _extract_direction(self, text_lower: str) -> str:
        """Extract the direction/outcome of the question."""
        words = set(text_lower.split())
        if words & WIN_WORDS:
            return "win"
        if words & LOSE_WORDS:
            return "lose"
        if words & EXCEED_WORDS:
            return "exceed"
        if words & REMAIN_WORDS:
            return "remain"
        return ""

    def _extract_thresholds(self, text: str, entities: MarketEntities):
        """Extract numerical thresholds and ranges."""
        for pattern in THRESHOLD_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1] and groups[1].replace(".", "").isdigit():
                    # Range: min-max
                    entities.threshold_min = float(groups[0])
                    entities.threshold_max = float(groups[1])
                elif "+" in match.group():
                    # Min only with +
                    entities.threshold_min = float(groups[0])
                elif "less" in text.lower() or "under" in text.lower():
                    entities.threshold_max = float(groups[0])
                elif "more" in text.lower() or "least" in text.lower() or "over" in text.lower():
                    entities.threshold_min = float(groups[0])
                else:
                    # Could be exact or range depending on context
                    try:
                        entities.threshold_exact = float(groups[0])
                    except (ValueError, TypeError):
                        pass

                # Extract unit
                if len(groups) > 1 and groups[-1]:
                    unit = groups[-1].lower()
                    if unit in ("percent", "%", "pct"):
                        entities.threshold_unit = "percent"
                    elif unit in ("goals", "points", "runs"):
                        entities.threshold_unit = unit
                    elif unit in ("seats",):
                        entities.threshold_unit = "seats"
                elif "%" in text:
                    entities.threshold_unit = "percent"

                break  # Use first matching pattern

    def _extract_time(self, text: str) -> str:
        """Extract temporal references."""
        # Year patterns
        year_match = re.search(r'20\d{2}(?:-\d{2})?', text)
        if year_match:
            return year_match.group()
        # "through YYYY", "by end of YYYY"
        through_match = re.search(r'(?:through|by\s+end\s+of|before)\s+(\d{4})', text, re.I)
        if through_match:
            return f"through {through_match.group(1)}"
        return ""

    def _extract_subject(self, text: str, entities: MarketEntities) -> str:
        """Extract the primary subject of the question."""
        if entities.persons:
            return entities.persons[0]
        if entities.organizations:
            return entities.organizations[0]
        if entities.teams:
            return entities.teams[0]
        return ""

    def _extract_object(self, text: str, entities: MarketEntities) -> str:
        """Extract the object (what is being won/lost/etc.)."""
        if entities.locations:
            return entities.locations[0]
        return ""

    def _tokenize_key_words(self, text_lower: str) -> list:
        """Extract key content words, removing stop words."""
        stop_words = {
            "will", "the", "a", "an", "in", "of", "for", "to", "by", "at",
            "is", "be", "this", "that", "it", "on", "with", "as", "or", "and",
            "if", "do", "does", "did", "has", "have", "had", "not", "no",
            "yes", "market", "resolve", "question",
        }
        tokens = re.findall(r'\b[a-z]+\b', text_lower)
        return [t for t in tokens if t not in stop_words and len(t) > 2]


# ─────────────────────────────────────────────
# DEPENDENCY SIGNALS FROM ENTITIES
# ─────────────────────────────────────────────

def compute_entity_overlap(e1: MarketEntities, e2: MarketEntities) -> dict:
    """
    Compute overlap metrics between two extracted entity sets.

    These metrics become features for the ML classifier.

    Returns dict with:
        - entity_jaccard: Overall Jaccard similarity of all entities
        - shared_person_count: Number of shared person entities
        - shared_location_count: Number of shared locations
        - shared_org_count: Number of shared organizations
        - threshold_containment: Whether one threshold range contains the other
        - same_event_type: Boolean, same event classification
        - same_event_subtype: Boolean, same subtype (presidential vs senate)
        - same_direction: Boolean, same direction (win/lose)
        - negation_mismatch: One is opposite framing of the other
    """
    # Jaccard on all entities combined
    all_e1 = set(e1.persons + e1.locations + e1.organizations + e1.teams)
    all_e2 = set(e2.persons + e2.locations + e2.organizations + e2.teams)
    union = all_e1 | all_e2
    intersection = all_e1 & all_e2
    entity_jaccard = len(intersection) / len(union) if union else 0.0

    # Per-type overlap
    shared_persons = set(e1.persons) & set(e2.persons)
    shared_locations = set(e1.locations) & set(e2.locations)
    shared_orgs = set(e1.organizations) & set(e2.organizations)

    # Threshold containment: does one range contain the other?
    threshold_containment = _check_threshold_containment(e1, e2)

    # Direction analysis
    same_direction = e1.direction == e2.direction and e1.direction != ""
    negation_mismatch = (
        (e1.direction in ("win",) and e2.direction in ("lose",)) or
        (e1.direction in ("lose",) and e2.direction in ("win",)) or
        (
            bool(shared_orgs) and len(e1.organizations) > 0 and len(e2.organizations) > 0
            and e1.organizations[0] != e2.organizations[0]  # Different parties
        )
    )

    # Key token overlap
    token_jaccard = 0.0
    t1 = set(e1.key_tokens)
    t2 = set(e2.key_tokens)
    if t1 | t2:
        token_jaccard = len(t1 & t2) / len(t1 | t2)

    return {
        "entity_jaccard": entity_jaccard,
        "shared_person_count": len(shared_persons),
        "shared_location_count": len(shared_locations),
        "shared_org_count": len(shared_orgs),
        "threshold_containment": float(threshold_containment),
        "same_event_type": float(e1.event_type == e2.event_type and e1.event_type != ""),
        "same_event_subtype": float(
            e1.event_subtype == e2.event_subtype and e1.event_subtype != ""
        ),
        "same_direction": float(same_direction),
        "negation_mismatch": float(negation_mismatch),
        "question_token_jaccard": token_jaccard,
    }


def _check_threshold_containment(e1: MarketEntities, e2: MarketEntities) -> bool:
    """
    Check if one market's threshold range is contained within the other's.

    Example: "wins by 2-3%" is contained within "wins" (no threshold = all margins).
    Example: "wins by 2-3%" is NOT contained within "wins by 4-5%".
    """
    # If one has thresholds and the other doesn't, the one without is broader
    has_t1 = e1.threshold_min is not None or e1.threshold_max is not None
    has_t2 = e2.threshold_min is not None or e2.threshold_max is not None

    if has_t1 and not has_t2:
        return True  # e2 is broader, contains e1
    if has_t2 and not has_t1:
        return True  # e1 is broader, contains e2
    if not has_t1 and not has_t2:
        return False  # Neither has thresholds

    # Both have thresholds: check range containment
    min1 = e1.threshold_min if e1.threshold_min is not None else float("-inf")
    max1 = e1.threshold_max if e1.threshold_max is not None else float("inf")
    min2 = e2.threshold_min if e2.threshold_min is not None else float("-inf")
    max2 = e2.threshold_max if e2.threshold_max is not None else float("inf")

    # e1 contained in e2?
    if min2 <= min1 and max1 <= max2:
        return True
    # e2 contained in e1?
    if min1 <= min2 and max2 <= max1:
        return True
    # Overlapping ranges also signal dependency
    if min1 <= max2 and min2 <= max1:
        return True

    return False

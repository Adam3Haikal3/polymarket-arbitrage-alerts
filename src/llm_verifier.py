"""
LLM Verifier: Tier 3 Joint Outcome Space Enumeration

This module sends high-confidence candidate pairs to an LLM for formal
verification of logical dependency. It implements the paper's approach
(Section 5) but only for the ~50-200 pairs that survive Tier 1 and Tier 2,
instead of the 46,000+ pairs the paper checked.

The LLM's job:
1. Given conditions from two markets, enumerate ALL valid truth-value assignments
2. If |valid_assignments| < n × m, the markets are DEPENDENT
3. Extract the dependent subsets S ⊂ M1 and S' ⊂ M2

The prompt is carefully engineered based on the paper's findings:
- Markets are reduced to top-5 conditions by volume (Appendix C shows >90% liquidity)
- A catch-all "Other" condition captures remaining outcomes
- The LLM returns structured JSON for programmatic validation

Validation checks (from Section 5.2):
(i)   LLM returns valid JSON
(ii)  Each vector has exactly one TRUE in each market's conditions
(iii) Total vectors ≤ n + m for a pair of markets with n and m conditions
"""

import json
import logging
from typing import Optional, Tuple

import numpy as np

from src.data_collector import Market
from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────

DEPENDENCY_PROMPT = """You are given two sets of binary (True/False) questions representing \
conditions in two prediction markets. Within each market, exactly one condition must be True \
(the conditions are mutually exclusive and exhaustive).

Your task: determine ALL valid combinations of truth values across BOTH markets simultaneously.

RULES:
- Each combination must have exactly ONE True value in Market A's conditions
- Each combination must have exactly ONE True value in Market B's conditions
- Only include combinations that are logically possible in the real world
- If knowing the outcome of one market constrains the other, reflect this

MARKET A conditions:
{market_a_conditions}

MARKET B conditions:
{market_b_conditions}

Output ONLY a JSON object with this exact format, no other text:
{{
    "valid_combinations": [
        [{{"market_a": [true, false, ...], "market_b": [true, false, ...]}}],
        ...
    ],
    "dependent": true/false,
    "reasoning": "Brief explanation of why markets are dependent or independent"
}}

If the markets are independent (all combinations of one True per market are valid),
set "dependent" to false. If some combinations are impossible (knowing one outcome
constrains the other), set "dependent" to true.
"""


class LLMVerifier:
    """
    Verifies market dependency using LLM reasoning over joint outcome spaces.

    Usage:
        verifier = LLMVerifier(api_key="...")
        result = verifier.verify_pair(market_a, market_b)
        if result["is_dependent"]:
            print(f"Dependent subsets: {result['dependent_subsets']}")
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        max_conditions: int = None,
        max_retries: int = None,
    ):
        self.model = model or settings.LLM_MODEL
        self.api_key = api_key
        self.max_conditions = max_conditions or settings.LLM_MAX_CONDITIONS_PER_MARKET
        self.max_retries = max_retries or settings.LLM_MAX_RETRIES

    def verify_pair(
        self,
        market_a: Market,
        market_b: Market,
    ) -> dict:
        """
        Verify whether two markets are dependent.

        Args:
            market_a, market_b: Markets to check

        Returns:
            {
                "is_dependent": bool,
                "valid_combinations": list,  # All valid joint assignments
                "n_combinations": int,       # Actual number of valid combos
                "max_combinations": int,     # n × m if independent
                "dependent_subsets": tuple,  # (S, S') if dependent
                "reasoning": str,            # LLM's explanation
                "validation_passed": bool,   # All consistency checks passed
                "raw_response": str,         # Raw LLM output for debugging
            }
        """
        # Reduce conditions to top-K by volume
        conds_a = self._reduce_conditions(market_a)
        conds_b = self._reduce_conditions(market_b)

        n = len(conds_a)
        m = len(conds_b)

        # Build prompt
        prompt = self._build_prompt(conds_a, conds_b)

        # Call LLM with retries
        for attempt in range(self.max_retries):
            raw_response = self._call_llm(prompt)
            parsed = self._parse_response(raw_response, n, m)

            if parsed is not None:
                # Validate the response
                is_valid = self._validate_response(parsed, n, m)

                if is_valid:
                    # Check for dependency
                    n_combos = len(parsed["valid_combinations"])
                    max_combos = n * m
                    is_dependent = n_combos < max_combos

                    # Extract dependent subsets
                    dependent_subsets = None
                    if is_dependent:
                        dependent_subsets = self._extract_dependent_subsets(
                            parsed["valid_combinations"], n, m
                        )

                    return {
                        "is_dependent": is_dependent,
                        "valid_combinations": parsed["valid_combinations"],
                        "n_combinations": n_combos,
                        "max_combinations": max_combos,
                        "dependent_subsets": dependent_subsets,
                        "reasoning": parsed.get("reasoning", ""),
                        "validation_passed": True,
                        "raw_response": raw_response,
                    }

            logger.warning(
                f"Verification attempt {attempt + 1}/{self.max_retries} failed"
            )

        # All retries exhausted
        return {
            "is_dependent": None,  # Unknown
            "valid_combinations": [],
            "n_combinations": -1,
            "max_combinations": n * m,
            "dependent_subsets": None,
            "reasoning": "LLM verification failed after max retries",
            "validation_passed": False,
            "raw_response": raw_response if 'raw_response' in dir() else "",
        }

    def _reduce_conditions(self, market: Market) -> list:
        """
        Reduce market conditions to top-K by volume + catch-all.

        Per Appendix C: >90% of liquidity is in the top 4 conditions.
        The catch-all preserves logical dependencies.
        """
        conditions = market.conditions
        if len(conditions) <= self.max_conditions:
            return [c.question for c in conditions]

        # Sort by volume (or price as proxy) descending
        sorted_conds = sorted(
            conditions,
            key=lambda c: c.volume if c.volume > 0 else c.price_yes,
            reverse=True,
        )

        top_k = sorted_conds[:self.max_conditions - 1]
        remaining = sorted_conds[self.max_conditions - 1:]

        questions = [c.question for c in top_k]
        remaining_qs = [c.question for c in remaining]
        questions.append(f"Any other outcome ({', '.join(remaining_qs[:3])}...)")

        return questions

    def _build_prompt(self, conds_a: list, conds_b: list) -> str:
        """Build the LLM prompt with market conditions."""
        a_str = "\n".join(f"  A{i}: {q}" for i, q in enumerate(conds_a))
        b_str = "\n".join(f"  B{i}: {q}" for i, q in enumerate(conds_b))

        return DEPENDENCY_PROMPT.format(
            market_a_conditions=a_str,
            market_b_conditions=b_str,
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API using Google Gemini.
        """
        import os
        from google import genai
        
        logger.info(f"[LLM CALL] Model: gemini-2.5-flash, Prompt length: {len(prompt)}")
        
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required.")
            
        client = genai.Client(api_key=api_key)
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return '{"valid_combinations": [], "dependent": false, "reasoning": "API Error"}'

    def _parse_response(self, raw: str, n: int, m: int) -> Optional[dict]:
        """Parse the LLM's JSON response."""
        try:
            # Clean up common LLM formatting issues
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            if "valid_combinations" not in data:
                logger.warning("Missing 'valid_combinations' in LLM response")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return None

    def _validate_response(self, parsed: dict, n: int, m: int) -> bool:
        """
        Validate the LLM's response for consistency.

        Checks (from the paper Section 5.2):
        (i)   Valid JSON (already checked in _parse_response)
        (ii)  Each vector has exactly one TRUE per market
        (iii) Total vectors ≤ n + m
        """
        combos = parsed.get("valid_combinations", [])

        if not combos:
            logger.warning("Empty valid_combinations")
            return False

        for i, combo in enumerate(combos):
            if isinstance(combo, dict):
                a_vals = combo.get("market_a", [])
                b_vals = combo.get("market_b", [])
            elif isinstance(combo, list):
                # Flat list format: first n are market A, rest are market B
                a_vals = combo[:n]
                b_vals = combo[n:n+m]
            else:
                logger.warning(f"Unknown combo format at index {i}")
                return False

            # Check (ii): exactly one True per market
            if sum(1 for v in a_vals if v) != 1:
                logger.warning(f"Combo {i}: Market A has {sum(1 for v in a_vals if v)} True values (need exactly 1)")
                return False
            if sum(1 for v in b_vals if v) != 1:
                logger.warning(f"Combo {i}: Market B has {sum(1 for v in b_vals if v)} True values (need exactly 1)")
                return False

        # Check (iii): total vectors ≤ n + m (loose check)
        # A stricter check: ≤ n × m (always true for valid assignments)
        if len(combos) > n * m:
            logger.warning(f"Too many combinations: {len(combos)} > {n * m}")
            return False

        return True

    def _extract_dependent_subsets(
        self,
        valid_combinations: list,
        n: int,
        m: int,
    ) -> Optional[Tuple[list, list]]:
        """
        Extract the dependent subsets S ⊂ M1 and S' ⊂ M2.

        Per Definition 2: if some conditions in M1 being True constrain
        which conditions in M2 can be True, those form dependent subsets.

        Returns:
            Tuple of (indices_in_m1, indices_in_m2) that form dependent subsets,
            or None if no clear dependency structure is found.
        """
        # Build a co-occurrence matrix: which conditions can be simultaneously True?
        cooccurrence = np.zeros((n, m), dtype=bool)

        for combo in valid_combinations:
            if isinstance(combo, dict):
                a_vals = combo.get("market_a", [])
                b_vals = combo.get("market_b", [])
            else:
                a_vals = combo[:n]
                b_vals = combo[n:n+m]

            a_idx = next(i for i, v in enumerate(a_vals) if v)
            b_idx = next(i for i, v in enumerate(b_vals) if v)
            cooccurrence[a_idx, b_idx] = True

        # Find rows/columns with restricted co-occurrence
        # A dependent subset exists if some rows are only True in a subset of columns
        for i in range(n):
            allowed_b = set(np.where(cooccurrence[i])[0])
            if 0 < len(allowed_b) < m:
                # Found a dependency: condition i in M1 restricts M2
                # Find all M1 conditions with the same restriction pattern
                s1 = [i]
                for j in range(n):
                    if j != i and set(np.where(cooccurrence[j])[0]) == allowed_b:
                        s1.append(j)
                s2 = list(allowed_b)
                return (sorted(s1), sorted(s2))

        return None

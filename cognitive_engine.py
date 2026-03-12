"""
cognitive_engine.py
───────────────────
Type 1 System: Hot Hand Cognitive Bias Simulator
CS6795 Term Project | Jason Schwartz | Georgia Tech OMSCS

Models a biased human observer who overweights recent shots when estimating
a player's probability of making the next shot. This simulates the
Representativeness Heuristic and Recency Bias described in Gilovich,
Vallone & Tversky (1985).

Theory
------
Instead of equal-weighted Bayesian updating, the cognitive engine uses an
exponentially decaying weighted average. Each shot's contribution to the
current belief decays geometrically with distance from the present:

    weight(k) = λ^k     where k = shots ago, λ ∈ (0, 1)

A λ close to 1.0 → slow decay, longer memory (more rational-like)
A λ close to 0.0 → fast decay, only the very last shot matters (extreme bias)

The "Heuristic Strength" parameter (0.0–1.0) controls λ via:
    λ = 1 - heuristic_strength

This maps intuitively: strength=0 is fully rational, strength=1 is
completely recency-dominated.

The weighted belief is then blended with the long-term prior using a
"prior anchor weight" to prevent the estimate from drifting to 0 or 1
on short streaks — reflecting that even a biased observer retains some
awareness of the player's baseline ability.

Reset Behavior
--------------
The shot history buffer resets at the start of each new game, but the
long-term prior is always retained (it is never observed-shot-dependent).
"""

import math


class CognitiveEngine:
    """
    Hot Hand bias simulator using exponential decay weighting.

    Parameters
    ----------
    prior_fg_pct : float
        The player's long-term field goal percentage. Used as the anchor
        prior and as the initial belief before any shots are seen.
    heuristic_strength : float
        Controls how aggressively recent shots are overweighted.
        Range: 0.0 (rational, no decay) → 1.0 (only last shot matters).
        Default 0.7 represents a moderately biased fan/observer.
    prior_anchor : float
        Weight given to the long-term prior vs. the recency-weighted
        in-game history. Range 0.0–1.0. Default 0.3 means the prior
        contributes 30% of the belief, recency history 70%.

    Usage
    -----
    engine = CognitiveEngine(prior_fg_pct=0.474, heuristic_strength=0.7)
    engine.new_game()
    p = engine.current_belief
    engine.update(shot_made=1)
    """

    def __init__(
        self,
        prior_fg_pct      : float,
        heuristic_strength: float = 0.7,
        prior_anchor      : float = 0.3,
    ):
        if not 0 < prior_fg_pct < 1:
            raise ValueError("prior_fg_pct must be between 0 and 1 (exclusive)")
        if not 0.0 <= heuristic_strength <= 1.0:
            raise ValueError("heuristic_strength must be in [0.0, 1.0]")
        if not 0.0 <= prior_anchor <= 1.0:
            raise ValueError("prior_anchor must be in [0.0, 1.0]")

        self.prior_fg_pct       = prior_fg_pct
        self.heuristic_strength = heuristic_strength
        self.prior_anchor       = prior_anchor

        # λ: decay factor per shot. Higher λ = slower decay = longer memory
        # At strength=0 → λ=1.0 (no decay, rational-like)
        # At strength=1 → λ=0.0 (only last shot)
        self.decay_lambda = 1.0 - heuristic_strength

        # In-game shot buffer — list of 1s and 0s, most recent LAST
        # Resets each game
        self._shot_buffer: list[int] = []

        # History log for analysis
        self.history: list[dict] = []

    # ── Public interface ───────────────────────────────────────────────────

    def new_game(self) -> None:
        """
        Clear the in-game shot buffer at the start of each new game.
        The long-term prior is always retained regardless.
        """
        self._shot_buffer = []

    def update(self, shot_made: int, metadata: dict | None = None) -> float:
        """
        Incorporate a new shot outcome and return the updated biased belief.

        Parameters
        ----------
        shot_made : int
            1 if the shot was made, 0 if missed.
        metadata : dict, optional
            Extra fields to log for analysis.

        Returns
        -------
        float
            The heuristic-weighted probability estimate AFTER this shot.
        """
        belief_before = self.current_belief

        # Append outcome to in-game buffer
        self._shot_buffer.append(shot_made)

        belief_after = self.current_belief

        # Record
        record = {
            "belief_before"     : round(belief_before,  6),
            "belief_after"      : round(belief_after,   6),
            "shot_made"         : shot_made,
            "heuristic_strength": self.heuristic_strength,
            "decay_lambda"      : self.decay_lambda,
            "shots_in_buffer"   : len(self._shot_buffer),
            "streak"            : self._current_streak(),
        }
        if metadata:
            record.update(metadata)
        self.history.append(record)

        return belief_after

    @property
    def current_belief(self) -> float:
        """
        Compute the current biased probability estimate.

        Formula:
            recency_est = Σ(weight_k * outcome_k) / Σ(weight_k)
            belief = prior_anchor * prior_fg_pct
                   + (1 - prior_anchor) * recency_est

        If no shots have been seen yet, returns the prior.
        """
        if not self._shot_buffer:
            return self.prior_fg_pct

        # Exponential weights: most recent shot gets weight λ^0 = 1,
        # shot before that gets λ^1, two shots ago gets λ^2, etc.
        n = len(self._shot_buffer)
        weights = [self.decay_lambda ** k for k in range(n - 1, -1, -1)]
        # Note: range(n-1, -1, -1) assigns weight λ^(n-1) to oldest,
        # λ^0 = 1 to most recent

        weighted_makes = sum(w * s for w, s in zip(weights, self._shot_buffer))
        total_weight   = sum(weights)

        recency_estimate = weighted_makes / total_weight if total_weight > 0 else self.prior_fg_pct

        # Blend recency estimate with the long-term prior anchor
        belief = (
            self.prior_anchor * self.prior_fg_pct
            + (1 - self.prior_anchor) * recency_estimate
        )

        # Clamp to valid probability range
        return max(0.01, min(0.99, belief))

    def _current_streak(self) -> int:
        """
        Return the current consecutive make/miss streak from the buffer.
        Positive = makes, negative = misses.
        """
        if not self._shot_buffer:
            return 0
        streak = 0
        last = self._shot_buffer[-1]
        for outcome in reversed(self._shot_buffer):
            if outcome == last:
                streak += 1 if last == 1 else -1
            else:
                break
        return streak

    def summary(self) -> dict:
        """Return a summary of the engine's current state."""
        return {
            "prior_fg_pct"      : self.prior_fg_pct,
            "heuristic_strength": self.heuristic_strength,
            "decay_lambda"      : self.decay_lambda,
            "prior_anchor"      : self.prior_anchor,
            "current_belief"    : round(self.current_belief, 4),
            "shots_in_buffer"   : len(self._shot_buffer),
            "current_streak"    : self._current_streak(),
            "shots_logged"      : len(self.history),
        }

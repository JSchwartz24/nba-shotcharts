"""
rational_engine.py
──────────────────
Type 2 System: Rational Bayesian Observer
CS6795 Term Project | Jason Schwartz | Georgia Tech OMSCS

Models a statistically rational observer who updates their belief about a
player's shooting probability using Bayesian inference. This acts as the
"ground truth" control against which the Hot Hand bias is measured.

Theory
------
We model each shot as a Bernoulli trial with unknown probability θ (true FG%).
Using a Beta-Binomial conjugate model:

    Prior:     θ ~ Beta(α₀, β₀)
    Likelihood: each shot is Bernoulli(θ)
    Posterior: θ ~ Beta(α₀ + makes, β₀ + misses)

The posterior mean E[θ] = α / (α + β) is used as the probability estimate
after each shot. This is the cleanest Bayesian formulation for binary outcomes
and is mathematically equivalent to updating a running proportion, but with
a principled prior that prevents overreaction to small samples.

Reset Behavior
--------------
Belief resets to the season-level prior at the start of each new game,
reflecting that a rational observer treats each game as a fresh context
while still anchoring to long-term historical performance.
"""

class RationalEngine:
    """
    Bayesian belief updater using a Beta-Binomial conjugate model.

    Parameters
    ----------
    prior_makes : float
        Alpha parameter of the Beta prior. Set to the player's historical
        makes (or a pseudo-count representing long-term FG%).
    prior_misses : float
        Beta parameter of the Beta prior. Set to the player's historical
        misses (or a pseudo-count).

    Usage
    -----
    engine = RationalEngine(prior_makes=47, prior_misses=53)
    engine.new_game()          # call at the start of each game
    p = engine.current_belief  # get current probability estimate
    engine.update(shot_made=1) # feed in outcome (1=make, 0=miss)
    """

    def __init__(self, prior_makes: float, prior_misses: float):
        if prior_makes <= 0 or prior_misses <= 0:
            raise ValueError("Prior parameters must be positive (Beta distribution requires α > 0, β > 0)")

        self.prior_alpha = prior_makes   # α₀ — encodes long-term FG%
        self.prior_beta  = prior_misses  # β₀ — encodes long-term miss rate

        # Running posterior parameters (reset each game)
        self.alpha = self.prior_alpha
        self.beta  = self.prior_beta

        # History log for analysis
        self.history: list[dict] = []

    # ── Public interface ───────────────────────────────────────────────────

    def new_game(self) -> None:
        """
        Reset posterior to the season prior at the start of a new game.
        The rational observer anchors back to long-term performance between
        games, not carrying in-game streaks across game boundaries.
        """
        self.alpha = self.prior_alpha
        self.beta  = self.prior_beta

    def update(self, shot_made: int, metadata: dict | None = None) -> float:
        """
        Incorporate a new shot outcome and return the updated belief.

        Parameters
        ----------
        shot_made : int
            1 if the shot was made, 0 if missed.
        metadata : dict, optional
            Any extra fields (e.g. shot_number, game_id) to log alongside
            the belief for later analysis.

        Returns
        -------
        float
            The posterior mean probability AFTER incorporating this shot.
        """
        # Record belief BEFORE this shot (what the observer believed going in)
        belief_before = self.current_belief

        # Bayesian update: Beta posterior is just incremented counts
        if shot_made == 1:
            self.alpha += 1
        else:
            self.beta += 1

        belief_after = self.current_belief

        # Log the transition
        record = {
            "belief_before": round(belief_before, 6),
            "belief_after" : round(belief_after,  6),
            "shot_made"    : shot_made,
            "alpha"        : self.alpha,
            "beta"         : self.beta,
        }
        if metadata:
            record.update(metadata)
        self.history.append(record)

        return belief_after

    @property
    def current_belief(self) -> float:
        """Posterior mean: E[θ] = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self) -> float:
        """
        Posterior variance: Var[θ] = αβ / ((α+β)²(α+β+1))
        Higher variance = less confident estimate.
        """
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def summary(self) -> dict:
        """Return a summary of the engine's current state."""
        return {
            "prior_fg_pct"  : round(self.prior_alpha / (self.prior_alpha + self.prior_beta), 4),
            "current_belief": round(self.current_belief, 4),
            "uncertainty"   : round(self.uncertainty, 6),
            "alpha"         : self.alpha,
            "beta"          : self.beta,
            "shots_seen"    : len(self.history),
        }

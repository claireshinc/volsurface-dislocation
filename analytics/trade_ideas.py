"""Generate trade ideas based on detected volatility anomalies.

Analyzes anomalies in volatility surface features and generates
actionable trade suggestions with supporting rationale.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from analytics.anomaly_detector import AnomalyReport, AnomalyResult

logger = logging.getLogger(__name__)


class TradeDirection(str, Enum):
    """Trade direction enumeration."""

    LONG_VOL = "long_vol"
    SHORT_VOL = "short_vol"
    LONG_SKEW = "long_skew"
    SHORT_SKEW = "short_skew"
    LONG_CONVEXITY = "long_convexity"
    SHORT_CONVEXITY = "short_convexity"
    CALENDAR = "calendar"


class TradeConfidence(str, Enum):
    """Trade confidence level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TradeIdea:
    """A trade suggestion based on volatility analysis.

    Attributes:
        ticker: Underlying symbol
        timestamp: When idea was generated
        direction: Trade direction
        strategy: Suggested options strategy
        description: Human-readable description
        rationale: Reasoning behind the trade
        confidence: Confidence level
        z_score: Z-score of the triggering anomaly
        percentile: Percentile rank
        entry_criteria: Conditions for entry
        exit_criteria: Conditions for exit
        risks: Key risks to monitor
    """

    ticker: str
    timestamp: datetime
    direction: TradeDirection
    strategy: str
    description: str
    rationale: str
    confidence: TradeConfidence
    z_score: float
    percentile: float
    entry_criteria: list[str]
    exit_criteria: list[str]
    risks: list[str]
    related_anomalies: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "strategy": self.strategy,
            "description": self.description,
            "rationale": self.rationale,
            "confidence": self.confidence.value,
            "z_score": self.z_score,
            "percentile": self.percentile,
            "entry_criteria": self.entry_criteria,
            "exit_criteria": self.exit_criteria,
            "risks": self.risks,
            "related_anomalies": self.related_anomalies,
        }


class TradeIdeaGenerator:
    """Generates trade ideas from anomaly reports.

    Analyzes patterns in volatility anomalies and generates specific
    trade suggestions with entry/exit criteria and risk factors.

    Example:
        >>> generator = TradeIdeaGenerator()
        >>> ideas = generator.generate(anomaly_report)
        >>> for idea in ideas:
        ...     print(f"{idea.strategy}: {idea.description}")
    """

    # Z-score thresholds for confidence levels
    HIGH_CONFIDENCE_THRESHOLD = 2.5
    MEDIUM_CONFIDENCE_THRESHOLD = 2.0

    def __init__(self, min_zscore: float = 2.0):
        """Initialize trade idea generator.

        Args:
            min_zscore: Minimum z-score to generate a trade idea
        """
        self.min_zscore = min_zscore

    def _get_confidence(self, z_score: float) -> TradeConfidence:
        """Determine confidence level from z-score."""
        abs_z = abs(z_score)
        if abs_z >= self.HIGH_CONFIDENCE_THRESHOLD:
            return TradeConfidence.HIGH
        elif abs_z >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return TradeConfidence.MEDIUM
        return TradeConfidence.LOW

    def _generate_atm_vol_idea(
        self,
        anomaly: AnomalyResult,
        ticker: str,
    ) -> TradeIdea | None:
        """Generate trade idea for ATM vol anomaly."""
        if not anomaly.is_anomaly:
            return None

        tenor = anomaly.feature_name.replace("atm_vol_", "").replace("d", "")

        if anomaly.direction == "high":
            return TradeIdea(
                ticker=ticker,
                timestamp=datetime.now(),
                direction=TradeDirection.SHORT_VOL,
                strategy=f"Sell {tenor}-day straddle",
                description=f"Sell ATM straddle - {tenor}d vol is historically rich",
                rationale=(
                    f"ATM {tenor}d volatility at {anomaly.current_value:.1%} is "
                    f"{anomaly.z_score:.1f} standard deviations above the 252-day mean "
                    f"of {anomaly.historical_mean:.1%}. Historically, this level "
                    f"represents the {anomaly.percentile:.0f}th percentile."
                ),
                confidence=self._get_confidence(anomaly.z_score),
                z_score=anomaly.z_score,
                percentile=anomaly.percentile,
                entry_criteria=[
                    f"Current {tenor}d ATM vol > {anomaly.historical_mean + 2*anomaly.historical_std:.1%}",
                    "No major events in the expiry window",
                    "VIX term structure in contango",
                ],
                exit_criteria=[
                    f"{tenor}d ATM vol returns to {anomaly.historical_mean + anomaly.historical_std:.1%}",
                    "Loss exceeds 50% of premium received",
                    "5 days before expiration",
                ],
                risks=[
                    "Gap risk on overnight moves",
                    "Earnings/macro events causing vol spike",
                    "Unlimited loss potential on wings",
                ],
                related_anomalies=[anomaly.feature_name],
            )

        elif anomaly.direction == "low":
            return TradeIdea(
                ticker=ticker,
                timestamp=datetime.now(),
                direction=TradeDirection.LONG_VOL,
                strategy=f"Buy {tenor}-day straddle",
                description=f"Buy ATM straddle - {tenor}d vol is historically cheap",
                rationale=(
                    f"ATM {tenor}d volatility at {anomaly.current_value:.1%} is "
                    f"{abs(anomaly.z_score):.1f} standard deviations below the 252-day mean "
                    f"of {anomaly.historical_mean:.1%}. Historically, this level "
                    f"represents only the {anomaly.percentile:.0f}th percentile."
                ),
                confidence=self._get_confidence(anomaly.z_score),
                z_score=anomaly.z_score,
                percentile=anomaly.percentile,
                entry_criteria=[
                    f"Current {tenor}d ATM vol < {anomaly.historical_mean - 2*anomaly.historical_std:.1%}",
                    "Realized vol showing signs of increase",
                    "Market approaching key support/resistance",
                ],
                exit_criteria=[
                    f"{tenor}d ATM vol returns to {anomaly.historical_mean - anomaly.historical_std:.1%}",
                    "Loss exceeds 50% of premium paid",
                    "Vol spike provides profitable exit",
                ],
                risks=[
                    "Time decay if vol remains low",
                    "Premium paid may decay quickly",
                    "Carry cost in low-vol environment",
                ],
                related_anomalies=[anomaly.feature_name],
            )

        return None

    def _generate_skew_idea(
        self,
        anomaly: AnomalyResult,
        ticker: str,
    ) -> TradeIdea | None:
        """Generate trade idea for skew anomaly."""
        if not anomaly.is_anomaly:
            return None

        tenor = anomaly.feature_name.replace("skew_25d_", "").replace("d", "")

        if anomaly.direction == "high":
            # Skew is steep (puts expensive relative to calls)
            return TradeIdea(
                ticker=ticker,
                timestamp=datetime.now(),
                direction=TradeDirection.SHORT_SKEW,
                strategy=f"Sell {tenor}-day risk reversal",
                description=f"Sell puts / buy calls - {tenor}d skew is historically steep",
                rationale=(
                    f"25-delta skew at {tenor}d of {anomaly.current_value:.2%} is "
                    f"{anomaly.z_score:.1f} standard deviations above the mean of "
                    f"{anomaly.historical_mean:.2%}. Put protection is expensive "
                    f"relative to historical norms ({anomaly.percentile:.0f}th percentile)."
                ),
                confidence=self._get_confidence(anomaly.z_score),
                z_score=anomaly.z_score,
                percentile=anomaly.percentile,
                entry_criteria=[
                    f"Skew > {anomaly.historical_mean + 2*anomaly.historical_std:.2%}",
                    "No imminent catalysts for downside move",
                    "Market sentiment overly bearish",
                ],
                exit_criteria=[
                    f"Skew normalizes to {anomaly.historical_mean:.2%}",
                    "Underlying drops significantly",
                    "Loss exceeds 2x premium collected",
                ],
                risks=[
                    "Market selloff causes skew to steepen further",
                    "Short put exposure to downside gap",
                    "Skew can stay elevated for extended periods",
                ],
                related_anomalies=[anomaly.feature_name],
            )

        elif anomaly.direction == "low":
            # Skew is flat (puts cheap relative to calls)
            return TradeIdea(
                ticker=ticker,
                timestamp=datetime.now(),
                direction=TradeDirection.LONG_SKEW,
                strategy=f"Buy {tenor}-day risk reversal",
                description=f"Buy puts / sell calls - {tenor}d skew is historically flat",
                rationale=(
                    f"25-delta skew at {tenor}d of {anomaly.current_value:.2%} is "
                    f"{abs(anomaly.z_score):.1f} standard deviations below the mean of "
                    f"{anomaly.historical_mean:.2%}. Put protection is historically cheap "
                    f"({anomaly.percentile:.0f}th percentile)."
                ),
                confidence=self._get_confidence(anomaly.z_score),
                z_score=anomaly.z_score,
                percentile=anomaly.percentile,
                entry_criteria=[
                    f"Skew < {anomaly.historical_mean - 2*anomaly.historical_std:.2%}",
                    "Hedging demand is light",
                    "Market complacency evident",
                ],
                exit_criteria=[
                    f"Skew normalizes to {anomaly.historical_mean:.2%}",
                    "Protection captured on selloff",
                    "Time decay erodes position value",
                ],
                risks=[
                    "Continued low vol environment",
                    "Skew normalization may take time",
                    "Short call limits upside participation",
                ],
                related_anomalies=[anomaly.feature_name],
            )

        return None

    def _generate_butterfly_idea(
        self,
        anomaly: AnomalyResult,
        ticker: str,
    ) -> TradeIdea | None:
        """Generate trade idea for butterfly/convexity anomaly."""
        if not anomaly.is_anomaly:
            return None

        tenor = anomaly.feature_name.replace("butterfly_", "").replace("d", "")

        if anomaly.direction == "high":
            # Wings are expensive (high convexity)
            return TradeIdea(
                ticker=ticker,
                timestamp=datetime.now(),
                direction=TradeDirection.SHORT_CONVEXITY,
                strategy=f"Sell {tenor}-day iron butterfly",
                description=f"Sell wings - {tenor}d convexity is historically high",
                rationale=(
                    f"Butterfly spread at {tenor}d of {anomaly.current_value:.2%} indicates "
                    f"wings are {anomaly.z_score:.1f}σ above the historical mean. "
                    f"The market is pricing excessive tail risk ({anomaly.percentile:.0f}th percentile)."
                ),
                confidence=self._get_confidence(anomaly.z_score),
                z_score=anomaly.z_score,
                percentile=anomaly.percentile,
                entry_criteria=[
                    f"Butterfly > {anomaly.historical_mean + 2*anomaly.historical_std:.2%}",
                    "Realized tails have been smaller than implied",
                    "No major binary events pending",
                ],
                exit_criteria=[
                    "Wing premium normalizes",
                    "Approaching expiration",
                    "Underlying approaches short wing strikes",
                ],
                risks=[
                    "Large moves cause wing losses",
                    "Convexity can expand further in crisis",
                    "Requires active management",
                ],
                related_anomalies=[anomaly.feature_name],
            )

        elif anomaly.direction == "low":
            # Wings are cheap (low convexity)
            return TradeIdea(
                ticker=ticker,
                timestamp=datetime.now(),
                direction=TradeDirection.LONG_CONVEXITY,
                strategy=f"Buy {tenor}-day iron butterfly",
                description=f"Buy wings - {tenor}d convexity is historically low",
                rationale=(
                    f"Butterfly spread at {tenor}d of {anomaly.current_value:.2%} indicates "
                    f"wings are {abs(anomaly.z_score):.1f}σ below the historical mean. "
                    f"Tail protection is cheap ({anomaly.percentile:.0f}th percentile)."
                ),
                confidence=self._get_confidence(anomaly.z_score),
                z_score=anomaly.z_score,
                percentile=anomaly.percentile,
                entry_criteria=[
                    f"Butterfly < {anomaly.historical_mean - 2*anomaly.historical_std:.2%}",
                    "Tail risk may be underpriced",
                    "Macro uncertainty is elevated",
                ],
                exit_criteria=[
                    "Wing premium expands",
                    "Tail move occurs",
                    "Loss exceeds acceptable threshold",
                ],
                risks=[
                    "Time decay in low-vol environment",
                    "Wings may stay cheap",
                    "Need large move to profit",
                ],
                related_anomalies=[anomaly.feature_name],
            )

        return None

    def _generate_term_structure_idea(
        self,
        anomaly: AnomalyResult,
        ticker: str,
    ) -> TradeIdea | None:
        """Generate trade idea for term structure anomaly."""
        if not anomaly.is_anomaly:
            return None

        if anomaly.direction == "high":
            # Steep term structure (far vols high relative to near)
            return TradeIdea(
                ticker=ticker,
                timestamp=datetime.now(),
                direction=TradeDirection.CALENDAR,
                strategy="Sell calendar spread",
                description="Sell back month / buy front month - term structure is steep",
                rationale=(
                    f"Term slope of {anomaly.current_value:.1%} indicates long-dated vol "
                    f"is {anomaly.z_score:.1f}σ above normal relative to short-dated. "
                    f"Back month appears rich ({anomaly.percentile:.0f}th percentile)."
                ),
                confidence=self._get_confidence(anomaly.z_score),
                z_score=anomaly.z_score,
                percentile=anomaly.percentile,
                entry_criteria=[
                    f"Term slope > {anomaly.historical_mean + 2*anomaly.historical_std:.1%}",
                    "No long-dated catalysts justifying premium",
                    "Front month has specific catalyst",
                ],
                exit_criteria=[
                    "Term structure flattens",
                    "Front month expires/rolls",
                    "Position reaches target P&L",
                ],
                risks=[
                    "Long-dated event causes further steepening",
                    "Front month vol spike causes losses",
                    "Timing risk on mean reversion",
                ],
                related_anomalies=[anomaly.feature_name],
            )

        elif anomaly.direction == "low":
            # Flat/inverted term structure
            return TradeIdea(
                ticker=ticker,
                timestamp=datetime.now(),
                direction=TradeDirection.CALENDAR,
                strategy="Buy calendar spread",
                description="Buy back month / sell front month - term structure is flat",
                rationale=(
                    f"Term slope of {anomaly.current_value:.1%} indicates term structure is "
                    f"{abs(anomaly.z_score):.1f}σ below normal. Front month appears rich "
                    f"relative to back ({anomaly.percentile:.0f}th percentile)."
                ),
                confidence=self._get_confidence(anomaly.z_score),
                z_score=anomaly.z_score,
                percentile=anomaly.percentile,
                entry_criteria=[
                    f"Term slope < {anomaly.historical_mean - 2*anomaly.historical_std:.1%}",
                    "Near-term vol spike expected to normalize",
                    "No fundamental reason for inversion",
                ],
                exit_criteria=[
                    "Term structure normalizes",
                    "Front month vol collapses",
                    "Position reaches target P&L",
                ],
                risks=[
                    "Near-term catalyst causes front month spike",
                    "Inversion can persist",
                    "Time decay on back month if wrong",
                ],
                related_anomalies=[anomaly.feature_name],
            )

        return None

    def generate(self, report: AnomalyReport) -> list[TradeIdea]:
        """Generate trade ideas from anomaly report.

        Args:
            report: AnomalyReport to analyze

        Returns:
            List of TradeIdea objects

        Example:
            >>> generator = TradeIdeaGenerator()
            >>> ideas = generator.generate(anomaly_report)
            >>> high_confidence = [i for i in ideas if i.confidence == TradeConfidence.HIGH]
        """
        ideas = []
        ticker = report.ticker

        for anomaly in report.anomalies:
            if not anomaly.is_anomaly:
                continue

            if abs(anomaly.z_score) < self.min_zscore:
                continue

            idea = None

            # Route to appropriate idea generator based on feature type
            if anomaly.feature_name.startswith("atm_vol_"):
                idea = self._generate_atm_vol_idea(anomaly, ticker)

            elif anomaly.feature_name.startswith("skew_"):
                idea = self._generate_skew_idea(anomaly, ticker)

            elif anomaly.feature_name.startswith("butterfly_"):
                idea = self._generate_butterfly_idea(anomaly, ticker)

            elif anomaly.feature_name == "term_slope":
                idea = self._generate_term_structure_idea(anomaly, ticker)

            if idea:
                ideas.append(idea)

        # Sort by confidence and z-score
        ideas.sort(key=lambda x: (x.confidence != TradeConfidence.HIGH, -abs(x.z_score)))

        logger.info(f"Generated {len(ideas)} trade ideas for {ticker}")

        return ideas

    def summarize_ideas(self, ideas: list[TradeIdea]) -> dict[str, Any]:
        """Generate summary of trade ideas.

        Args:
            ideas: List of trade ideas

        Returns:
            Summary dictionary
        """
        if not ideas:
            return {"total": 0, "by_direction": {}, "by_confidence": {}}

        by_direction = {}
        by_confidence = {}

        for idea in ideas:
            dir_name = idea.direction.value
            by_direction[dir_name] = by_direction.get(dir_name, 0) + 1

            conf_name = idea.confidence.value
            by_confidence[conf_name] = by_confidence.get(conf_name, 0) + 1

        return {
            "total": len(ideas),
            "by_direction": by_direction,
            "by_confidence": by_confidence,
            "top_idea": ideas[0].to_dict() if ideas else None,
        }

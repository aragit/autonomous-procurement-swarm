"""Deterministic reservation price estimation from historical bid data."""

from dataclasses import dataclass


@dataclass
class SupplierProfile:
    """Historical performance snapshot for a single supplier."""

    supplier_id: str
    auctions_participated: int = 0
    auctions_won: int = 0
    avg_concession_slope: float = 0.0  # $ dropped per turn
    avg_margin_at_win: float = 0.0
    avg_final_price_vs_spot: float = 0.0
    reliability_score: float = 0.0
    concession_speed: str = "unknown"  # fast / medium / slow

    def to_dict(self) -> dict[str, object]:
        return {
            "supplier_id": self.supplier_id,
            "auctions_participated": self.auctions_participated,
            "auctions_won": self.auctions_won,
            "avg_concession_slope": round(self.avg_concession_slope, 2),
            "avg_margin_at_win": round(self.avg_margin_at_win, 4),
            "avg_final_price_vs_spot": round(self.avg_final_price_vs_spot, 4),
            "reliability_score": self.reliability_score,
            "concession_speed": self.concession_speed,
        }


class HeuristicReservationEstimator:
    """
    Estimates a supplier's reservation price floor from historical auction data.
    No Bayesian math — pure deterministic heuristics.
    """

    def __init__(self) -> None:
        self.profiles: dict[str, SupplierProfile] = {}

    def record_auction_result(
        self,
        supplier_id: str,
        original_bid_price: float,
        final_price: float | None,
        spot_price: float,
        turns_taken: int,
        won: bool,
    ) -> None:
        """Update supplier profile after an auction concludes."""
        if supplier_id not in self.profiles:
            self.profiles[supplier_id] = SupplierProfile(supplier_id=supplier_id)

        profile = self.profiles[supplier_id]
        profile.auctions_participated += 1

        if won and final_price is not None:
            profile.auctions_won += 1
            margin = (final_price - spot_price * 0.7) / max(spot_price * 0.7, 1)
            # Simple running average
            n = profile.auctions_won
            profile.avg_margin_at_win = (profile.avg_margin_at_win * (n - 1) + margin) / n
            profile.avg_final_price_vs_spot = (
                profile.avg_final_price_vs_spot * (n - 1) + final_price / spot_price
            ) / n

            # Concession slope: price drop per turn
            if turns_taken > 0 and original_bid_price > final_price:
                slope = (original_bid_price - final_price) / turns_taken
                if profile.avg_concession_slope == 0:
                    profile.avg_concession_slope = slope
                else:
                    profile.avg_concession_slope = (
                        profile.avg_concession_slope * 0.7 + slope * 0.3
                    )  # Exponential moving average

        # Classify concession speed
        if profile.avg_concession_slope > spot_price * 0.05:
            profile.concession_speed = "fast"
        elif profile.avg_concession_slope > spot_price * 0.01:
            profile.concession_speed = "medium"
        else:
            profile.concession_speed = "slow"

    def estimate_reservation_price(
        self,
        supplier_id: str,
        current_spot_price: float,
        default_margin_discount: float = 0.08,
    ) -> float:
        """
        Estimate floor price for opening negotiation.
        Uses historical margin if available, else spot-based heuristic.
        """
        profile = self.profiles.get(supplier_id)
        if not profile or profile.auctions_won == 0:
            # No history: assume they take spot minus standard margin
            return current_spot_price * (1.0 - default_margin_discount)

        # Historical average: they typically close at X% of spot
        estimated = current_spot_price * profile.avg_final_price_vs_spot

        # Adjust by concession speed: fast conceders have lower floors
        if profile.concession_speed == "fast":
            estimated *= 0.95
        elif profile.concession_speed == "slow":
            estimated *= 1.05

        # Floor: never below 60% of spot (safety)
        return max(estimated, current_spot_price * 0.6)

    def get_profile(self, supplier_id: str) -> SupplierProfile | None:
        return self.profiles.get(supplier_id)

    def all_profiles(self) -> list[SupplierProfile]:
        return list(self.profiles.values())

import math
from datetime import datetime, timezone


def get_market_multiplier(reference_time=None):
    reference_time = reference_time or datetime.now(timezone.utc)
    day_of_year = reference_time.timetuple().tm_yday
    # Smooth macro cycle across the year, oscillating between 0.7x and 1.3x growth.
    return round(1.0 + 0.3 * math.sin(2 * math.pi * day_of_year / 365), 3)


def get_market_label(multiplier):
    if multiplier > 1.15:
        return "booming"
    if multiplier > 1.0:
        return "favorable"
    if multiplier > 0.85:
        return "cooling"
    return "recessionary"

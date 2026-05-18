import json
import math
import random
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Product:
    product_id: str
    width: float
    length: float
    order_quantity: int = 0


@dataclass
class Defect:
    defect_id: str
    x: float
    y: float
    length: float
    width: float


@dataclass
class Stock:
    stock_id: str
    width: float
    length: float
    activation_cost: float
    defects: List[Defect]


@dataclass
class PatternStrip:
    product_id: str
    quantity: int
    x_left: float
    x_right: float


@dataclass
class Pattern:
    pattern_id: str
    length: float
    strips: List[PatternStrip]

    @property
    def produced_quantities(self) -> Dict[str, int]:
        out = defaultdict(int)
        for s in self.strips:
            out[s.product_id] += s.quantity
        return dict(out)

    @property
    def used_width(self) -> float:
        if not self.strips:
            return 0.0
        return max(s.x_right for s in self.strips)


# ---------------------------------------------------------------------------
# Size classes
# ---------------------------------------------------------------------------

# fixed size classes for instance size sweep
FIXED_SIZE_CONFIGS = {
    "XS": {"n_stocks": (5,   15),  "n_products": (2,  5),  "n_length_groups": (2, 3), "cap_per_product": 1000},
    "S" : {"n_stocks": (15,  30),  "n_products": (4,  8),  "n_length_groups": (2, 4), "cap_per_product": 1000},
    "M" : {"n_stocks": (30,  60),  "n_products": (8,  15), "n_length_groups": (3, 6), "cap_per_product": 1000},
    "L" : {"n_stocks": (60,  90), "n_products": (15, 30), "n_length_groups": (5, 15), "cap_per_product": 1000},
    "XL": {"n_stocks": (130, 160), "n_products": (20, 50), "n_length_groups": (8, 18), "cap_per_product": 1000},
}

SWEEP_SIZE        = ["XS", "S", "M", "L", "XL"]
SWEEP_SIZE_LABELS = ["XS", "S", "M", "L", "XL"]

# ---------------------------------------------------------------------------
# Base config — calibrated from OMP
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "length_mean"                   : 3000.0,
    "length_std"                    : 900.0,
    "length_min"                    : 800.0,
    "length_max"                    : 5500.0,

    "width_classes": [
        {"prob": 0.35, "range": (0.30, 0.70)},
        {"prob": 0.40, "range": (0.70, 1.15)},
        {"prob": 0.25, "range": (1.15, 1.80)},
    ],

    "stock_length_lognormal_mu"     : 8.584,
    "stock_length_lognormal_sigma"  : 0.516,
    "stock_length_min"              : 1652.0,
    "stock_length_max"              : 99999.0,  # relaxed — ratio control adjusts this

    "stock_width_mean"              : 3.245,
    "stock_width_std"               : 1.602,
    "stock_width_min"               : 2.0,
    "stock_width_max"               : 6.042,

    "activation_cost_range"         : (60.0, 400.0),

    "defect_count_distribution": [
        {"n": 1, "prob": 0.177},
        {"n": 2, "prob": 0.292},
        {"n": 3, "prob": 0.200},
        {"n": 4, "prob": 0.215},
        {"n": 5, "prob": 0.085},
        {"n": 6, "prob": 0.031},
    ],

    # defect lognormal params — calibrated from OMP instances (130 stocks, 368 defects)
    "defect_length_lognormal_mu"    : 6.256,
    "defect_length_lognormal_sigma" : 0.708,
    "defect_length_max"             : 2500.0,

    "defect_width_lognormal_mu"     : -0.563,
    "defect_width_lognormal_sigma"  : 0.750,
    "defect_width_max"              : 3.0,

    "setup_cost"                    : 43.29,
    "repetition_cost_alpha"         : 0.035,
    "repetition_cost_beta"          : 18.0,
    "repetition_cost_noise"         : 8.0,
}

# ---------------------------------------------------------------------------
# Sensitivity sweep configurations
# ---------------------------------------------------------------------------

# baseline intervals (matching OMP characteristics)
BASELINE = {
    "defect_length_ratio"        : (0.25, 0.35),
    "demand_tightness"           : (0.25, 0.35),
    "stock_product_length_ratio" : (1.75, 2.5),    # OMP mean=2.05
    "stock_product_width_ratio"  : (2.5, 4.5),    # OMP mean=3.33
    "size_class"                 : "medium",
}

# baseline size config — broader M range, used for all non-size sweeps
BASELINE_SIZE_CONFIG = FIXED_SIZE_CONFIGS["M"]

# sweep 1 — defect length ratio (3 broad levels)
SWEEP_DEFECT = [
    (0.01, 0.15),   # Low
    (0.15, 0.35),   # Medium — baseline OMP range
    (0.35, 0.60),   # High
]
SWEEP_DEFECT_LABELS = ["Low", "Medium", "High"]

# sweep 2 — demand tightness (3 broad levels)
SWEEP_DEMAND = [
    (0.08, 0.20),   # Loose
    (0.25, 0.35),   # Medium — baseline OMP range
    (0.35, 0.50),   # Tight
]
SWEEP_DEMAND_LABELS = ["Loose", "Medium", "Tight"]

# sweep 3 — stock/product length ratio (3 broad levels)
SWEEP_LENGTH_RATIO = [
    (1.2, 2.0),     # Short
    (2.0, 4.0),     # Medium — baseline OMP range
    (4.0, 8.0),     # Long
]
SWEEP_LENGTH_RATIO_LABELS = ["Short", "Medium", "Long"]

# sweep 4 — stock/product width ratio (3 broad levels)
SWEEP_WIDTH_RATIO = [
    (1.5, 2.5),     # Low
    (2.5, 4.5),     # Medium — baseline OMP range
    (4.5, 8.0),     # High
]
SWEEP_WIDTH_RATIO_LABELS = ["Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPS = 1e-9

def round3(x): return round(x, 3)
def round4(x): return round(x, 4)

def weighted_choice(rng, items):
    r = rng.random()
    cum = 0.0
    for item in items:
        cum += item["prob"]
        if r <= cum:
            return item
    return items[-1]

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        x, y = merged[-1]
        if a <= y + EPS:
            merged[-1] = (x, max(y, b))
        else:
            merged.append((a, b))
    return merged

def subtract_intervals(base, blocked):
    a, b = base
    if b < a + EPS:
        return []
    blocked = merge_intervals([(max(a, x), min(b, y)) for x, y in blocked if y > a and x < b])
    if not blocked:
        return [(a, b)]
    result = []
    cursor = a
    for x, y in blocked:
        if cursor < x - EPS:
            result.append((cursor, x))
        cursor = max(cursor, y)
    if cursor < b - EPS:
        result.append((cursor, b))
    return result

def overlap_1d(a1, a2, b1, b2):
    return max(a1, b1) < min(a2, b2) - EPS

def canonical_sig(counts):
    return tuple(sorted((pid, q) for pid, q in counts.items() if q > 0))


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def sample_length(rng, config):
    l = rng.gauss(config["length_mean"], config["length_std"])
    return max(config["length_min"], min(config["length_max"], l))

def sample_product_width(rng, config):
    cls = weighted_choice(rng, config["width_classes"])
    a, b = cls["range"]
    return round3(rng.uniform(a, b))

def generate_products(rng, config, n_products, n_length_groups):
    lengths = []
    attempts = 0
    while len(lengths) < n_length_groups and attempts < 200:
        l = round4(sample_length(rng, config))
        if l not in lengths:
            lengths.append(l)
        attempts += 1

    products = []
    for i in range(1, n_products + 1):
        if i <= len(lengths):
            length = lengths[i - 1]
        else:
            length = rng.choice(lengths)

        products.append(Product(
            product_id=f"P{i}",
            width=sample_product_width(rng, config),
            length=length,
            order_quantity=0,
        ))
    return products


# ---------------------------------------------------------------------------
# Demand assignment — controlled by demand_tightness interval
# ---------------------------------------------------------------------------

def assign_demand(rng, products, stocks, demand_tightness_interval, config):
    """
    Assign demand to hit a target surface utilization within the given interval.
    demand_tightness = total demanded area / total available area (after defects).
    """
    total_available_area = sum(
        s.length * s.width - sum(d.length * d.width for d in s.defects)
        for s in stocks
    )
    total_product_surface_per_unit = sum(p.length * p.width for p in products)

    target_utilization = rng.uniform(*demand_tightness_interval)
    total_demand_surface = target_utilization * total_available_area

    demand_min = 1
    demand_max = 15

    for p in products:
        product_share = (p.length * p.width) / total_product_surface_per_unit
        raw_demand = (total_demand_surface * product_share) / (p.length * p.width)
        raw_demand *= rng.uniform(0.7, 1.3)
        p.order_quantity = max(demand_min, min(demand_max, round(raw_demand)))

    actual_demand_surface = sum(p.length * p.width * p.order_quantity for p in products)
    actual_tightness = actual_demand_surface / total_available_area if total_available_area > 0 else 0
    return actual_tightness


# ---------------------------------------------------------------------------
# Stocks — controlled by stock_product_length_ratio interval
# ---------------------------------------------------------------------------

def generate_stocks(rng, config, n_stocks, avg_product_length,
                    avg_product_width,
                    stock_product_length_ratio_interval,
                    stock_product_width_ratio_interval,
                    min_pattern_length=None):

    target_length_ratio = rng.uniform(*stock_product_length_ratio_interval)
    target_avg_stock_length = target_length_ratio * avg_product_length

    target_width_ratio = rng.uniform(*stock_product_width_ratio_interval)
    target_avg_stock_width = target_width_ratio * avg_product_width
    # clamp to config bounds
    target_avg_stock_width = max(
        config["stock_width_min"],
        min(config["stock_width_max"], target_avg_stock_width)
    )

    stock_length_min  = max(config["stock_length_min"],
                            min_pattern_length * 1.5 if min_pattern_length
                            else config["stock_length_min"])
    stock_length_low  = target_avg_stock_length * 0.85
    stock_length_high = target_avg_stock_length * 1.3
    stock_length_max  = stock_length_high

    stocks = []
    amin, amax = config["activation_cost_range"]

    for i in range(1, n_stocks + 1):
        sl = round4(rng.uniform(stock_length_low, stock_length_high))

        sw = rng.gauss(target_avg_stock_width, config["stock_width_std"] * 0.3)
        sw = round3(max(config["stock_width_min"],
                        min(config["stock_width_max"], sw)))

        size_ratio = (
            0.45 * (sw - config["stock_width_min"]) /
            (config["stock_width_max"] - config["stock_width_min"]) +
            0.55 * (sl - stock_length_min) /
            (stock_length_max - stock_length_min + EPS)
        )
        cost = round4(max(0.0, amin + (amax - amin) * size_ratio
                          + rng.uniform(-10, 10)))

        stock = Stock(stock_id=f"S{i}", width=sw, length=sl,
                      activation_cost=cost, defects=[])
        stocks.append(stock)

    return stocks


# ---------------------------------------------------------------------------
# Defects — controlled by defect_length_ratio interval
# ---------------------------------------------------------------------------

def defects_overlap(d1, d2):
    return (d1.x < d2.x + d2.length and d1.x + d1.length > d2.x and
            d1.y < d2.y + d2.width  and d1.y + d1.width  > d2.y)

def _feasible_domain_size(stock, defects, min_pattern_length):
    admissible = (0.0, stock.length - min_pattern_length)
    if admissible[1] <= admissible[0]:
        return 0.0
    blocked = []
    for d in defects:
        block_a = max(0.0, d.x - min_pattern_length)
        block_b = min(admissible[1], d.x + d.length)
        if block_b > block_a + EPS:
            blocked.append((block_a, block_b))
    feasible = subtract_intervals(admissible, blocked)
    return sum(b - a for a, b in feasible)

def generate_defects_for_stock(rng, stock, config, target_defect_length,
                                min_pattern_length=None):
    """
    Generate defects for one stock targeting a specific total defect length.
    Defect count from weighted distribution, individual lengths scaled to
    collectively hit target_defect_length (= target_ratio * stock.length).
    """
    n_def = weighted_choice(rng, config["defect_count_distribution"])["n"]
    if n_def == 0 or target_defect_length <= 0:
        return []

    target_per_defect = target_defect_length / n_def

    defects = []
    max_attempts = 30
    check_domain = min_pattern_length is not None and stock.length > min_pattern_length

    for j in range(1, n_def + 1):
        for _ in range(max_attempts):
            base_len = math.exp(rng.gauss(
                config["defect_length_lognormal_mu"],
                config["defect_length_lognormal_sigma"]
            ))
            dlen = 0.5 * base_len + 0.5 * target_per_defect * rng.uniform(0.7, 1.3)
            dlen = max(3.0, min(dlen, config["defect_length_max"]))
            dlen = min(dlen, stock.length * 0.5)

            dwid = math.exp(rng.gauss(
                config["defect_width_lognormal_mu"],
                config["defect_width_lognormal_sigma"]
            ))
            dwid = max(0.002, min(dwid, config["defect_width_max"]))
            dwid = min(dwid, stock.width * 0.9)

            x = rng.uniform(0.0, max(0.0, stock.length - dlen))
            y = rng.uniform(0.0, max(0.0, stock.width - dwid))

            candidate = Defect(
                defect_id=f"{stock.stock_id}_D{j}",
                x=round4(x), y=round4(y),
                length=round4(dlen), width=round4(dwid),
            )

            if any(defects_overlap(candidate, existing) for existing in defects):
                continue

            if check_domain:
                trial = defects + [candidate]
                admissible_size = stock.length - min_pattern_length
                remaining = _feasible_domain_size(stock, trial, min_pattern_length)
                if remaining < admissible_size * 0.10:
                    continue

            defects.append(candidate)
            break

    return defects

def generate_defects_all_stocks(rng, stocks, config, defect_length_ratio_interval,
                                 min_pattern_length=None):
    """
    Generate defects for all stocks targeting the defect_length_ratio interval.
    """
    for stock in stocks:
        target_ratio = rng.uniform(*defect_length_ratio_interval)
        target_defect_length = target_ratio * stock.length
        stock.defects = generate_defects_for_stock(
            rng, stock, config, target_defect_length,
            min_pattern_length=min_pattern_length
        )


# ---------------------------------------------------------------------------
# Pattern enumeration
# ---------------------------------------------------------------------------

def enumerate_patterns(rng, products_in_group, max_width, cap_per_product=50):
    results = []
    seen = set()

    for p in products_in_group:
        max_qty = int(max_width // p.width)
        for qty in range(1, max_qty + 1):
            sig = ((p.product_id, qty),)
            if sig not in seen:
                seen.add(sig)
                results.append({p.product_id: qty})

    shuffled = list(products_in_group)
    rng.shuffle(shuffled)
    max_patterns = len(products_in_group) * cap_per_product

    def recurse(idx, current_counts, current_width):
        if len(results) >= max_patterns:
            return
        if current_width > EPS and len(current_counts) > 1:
            sig = canonical_sig(current_counts)
            if sig not in seen:
                seen.add(sig)
                results.append(dict(current_counts))
        if idx == len(shuffled):
            return
        p = shuffled[idx]
        max_qty = int((max_width - current_width) // p.width)
        for qty in range(1, max_qty + 1):
            if len(results) >= max_patterns:
                return
            current_counts[p.product_id] = qty
            recurse(idx + 1, current_counts, current_width + qty * p.width)
        current_counts.pop(p.product_id, None)
        recurse(idx + 1, current_counts, current_width)

    recurse(0, {}, 0.0)
    return results

def build_pattern(pattern_id, counts, products_by_id):
    length = products_by_id[next(iter(counts))].length
    x_cursor = 0.0
    strips = []
    for pid, qty in sorted(counts.items()):
        p = products_by_id[pid]
        strip_width = p.width * qty
        strips.append(PatternStrip(
            product_id=pid, quantity=qty,
            x_left=round4(x_cursor),
            x_right=round4(x_cursor + strip_width),
        ))
        x_cursor += strip_width
    return Pattern(pattern_id=pattern_id, length=length, strips=strips)

def generate_patterns(rng, products, stocks, config, cap_per_product=50):
    products_by_id = {p.product_id: p for p in products}
    max_stock_width = max(s.width for s in stocks)

    by_length = defaultdict(list)
    for p in products:
        by_length[p.length].append(p)

    patterns = []
    pattern_counter = 1

    for length, group in by_length.items():
        combos = enumerate_patterns(rng, group, max_stock_width,
                                    cap_per_product=cap_per_product)
        for counts in combos:
            patterns.append(build_pattern(f"C{pattern_counter}", counts, products_by_id))
            pattern_counter += 1

    return patterns


# ---------------------------------------------------------------------------
# Feasibility windows
# ---------------------------------------------------------------------------

def _merge_lateral_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        x, y = merged[-1]
        if a <= y + EPS:
            merged[-1] = (x, max(y, b))
        else:
            merged.append((a, b))
    return merged


def _build_strip_widths(pattern):
    widths = []
    for s in pattern.strips:
        unit_width = (s.x_right - s.x_left) / s.quantity
        for _ in range(s.quantity):
            widths.append(unit_width)
    return widths


def _strip_widths_fit_in_free_zones(strip_widths, stock_width, defects_in_window):
    if not strip_widths:
        return True

    defect_intervals = _merge_lateral_intervals(
        [(d.y, d.y + d.width) for d in defects_in_window]
    )

    free_widths = []
    cursor = 0.0
    for d_start, d_end in defect_intervals:
        if d_start > cursor + EPS:
            free_widths.append(d_start - cursor)
        cursor = max(cursor, d_end)

    if stock_width > cursor + EPS:
        free_widths.append(stock_width - cursor)

    if not free_widths:
        return False

    free_widths.sort(reverse=True)
    strip_widths = sorted(strip_widths, reverse=True)

    for sw in strip_widths:
        placed = False
        for i in range(len(free_widths)):
            if free_widths[i] + EPS >= sw:
                free_widths[i] -= sw
                placed = True
                break
        if not placed:
            return False
        free_widths.sort(reverse=True)

    return True


def generate_feasible_candidate_windows(pattern, stock):
    """
    Generate all feasible candidate windows [a, b] induced by split points
    (0, stock.length, defect starts, defect ends), for all i < j.
    """
    pattern_used_width = max((s.x_right for s in pattern.strips), default=0.0)

    if pattern_used_width > stock.width + EPS:
        return []
    if pattern.length > stock.length + EPS:
        return []

    strip_widths = _build_strip_widths(pattern)

    split_points = sorted(set(
        [0.0, stock.length] +
        [d.x for d in stock.defects if EPS < d.x < stock.length - EPS] +
        [d.x + d.length for d in stock.defects if EPS < d.x + d.length < stock.length - EPS]
    ))

    feasible_windows = []

    for i in range(len(split_points) - 1):
        for j in range(i + 1, len(split_points)):
            a = split_points[i]
            b = split_points[j]

            if b - a < pattern.length - EPS:
                continue

            defects_in_interval = [
                d for d in stock.defects
                if d.x < b - EPS and d.x + d.length > a + EPS
            ]

            if not defects_in_interval:
                feasible_windows.append((round4(a), round4(b)))
                continue

            if _strip_widths_fit_in_free_zones(strip_widths, stock.width, defects_in_interval):
                feasible_windows.append((round4(a), round4(b)))

    return feasible_windows

def keep_maximal_windows(windows):
    """
    Keep only windows that are maximal under inclusion.
    """
    if not windows:
        return []

    maximal = []

    for i, (a, b) in enumerate(windows):
        dominated = False
        for j, (c, d) in enumerate(windows):
            if i == j:
                continue

            contains = (c <= a + EPS) and (b <= d + EPS)
            strictly_larger = (c < a - EPS) or (d > b + EPS)

            if contains and strictly_larger:
                dominated = True
                break

        if not dominated:
            maximal.append((round4(a), round4(b)))

    return sorted(set(maximal))

def merge_maximal_windows_if_feasible(windows, pattern, stock):
    """
    Merge overlapping maximal windows only if their union is still feasible.
    """
    if not windows:
        return []

    strip_widths = _build_strip_widths(pattern)

    windows = sorted(windows)
    changed = True
    current = windows[:]

    while changed:
        changed = False
        merged_result = []
        used = [False] * len(current)

        for i in range(len(current)):
            if used[i]:
                continue

            best_a, best_b = current[i]
            used[i] = True

            progress = True
            while progress:
                progress = False
                for j in range(len(current)):
                    if used[j]:
                        continue

                    c, d = current[j]

                    if c <= best_b + EPS and best_a <= d + EPS:
                        candidate = (min(best_a, c), max(best_b, d))

                        defects_in_candidate = [
                            defect for defect in stock.defects
                            if defect.x < candidate[1] - EPS
                            and defect.x + defect.length > candidate[0] + EPS
                        ]

                        feasible = (
                            not defects_in_candidate or
                            _strip_widths_fit_in_free_zones(
                                strip_widths, stock.width, defects_in_candidate
                            )
                        )

                        if feasible:
                            best_a, best_b = candidate
                            used[j] = True
                            changed = True
                            progress = True

            merged_result.append((round4(best_a), round4(best_b)))

        current = sorted(set(merged_result))

    return current

def compute_feasible_start_windows(pattern, stock):
    candidates = generate_feasible_candidate_windows(pattern, stock)
    maximal = keep_maximal_windows(candidates)
    final_windows = merge_maximal_windows_if_feasible(maximal, pattern, stock)

    return [
        (round4(a), round4(b))
        for a, b in final_windows
        if b - a >= pattern.length - EPS
    ]

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def compute_repetition_cost(rng, pattern, stock, config):
    used_ratio = pattern.used_width / stock.width if stock.width > EPS else 1.0
    val = (
        config["repetition_cost_alpha"] * pattern.length +
        config["repetition_cost_beta"] * used_ratio +
        0.015 * stock.activation_cost +
        rng.uniform(-config["repetition_cost_noise"], config["repetition_cost_noise"])
    )
    return round4(max(1.0, val))


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------

def generate_proposals(rng, patterns, stocks, config):
    proposals = []
    proposal_counter = 1

    for pattern in patterns:
        for stock in stocks:
            windows = compute_feasible_start_windows(pattern, stock)
            if not windows:
                continue

            rep_cost = compute_repetition_cost(rng, pattern, stock, config)

            for earliest_start, latest_end in windows:
                proposals.append({
                    "proposal_id": f"C{proposal_counter}",
                    "costs": [{
                        "setup_cost": config["setup_cost"],
                        "cost_per_repetition": rep_cost,
                    }],
                    "products_produced": [
                        {
                            "product_id": pid,
                            "quantity_produced_per_repetition": qty,
                        }
                        for pid, qty in sorted(pattern.produced_quantities.items())
                    ],
                    "stock_consumed": [{
                        "stock_id": stock.stock_id,
                        "earliest_start": earliest_start,
                        "latest_end": latest_end,
                        "length_consumed_per_repetition": round4(pattern.length),
                    }],
                })
                proposal_counter += 1

    return proposals


# ---------------------------------------------------------------------------
# Feasibility checker
# ---------------------------------------------------------------------------

def check_feasibility(products, proposals):
    covered = set()
    for prop in proposals:
        for p in prop["products_produced"]:
            covered.add(p["product_id"])
    for product in products:
        if product.product_id not in covered:
            return False, f"Product {product.product_id} has no proposal"
    return True, "OK"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_omp_format(instance_id, description, products, proposals, stocks):
    return {
        "problem": {
            "description": description,
            "instance_id": instance_id,
            "products": [
                {
                    "product_id": p.product_id,
                    "length": p.length,
                    "width": p.width,
                    "order_quantity": p.order_quantity,
                }
                for p in products
            ],
            "proposals": proposals,
            "stocks": [
                {
                    "stock_id": s.stock_id,
                    "width": s.width,
                    "length": s.length,
                    "cost": s.activation_cost,
                    "defects": [
                        {
                            "start_in_length": d.x,
                            "start_in_width": d.y,
                            "length": d.length,
                            "width": d.width,
                        }
                        for d in s.defects
                    ],
                }
                for s in stocks
            ],
        }
    }


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------

def generate_instance(
    size_class="medium",
    defect_length_ratio=(0.25, 0.35),
    demand_tightness=(0.25, 0.35),
    stock_product_length_ratio=(1.75, 2.25),
    stock_product_width_ratio=(2.5, 4.5),
    size_config=None,
    seed=0,
    instance_id=None,
    config_overrides=None,
    max_retries=5,
    verbose=False,
):
    """
    Generate one synthetic instance with controlled characteristics.

    Args:
        size_class                 : "small", "medium", or "large"
        defect_length_ratio        : (low, high) interval for avg defect length / stock length
        demand_tightness           : (low, high) interval for demanded area / available area
        stock_product_length_ratio : (low, high) interval for avg stock length / avg product length
        stock_product_width_ratio  : (low, high) interval for avg stock width / avg product width
        size_config                : optional fixed size config (overrides size_class)
        seed                       : random seed
        instance_id                : optional string ID
        config_overrides           : dict of BASE_CONFIG overrides
        max_retries                : retries if feasibility check fails
        verbose                    : print generation details
    """
    config = dict(BASE_CONFIG)
    if config_overrides:
        config.update(config_overrides)

    size = size_config if size_config is not None else FIXED_SIZE_CONFIGS["M"]

    if instance_id is None:
        dr_lo, dr_hi = defect_length_ratio
        dt_lo, dt_hi = demand_tightness
        rr_lo, rr_hi = stock_product_length_ratio
        instance_id = (
            f"SYN_{size_class.upper()}_"
            f"DR{dr_lo:.2f}-{dr_hi:.2f}_"
            f"DT{dt_lo:.2f}-{dt_hi:.2f}_"
            f"SR{rr_lo:.1f}-{rr_hi:.1f}_"
            f"S{seed:04d}"
        )

    for attempt in range(max_retries):
        rng = random.Random(seed + attempt * 1000)

        n_products      = rng.randint(*size["n_products"])
        n_stocks        = rng.randint(*size["n_stocks"])
        n_length_groups = rng.randint(*size["n_length_groups"])
        cap_per_product = size["cap_per_product"]

        products = generate_products(rng, config, n_products, n_length_groups)
        avg_product_length = sum(p.length for p in products) / len(products)
        min_product_length = min(p.length for p in products)

        stocks = generate_stocks(
            rng, config, n_stocks,
            avg_product_length=avg_product_length,
            avg_product_width=sum(p.width for p in products) / len(products),
            stock_product_length_ratio_interval=stock_product_length_ratio,
            stock_product_width_ratio_interval=stock_product_width_ratio,
            min_pattern_length=min_product_length,
        )

        generate_defects_all_stocks(
            rng, stocks, config,
            defect_length_ratio_interval=defect_length_ratio,
            min_pattern_length=min_product_length,
        )

        actual_tightness = assign_demand(
            rng, products, stocks,
            demand_tightness_interval=demand_tightness,
            config=config,
        )

        patterns  = generate_patterns(rng, products, stocks, config,
                                      cap_per_product=cap_per_product)
        proposals = generate_proposals(rng, patterns, stocks, config)

        feasible, reason = check_feasibility(products, proposals)

        if verbose:
            avg_stock_len = sum(s.length for s in stocks) / len(stocks)
            avg_defect_ratio = sum(
                sum(d.length for d in s.defects) / s.length
                for s in stocks
            ) / len(stocks)
            print(
                f"  [{instance_id}] attempt={attempt+1}  "
                f"P={n_products}  S={n_stocks}  props={len(proposals)}  "
                f"defect_ratio={avg_defect_ratio:.3f}  "
                f"tightness={actual_tightness:.3f}  "
                f"len_ratio={avg_stock_len/avg_product_length:.2f}  "
                f"feasible={feasible}"
            )

        if feasible:
            return export_omp_format(
                instance_id=instance_id,
                description=(
                    f"Synthetic {size_class} instance — seed={seed}  "
                    f"defect_ratio={sum(sum(d.length for d in s.defects)/s.length for s in stocks)/len(stocks):.3f}  "
                    f"tightness={actual_tightness:.3f}  "
                    f"len_ratio={sum(s.length for s in stocks)/len(stocks)/avg_product_length:.2f}"
                ),
                products=products,
                proposals=proposals,
                stocks=stocks,
            )
        else:
            if verbose:
                print(f"  Attempt {attempt+1}/{max_retries} failed: {reason} — retrying")

    print(f"  WARNING: could not generate feasible instance after {max_retries} attempts — {instance_id}")
    return export_omp_format(
        instance_id=instance_id,
        description=f"INFEASIBLE — {instance_id}",
        products=products,
        proposals=proposals,
        stocks=stocks,
    )


# ---------------------------------------------------------------------------
# Sensitivity sweep generation
# ---------------------------------------------------------------------------

def generate_baseline(output_dir="../data/sensitivity/baseline/", n_instances=30,
                      base_seed=0, verbose=False):
    """
    Generate baseline instances — all characteristics in medium range.
    These are the reference instances for all comparisons.
    Uses FIXED_SIZE_CONFIGS["M"] for size (broader range than other levels).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Baseline generation ({n_instances} instances)")
    print(f"{'='*60}")

    for i in range(n_instances):
        seed = base_seed + i
        instance_id = f"SYN_BASELINE_{i:03d}"
        instance = generate_instance(
            size_class="medium",
            defect_length_ratio=BASELINE["defect_length_ratio"],
            demand_tightness=BASELINE["demand_tightness"],
            stock_product_length_ratio=BASELINE["stock_product_length_ratio"],
            stock_product_width_ratio=BASELINE["stock_product_width_ratio"],
            size_config=BASELINE_SIZE_CONFIG,
            seed=seed,
            instance_id=instance_id,
            verbose=verbose,
        )
        path = os.path.join(output_dir, f"{instance_id}.json")
        save_json(instance, path)

    print(f"  {n_instances} baseline instances saved to {output_dir}")


def generate_sensitivity_sweep(output_dir="../data/sensitivity/", n_per_point=30,
                                n_baseline=90, n_per_size=60,
                                base_seed=0, verbose=False):
    """
    Generate all instances for the sensitivity analysis.

    Structure:
        baseline/            — 90 baseline instances (all medium) — 30 for tuning, 60 for testing
        sweep1_defect/       — Low / High defect length ratio      — 30 per level
        sweep2_demand/       — Loose / Tight demand tightness      — 30 per level
        sweep3_length_ratio/ — Short / Long stock/product length   — 30 per level
        sweep4_width_ratio/  — Low / High stock/product width      — 30 per level
        sweep5_size/         — XS / S / L / XL                    — 60 per level
    Total: 90 + 4x2x30 + 4x60 = 570 instances
    Tuning set (33%): 30 baseline + 10 per characteristic level + 20 per size level
    """
    os.makedirs(output_dir, exist_ok=True)
    total = 0

    # Baseline
    baseline_dir = os.path.join(output_dir, "baseline")
    generate_baseline(output_dir=baseline_dir, n_instances=n_baseline,
                      base_seed=base_seed, verbose=verbose)
    total += n_baseline

    # Sweep 1 — defect length ratio (Low / High only — M is baseline)
    sweep1_dir = os.path.join(output_dir, "sweep1_defect")
    os.makedirs(sweep1_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Sweep 1 — Defect length ratio (Low / High x {n_per_point} instances)")
    print(f"{'='*60}")

    for level_idx, (defect_interval, label) in enumerate(
            zip([SWEEP_DEFECT[0], SWEEP_DEFECT[2]],
                [SWEEP_DEFECT_LABELS[0], SWEEP_DEFECT_LABELS[2]])):
        print(f"\n  [{label}]  interval={defect_interval}")
        for i in range(n_per_point):
            seed = base_seed + level_idx * 100 + i
            instance_id = f"SYN_SWEEP1_DEFECT_{label.upper()}_{i:03d}"
            instance = generate_instance(
                size_class="medium",
                defect_length_ratio=defect_interval,
                demand_tightness=BASELINE["demand_tightness"],
                stock_product_length_ratio=BASELINE["stock_product_length_ratio"],
                stock_product_width_ratio=BASELINE["stock_product_width_ratio"],
                size_config=BASELINE_SIZE_CONFIG,
                seed=seed,
                instance_id=instance_id,
                verbose=verbose,
            )
            path = os.path.join(sweep1_dir, f"{instance_id}.json")
            save_json(instance, path)
            total += 1

    # Sweep 2 — demand tightness (Loose / Tight only)
    sweep2_dir = os.path.join(output_dir, "sweep2_demand")
    os.makedirs(sweep2_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Sweep 2 — Demand tightness (Loose / Tight x {n_per_point} instances)")
    print(f"{'='*60}")

    for level_idx, (demand_interval, label) in enumerate(
            zip([SWEEP_DEMAND[0], SWEEP_DEMAND[2]],
                [SWEEP_DEMAND_LABELS[0], SWEEP_DEMAND_LABELS[2]])):
        print(f"\n  [{label}]  interval={demand_interval}")
        for i in range(n_per_point):
            seed = base_seed + 1000 + level_idx * 100 + i
            instance_id = f"SYN_SWEEP2_DEMAND_{label.upper()}_{i:03d}"
            instance = generate_instance(
                size_class="medium",
                defect_length_ratio=BASELINE["defect_length_ratio"],
                demand_tightness=demand_interval,
                stock_product_length_ratio=BASELINE["stock_product_length_ratio"],
                stock_product_width_ratio=BASELINE["stock_product_width_ratio"],
                size_config=BASELINE_SIZE_CONFIG,
                seed=seed,
                instance_id=instance_id,
                verbose=verbose,
            )
            path = os.path.join(sweep2_dir, f"{instance_id}.json")
            save_json(instance, path)
            total += 1

    # Sweep 3 — stock/product length ratio (Short / Long only)
    sweep3_dir = os.path.join(output_dir, "sweep3_length_ratio")
    os.makedirs(sweep3_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Sweep 3 — Length ratio (Short / Long x {n_per_point} instances)")
    print(f"{'='*60}")

    for level_idx, (ratio_interval, label) in enumerate(
            zip([SWEEP_LENGTH_RATIO[0], SWEEP_LENGTH_RATIO[2]],
                [SWEEP_LENGTH_RATIO_LABELS[0], SWEEP_LENGTH_RATIO_LABELS[2]])):
        print(f"\n  [{label}]  interval={ratio_interval}")
        for i in range(n_per_point):
            seed = base_seed + 2000 + level_idx * 100 + i
            instance_id = f"SYN_SWEEP3_LENRATIO_{label.upper()}_{i:03d}"
            instance = generate_instance(
                size_class="medium",
                defect_length_ratio=BASELINE["defect_length_ratio"],
                demand_tightness=BASELINE["demand_tightness"],
                stock_product_length_ratio=ratio_interval,
                stock_product_width_ratio=BASELINE["stock_product_width_ratio"],
                size_config=BASELINE_SIZE_CONFIG,
                seed=seed,
                instance_id=instance_id,
                verbose=verbose,
            )
            path = os.path.join(sweep3_dir, f"{instance_id}.json")
            save_json(instance, path)
            total += 1

    # Sweep 4 — stock/product width ratio (Low / High only)
    sweep4_dir = os.path.join(output_dir, "sweep4_width_ratio")
    os.makedirs(sweep4_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Sweep 4 — Width ratio (Low / High x {n_per_point} instances)")
    print(f"{'='*60}")

    for level_idx, (ratio_interval, label) in enumerate(
            zip([SWEEP_WIDTH_RATIO[0], SWEEP_WIDTH_RATIO[2]],
                [SWEEP_WIDTH_RATIO_LABELS[0], SWEEP_WIDTH_RATIO_LABELS[2]])):
        print(f"\n  [{label}]  interval={ratio_interval}")
        for i in range(n_per_point):
            seed = base_seed + 3000 + level_idx * 100 + i
            instance_id = f"SYN_SWEEP4_WIDRATIO_{label.upper()}_{i:03d}"
            instance = generate_instance(
                size_class="medium",
                defect_length_ratio=BASELINE["defect_length_ratio"],
                demand_tightness=BASELINE["demand_tightness"],
                stock_product_length_ratio=BASELINE["stock_product_length_ratio"],
                stock_product_width_ratio=ratio_interval,
                size_config=BASELINE_SIZE_CONFIG,
                seed=seed,
                instance_id=instance_id,
                verbose=verbose,
            )
            path = os.path.join(sweep4_dir, f"{instance_id}.json")
            save_json(instance, path)
            total += 1

    # Sweep 5 — instance size (XS / S / L / XL — M is baseline)
    sweep5_dir = os.path.join(output_dir, "sweep5_size")
    os.makedirs(sweep5_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Sweep 5 — Instance size (XS / S / L / XL x {n_per_point} instances)")
    print(f"{'='*60}")

    for level_idx, size_key in enumerate(["XS", "S", "L", "XL"]):
        size_cfg = FIXED_SIZE_CONFIGS[size_key]
        print(f"\n  [{size_key}]  n_stocks={size_cfg['n_stocks']}")
        for i in range(n_per_size):
            seed = base_seed + 4000 + level_idx * 100 + i
            instance_id = f"SYN_SWEEP5_SIZE_{size_key}_{i:03d}"
            instance = generate_instance(
                size_class="medium",
                defect_length_ratio=BASELINE["defect_length_ratio"],
                demand_tightness=BASELINE["demand_tightness"],
                stock_product_length_ratio=BASELINE["stock_product_length_ratio"],
                stock_product_width_ratio=BASELINE["stock_product_width_ratio"],
                size_config=size_cfg,
                seed=seed,
                instance_id=instance_id,
                verbose=verbose,
            )
            path = os.path.join(sweep5_dir, f"{instance_id}.json")
            save_json(instance, path)
            total += 1

    print(f"\n{'='*60}")
    print(f"  Complete — {total} instances saved to {output_dir}")
    print(f"  baseline: {n_baseline}  |  4 sweeps x 2 levels x {n_per_point} = {4*2*n_per_point}  |  size sweep x 4 levels x {n_per_size} = {4*n_per_size}")
    print(f"{'='*60}")


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing baseline instance generation...")
    inst = generate_instance(
        size_class="medium",
        defect_length_ratio=(0.25, 0.35),
        demand_tightness=(0.25, 0.35),
        stock_product_length_ratio=(1.75, 2.25),
        seed=42,
        verbose=True,
    )
    p = inst["problem"]
    stock_surface = sum(s["length"] * s["width"] for s in p["stocks"])
    demand_surface = sum(
        prod["length"] * prod["width"] * prod["order_quantity"]
        for prod in p["products"]
    )
    avg_defect_ratio = sum(
        sum(d["length"] for d in s["defects"]) / s["length"]
        for s in p["stocks"]
    ) / len(p["stocks"])
    avg_stock_len = sum(s["length"] for s in p["stocks"]) / len(p["stocks"])
    avg_prod_len  = sum(prod["length"] for prod in p["products"]) / len(p["products"])

    print(f"\n  Products  : {len(p['products'])}")
    print(f"  Stocks    : {len(p['stocks'])}")
    print(f"  Proposals : {len(p['proposals'])}")
    print(f"  Defect ratio (length) : {avg_defect_ratio:.3f}")
    print(f"  Demand tightness      : {demand_surface/stock_surface:.3f}")
    print(f"  Stock/product ratio   : {avg_stock_len/avg_prod_len:.2f}")
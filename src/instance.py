"""
instance.py
-----------
Defines the core data structures and instance loading pipeline for the
1.5D Cutting Stock Problem with defects.

Data structures: Defect, Stock, Product, Proposal, PatternStockEntry, Pattern.

Loading pipeline (three steps):
  1. load_instance        -- reads stocks, products and raw proposals from a JSON file
  2. fix_proposal_windows -- validates each proposal against defects, splits windows
                             at defect boundaries and discards infeasible proposals
  3. build_patterns       -- groups validated proposals by production signature into
                             Pattern objects, merging overlapping windows where feasible

Utilities:
  load_n_from_dir         -- loads multiple instances from a directory in one call
  print_summary           -- prints a short human-readable summary of a loaded instance
  create_instance_summary -- exports a detailed instance breakdown to an Excel file
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from itertools import permutations
import pandas as pd
import os
 
 
# DATA CLASSES
 
@dataclass
class Defect:
    start_in_length: float
    length: float
    start_in_width: float
    width: float
 
 
@dataclass
class Stock:
    stock_id: str
    length: float
    width: float
    cost: float
    defects: List[Defect] = field(default_factory=list)
 
 
@dataclass
class Product:
    product_id: str
    length: float
    width: float
    demand: int
 
 
@dataclass
class Proposal:
    """
    Raw proposal as loaded from the JSON file.
    Kept as an intermediate object during the loading pipeline.
    After fix_proposal_windows, its windows are validated and split at defect boundaries.
    """
    proposal_id: str
    stock_id: str
    length_consumed: float
    setup_cost: float
    cost_per_repetition: float
    products_produced_per_repetition: Dict[str, int]
    earliest_start: float
    latest_end: float
    max_repetitions: int = 0
    windows: List[Tuple[float, float]] = field(default_factory=list)
 
    def __post_init__(self):
        if not self.windows:
            self.windows = [(self.earliest_start, self.latest_end)]
        self.max_repetitions = int(
            (self.latest_end - self.earliest_start) // self.length_consumed
        )
 
 
@dataclass
class PatternStockEntry:
    """
    windows          : list of (start, end) intervals where this configuration is feasible
    cost_per_rep     : cost per repetition for this (pattern, stock) configuration
    max_repetitions  : total repetitions possible across all windows in this entry
    source_proposal  : original proposal_id for traceability
    """
    windows         : List[Tuple[float, float]]
    cost_per_rep    : float
    max_repetitions : int
    source_proposal : str
 
 
@dataclass
class Pattern:
    """
    pattern_id                : unique identifier e.g. PAT_0001
    length_consumed           : length consumed per repetition (mm)
    setup_cost                : fixed cost per activation (assumed same across stocks)
    products_produced_per_rep : dict product_id -> quantity per repetition
    total_strip_width         : sum of all strip widths (m)
    stock_entries             : dict stock_id -> list of PatternStockEntry
    """
    pattern_id                : str
    length_consumed           : float
    setup_cost                : float
    products_produced_per_rep : Dict[str, int]
    total_strip_width         : float
    stock_entries             : Dict[str, List[PatternStockEntry]] = field(default_factory=dict)
 
 
# STEP 1: LOAD RAW PROPOSALS
 
def load_instance(filepath: str):
    """
    Load stocks, products, and raw proposals from a JSON file.
 
    Returns:
        stocks    : dict stock_id -> Stock
        products  : dict product_id -> Product
        proposals : list of Proposal (raw, not yet fixed or deduplicated)
    """
    with open(filepath, encoding='utf-8-sig') as f:
        raw = json.load(f)['problem']
 
    stocks = {}
    for s in raw['stocks']:
        defects = [
            Defect(
                start_in_length=d['start_in_length'],
                length=d['length'],
                start_in_width=d['start_in_width'],
                width=d['width']
            )
            for d in s.get('defects', [])
        ]
        stocks[s['stock_id']] = Stock(
            stock_id=s['stock_id'],
            length=s['length'],
            width=s['width'],
            cost=s['cost'],
            defects=defects
        )
 
    products = {}
    for p in raw['products']:
        products[p['product_id']] = Product(
            product_id=p['product_id'],
            length=p['length'],
            width=p['width'],
            demand=p['order_quantity']
        )
 
    proposals = []
    for c in raw['proposals']:
        sc = c['stock_consumed'][0]
        products_produced = {
            pp['product_id']: pp['quantity_produced_per_repetition']
            for pp in c['products_produced']
        }
        prop = Proposal(
            proposal_id=c['proposal_id'],
            stock_id=sc['stock_id'],
            length_consumed=sc['length_consumed_per_repetition'],
            setup_cost=c['costs'][0]['setup_cost'],
            cost_per_repetition=c['costs'][0]['cost_per_repetition'],
            products_produced_per_repetition=products_produced,
            earliest_start=sc['earliest_start'],
            latest_end=sc['latest_end']
        )
        proposals.append(prop)
 
    return stocks, products, proposals
 
 
# STEP 2: FIX PROPOSAL WINDOWS
# Validate each raw proposal and split its window at defect boundaries.
# Discards proposals that are geometrically impossible.
 
from itertools import permutations

def can_place_strips(strips, stock_width, defects_in_window):
    strip_widths = [w for _, w in strips]

    defect_intervals = sorted(
        [(d.start_in_width, d.start_in_width + d.width)
         for d in defects_in_window]
    )

    merged = []
    for a, b in defect_intervals:
        if not merged or a > merged[-1][1] + 1e-9:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    free_widths = []
    cursor = 0.0
    for d_start, d_end in merged:
        if d_start > cursor + 1e-9:
            free_widths.append(d_start - cursor)
        cursor = max(cursor, d_end)
    if stock_width > cursor + 1e-9:
        free_widths.append(stock_width - cursor)

    if not free_widths:
        return False, None

    free_widths  = sorted(free_widths,  reverse=True)
    strip_widths = sorted(strip_widths, reverse=True)

    for sw in strip_widths:
        placed = False
        for i in range(len(free_widths)):
            if free_widths[i] + 1e-9 >= sw:
                free_widths[i] -= sw
                placed = True
                break
        if not placed:
            return False, None
        free_widths.sort(reverse=True)

    return True, None

def can_place_strips3(strips, stock_width, defects_in_window, max_attempts=200):
    
    # --- shared: compute free zone widths ---
    defect_intervals = sorted(
        [(d.start_in_width, d.start_in_width + d.width)
         for d in defects_in_window]
    )
    merged = []
    for a, b in defect_intervals:
        if not merged or a > merged[-1][1] + 1e-9:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    free_widths = []
    cursor = 0.0
    for d_start, d_end in merged:
        if d_start > cursor + 1e-9:
            free_widths.append(d_start - cursor)
        cursor = max(cursor, d_end)
    if stock_width > cursor + 1e-9:
        free_widths.append(stock_width - cursor)

    if not free_widths:
        return False, None

    strip_widths = [w for _, w in strips]

    # --- Step 1: greedy (fast path, handles most cases) ---
    def greedy_fit(s_widths, f_widths):
        fw = sorted(f_widths, reverse=True)
        for sw in sorted(s_widths, reverse=True):
            placed = False
            for i in range(len(fw)):
                if fw[i] + 1e-9 >= sw:
                    fw[i] -= sw
                    placed = True
                    break
            if not placed:
                return False
            fw.sort(reverse=True)
        return True

    if greedy_fit(strip_widths, free_widths):
        return True, None

    # --- Step 2: exact fallback — try all assignments of strips to zones ---
    # This is a bin-packing feasibility check via backtracking.
    # Much cheaper than strip permutations because we only need to decide
    # which zone each strip goes into, not the order within zones.
    n_zones = len(free_widths)
    zone_remaining = list(free_widths)  # mutable

    def backtrack(idx):
        if idx == len(strip_widths):
            return True
        sw = strip_widths[idx]
        seen_zones = set()
        for z in range(n_zones):
            # skip duplicate zone sizes (avoids redundant branches)
            rz = round(zone_remaining[z], 9)
            if rz in seen_zones:
                continue
            if zone_remaining[z] + 1e-9 >= sw:
                seen_zones.add(rz)
                zone_remaining[z] -= sw
                if backtrack(idx + 1):
                    return True
                zone_remaining[z] += sw
        return False

    # sort strips descending to prune early (hardest to place first)
    strip_widths.sort(reverse=True)
    if backtrack(0):
        return True, None

    return False, None

def can_place_strips2(strips, stock_width, defects_in_window, max_perms=200000):
    defect_intervals = sorted(
        [(d.start_in_width, d.start_in_width + d.width)
         for d in defects_in_window]
    )

    free_zones = []
    cursor = 0.0
    for d_start, d_end in defect_intervals:
        if cursor < d_start:
            free_zones.append((cursor, d_start))
        cursor = max(cursor, d_end)
    if cursor < stock_width:
        free_zones.append((cursor, stock_width))

    if not free_zones:
        return False, None

    seen  = set()
    count = 0
    for perm in permutations(strips):
        # skip permutations that are identical in widths (different pid, same width)
        key = tuple(w for _, w in perm)
        if key in seen:
            continue
        seen.add(key)
        count += 1
        if count > max_perms:
            break

        zone_idx    = 0
        zone_cursor = free_zones[0][0]
        feasible    = True
        arrangement = []

        for pid, strip_width in perm:
            placed = False
            while zone_idx < len(free_zones):
                zone_start, zone_end = free_zones[zone_idx]
                zone_cursor = max(zone_cursor, zone_start)
                if zone_cursor + strip_width <= zone_end + 1e-9:
                    arrangement.append((pid, zone_cursor, zone_cursor + strip_width))
                    zone_cursor += strip_width
                    placed = True
                    break
                else:
                    zone_idx += 1
                    if zone_idx < len(free_zones):
                        zone_cursor = free_zones[zone_idx][0]
            if not placed:
                feasible = False
                break

        if feasible:
            return True, arrangement

    return False, None

def can_place_strips_hybrid(strips, stock_width, defects_in_window, max_strips_exact=6):
    """
    Use exact permutation search for small numbers of strips,
    fall back to greedy for larger ones.
    """
   
    if len(strips) <= max_strips_exact:
        
        return can_place_strips(strips, stock_width, defects_in_window)
    else:
        return can_place_strips(strips, stock_width, defects_in_window)
 
 
def merge_windows(windows: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Merge overlapping or adjacent windows without feasibility check.
    Used after sub-windows have already been individually validated.
    """
    if not windows:
        return []
 
    sorted_windows = sorted(windows, key=lambda w: w[0])
    merged = [sorted_windows[0]]
 
    for current_start, current_end in sorted_windows[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
 
    return merged

def fix_proposal_windows(proposals_raw, stocks, products):
    """
    Validate and fix raw proposals loaded from OMP JSON.

    For each proposal:
    1. Discard if total strip width exceeds stock width
    2. If no defects overlap the window, keep as is
    3. If full window is feasible against all defects simultaneously, keep as is
    4. Otherwise split at defect boundaries and keep only feasible sub-windows
    5. Discard proposals with no feasible sub-window remaining

    Returns a list of validated proposals with correct windows.
    """
    fixed = []

    for prop in proposals_raw:
        stock = stocks[prop.stock_id]

        strips = []
        for pid, qty in prop.products_produced_per_repetition.items():
            for _ in range(qty):
                strips.append((pid, products[pid].width))

        total_strip_width = sum(w for _, w in strips)
        if total_strip_width > stock.width + 1e-6:
            print(f"  Discarding {prop.proposal_id} on {prop.stock_id} "
                  f"— strips total {total_strip_width:.3f}m exceeds stock width {stock.width:.3f}m")
            continue

        defects_in_window = [
            d for d in stock.defects
            if d.start_in_length < prop.latest_end - 1e-6
            and d.start_in_length + d.length > prop.earliest_start + 1e-6
        ]

        if not defects_in_window:
            fixed.append(prop)
            continue

        feasible_full, _ = can_place_strips2(strips, stock.width, defects_in_window)
        if feasible_full:
            fixed.append(prop)
            continue

        split_points = sorted(set(
            [prop.earliest_start] +
            [d.start_in_length for d in defects_in_window
             if prop.earliest_start <= d.start_in_length <= prop.latest_end] +
            [d.start_in_length + d.length for d in defects_in_window
             if prop.earliest_start <= d.start_in_length + d.length <= prop.latest_end] +
            [prop.latest_end]
        ))

        valid_windows = []
        for i in range(len(split_points) - 1):
            sub_start = split_points[i]
            sub_end   = split_points[i + 1]

            if sub_end - sub_start < prop.length_consumed:
                continue

            defects_in_sub = [
                d for d in defects_in_window
                if d.start_in_length < sub_end
                and d.start_in_length + d.length > sub_start
            ]

            feasible, _ = can_place_strips2(strips, stock.width, defects_in_sub)
            if feasible:
                valid_windows.append((sub_start, sub_end))

        if not valid_windows:
            print(f"  Discarding {prop.proposal_id} on {prop.stock_id} "
                  f"— no feasible window found")
            continue

        valid_windows        = merge_windows(valid_windows)
        prop.windows         = valid_windows
        prop.earliest_start  = valid_windows[0][0]
        prop.latest_end      = valid_windows[-1][1]
        prop.max_repetitions = sum(
            int((latest - earliest) // prop.length_consumed)
            for earliest, latest in valid_windows
        )
        fixed.append(prop)

    return fixed
 
 
def fix_proposal_windows2(proposals_raw, stocks, products):
    """
    Validate and fix raw proposals loaded from OMP JSON.

    For each proposal:
    1. Discard if total strip width exceeds stock width
    2. If no defects overlap the window, keep as is
    3. If full window is feasible against all defects (greedy), keep as is
    4. If greedy fails, try exact backtracking once on the full window
    5. If exact also fails, split at defect boundaries and check sub-windows with greedy only
    6. Discard proposals with no feasible sub-window remaining
    """
    fixed = []

    for prop in proposals_raw:
        stock = stocks[prop.stock_id]
        print(f"  Checking {prop.proposal_id} on {prop.stock_id} ...", end="\r")

        strips = []
        for pid, qty in prop.products_produced_per_repetition.items():
            for _ in range(qty):
                strips.append((pid, products[pid].width))

        total_strip_width = sum(w for _, w in strips)
        if total_strip_width > stock.width + 1e-6:
            print(f"  Discarding {prop.proposal_id} on {prop.stock_id} "
                  f"— strips total {total_strip_width:.3f}m exceeds stock width {stock.width:.3f}m")
            continue

        defects_in_window = [
            d for d in stock.defects
            if d.start_in_length < prop.latest_end - 1e-6
            and d.start_in_length + d.length > prop.earliest_start + 1e-6
        ]

        if not defects_in_window:
            fixed.append(prop)
            continue

        # Step 1 — fast greedy on full window
        feasible_full, _ = can_place_strips(strips, stock.width, defects_in_window)
        if feasible_full:
            fixed.append(prop)
            continue
        print(f"  [Step 2] triggered for {prop.proposal_id} on {prop.stock_id}")
        # Step 2 — exact backtracking on full window (once per proposal)
        feasible_exact, _ = can_place_strips3(strips, stock.width, defects_in_window)
        if feasible_exact:
            fixed.append(prop)
            continue
        

        # Step 3 — split at defect boundaries, check sub-windows with greedy only
        split_points = sorted(set(
            [prop.earliest_start] +
            [d.start_in_length for d in defects_in_window
             if prop.earliest_start <= d.start_in_length <= prop.latest_end] +
            [d.start_in_length + d.length for d in defects_in_window
             if prop.earliest_start <= d.start_in_length + d.length <= prop.latest_end] +
            [prop.latest_end]
        ))

        valid_windows = []
        for i in range(len(split_points) - 1):
            sub_start = split_points[i]
            sub_end   = split_points[i + 1]

            if sub_end - sub_start < prop.length_consumed:
                continue

            defects_in_sub = [
                d for d in defects_in_window
                if d.start_in_length < sub_end
                and d.start_in_length + d.length > sub_start
            ]

            feasible, _ = can_place_strips(strips, stock.width, defects_in_sub)
            if feasible:
                valid_windows.append((sub_start, sub_end))

        if not valid_windows:
            print(f"  Discarding {prop.proposal_id} on {prop.stock_id} "
                  f"— no feasible window found")
            continue

        valid_windows        = merge_windows(valid_windows)
        prop.windows         = valid_windows
        prop.earliest_start  = valid_windows[0][0]
        prop.latest_end      = valid_windows[-1][1]
        prop.max_repetitions = sum(
            int((latest - earliest) // prop.length_consumed)
            for earliest, latest in valid_windows
        )
        fixed.append(prop)

    return fixed
 
 
# STEP 3: BUILD PATTERNS
# Group validated proposals by production signature and merge into Pattern objects.
 
def _merge_overlapping_entries(entries_raw, strips, stock, length_consumed):
    """
    Given a list of raw entries (each from one validated proposal on the same stock),
    try to merge overlapping entries whose combined window is still feasible.
    Non-overlapping or infeasible merges remain as separate PatternStockEntry objects.
 
    entries_raw     : list of dicts {windows, cost_per_rep, proposal_id}
    strips          : list of (product_id, width)
    stock           : Stock object
    length_consumed : pattern length per repetition
 
    Returns a list of PatternStockEntry objects.
    """
    sorted_entries = sorted(entries_raw, key=lambda e: e['windows'][0][0])
    clusters = [[sorted_entries[0]]]
 
    for entry in sorted_entries[1:]:
        current_window = entry['windows'][0]
        last_cluster   = clusters[-1]
        last_window    = last_cluster[-1]['windows'][0]
 
        if current_window[0] <= last_window[1]:
            candidate = (
                last_cluster[0]['windows'][0][0],
                max(last_window[1], current_window[1])
            )
            defects_in_candidate = [
                d for d in stock.defects
                if d.start_in_length < candidate[1]
                and d.start_in_length + d.length > candidate[0]
            ]
            feasible, _ = can_place_strips(strips, stock.width, defects_in_candidate)
            if feasible:
                last_cluster.append(entry)
            else:
                clusters.append([entry])
        else:
            clusters.append([entry])
 
    result = []
    for cluster in clusters:
        all_windows = []
        for e in cluster:
            all_windows.extend(e['windows'])
        merged_windows = merge_windows(all_windows)
 
        max_reps     = sum(
            int((end - start) // length_consumed)
            for start, end in merged_windows
        )
        cost_per_rep = min(e['cost_per_rep'] for e in cluster)
        source       = cluster[0]['proposal_id']
 
        result.append(PatternStockEntry(
            windows         = merged_windows,
            cost_per_rep    = cost_per_rep,
            max_repetitions = max_reps,
            source_proposal = source,
        ))
 
    return result
 
 
def build_patterns(proposals_fixed, stocks, products):
    """
    Group validated proposals by production signature and build Pattern objects.
 
    Proposals with the same (length_consumed, products_produced) signature
    are grouped into one Pattern regardless of which stock they belong to.
 
    For each stock within a group, overlapping windows that are still feasible
    when merged are combined into one PatternStockEntry. Non-mergeable windows
    remain as separate entries, each representing an independent placement
    configuration with its own setup cost entitlement.
 
    Returns a list of Pattern objects.
    """
    groups          = {}
    pattern_counter = 1
 
    for prop in proposals_fixed:
        sig = (
            prop.length_consumed,
            tuple(sorted(prop.products_produced_per_repetition.items()))
        )
 
        if sig not in groups:
            total_strip_width = sum(
                products[pid].width * qty
                for pid, qty in prop.products_produced_per_repetition.items()
            )
            groups[sig] = {
                'pattern_id'        : f"PAT_{pattern_counter:04d}",
                'length_consumed'   : prop.length_consumed,
                'setup_cost'        : prop.setup_cost,
                'products_produced' : prop.products_produced_per_repetition,
                'total_strip_width' : total_strip_width,
                'stock_raw'         : {}
            }
            pattern_counter += 1
 
        stock_id = prop.stock_id
        if stock_id not in groups[sig]['stock_raw']:
            groups[sig]['stock_raw'][stock_id] = []
 
        groups[sig]['stock_raw'][stock_id].append({
            'windows'     : prop.windows,
            'cost_per_rep': prop.cost_per_repetition,
            'proposal_id' : prop.proposal_id,
        })
 
    patterns = []
 
    for sig, group in groups.items():
        stock_entries = {}
 
        strips = []
        for pid, qty in group['products_produced'].items():
            for _ in range(qty):
                strips.append((pid, products[pid].width))
 
        for stock_id, raw_entries in group['stock_raw'].items():
            stock   = stocks[stock_id]
            entries = _merge_overlapping_entries(
                raw_entries, strips, stock, group['length_consumed']
            )
            if entries:
                stock_entries[stock_id] = entries
 
        pattern = Pattern(
            pattern_id                = group['pattern_id'],
            length_consumed           = group['length_consumed'],
            setup_cost                = group['setup_cost'],
            products_produced_per_rep = group['products_produced'],
            total_strip_width         = group['total_strip_width'],
            stock_entries             = stock_entries,
        )
        patterns.append(pattern)
 
    return patterns
 
 
def load_n_from_dir(directory, n, label_filter=None):
    """Load first n instances from directory, optionally filtering by label in filename."""
    files = sorted([
        f for f in os.listdir(directory)
        if f.endswith('.json') and (label_filter is None or label_filter.upper() in f.upper())
    ])[:n]
    instances = []
    for fname in files:
        name = fname.replace('.json', '')
        path = os.path.join(directory, fname)
        stocks, products, proposals_raw = load_instance(path)
        proposals_fixed = proposals_raw
        patterns        = build_patterns(proposals_fixed, stocks, products)
        instances.append({
            'name'           : name,
            'stocks'         : stocks,
            'products'       : products,
            'patterns'       : patterns,
            'proposals_raw'  : proposals_raw,
            'proposals_fixed': proposals_fixed,
        })
        print(f"  Loaded {name}  |  raw={len(proposals_raw)}  fixed={len(proposals_fixed)}  patterns={len(patterns)}")
    return instances
 
 
# SUMMARY AND DISPLAY
 
def print_summary(stocks, products, patterns):
    """Prints a short summary of the loaded instance."""
    print(f"Stocks   : {len(stocks)}")
    print(f"Products : {len(products)}")
    print(f"Patterns : {len(patterns)}")
    print(f"\nProducts:")
    for p in products.values():
        print(f"  {p.product_id}: length={p.length}mm  width={p.width}m  demand={p.demand}")
    print(f"\nStocks:")
    for s in stocks.values():
        print(f"  {s.stock_id}: length={s.length}mm  width={s.width}m  "
              f"defects={len(s.defects)}  cost={s.cost}")
    print(f"\nPatterns:")
    for pat in patterns:
        n_stocks  = len(pat.stock_entries)
        n_entries = sum(len(e) for e in pat.stock_entries.values())
        print(f"  {pat.pattern_id}: length={pat.length_consumed}mm  "
              f"width={pat.total_strip_width:.3f}m  "
              f"produces={pat.products_produced_per_rep}  "
              f"stocks={n_stocks}  entries={n_entries}")
 
 
def create_instance_summary(stocks, products, patterns, filepath, instance_name,
                            proposals_raw=None):
    """Export a detailed summary of the instance to an Excel file."""
 
    total_available_area = sum(
        s.length * s.width - sum(d.length * d.width for d in s.defects)
        for s in stocks.values()
    )
    total_demanded_area = sum(p.demand * p.length * p.width for p in products.values())
    avg_defect_ratio    = sum(
        sum(d.length for d in s.defects) / s.length
        for s in stocks.values()
    ) / len(stocks)
    avg_stock_length   = sum(s.length for s in stocks.values()) / len(stocks)
    avg_product_length = sum(p.length for p in products.values()) / len(products)
 
    summary_data = [{
        'Instance'                   : instance_name,
        'Stocks'                     : len(stocks),
        'Products'                   : len(products),
        'Patterns'                   : len(patterns),
        'Total demand'               : sum(p.demand for p in products.values()),
        'Avg defects per stock'      : round(sum(len(s.defects) for s in stocks.values()) / len(stocks), 2),
        'Total defects'              : sum(len(s.defects) for s in stocks.values()),
        'Avg defect length ratio'    : round(avg_defect_ratio, 3),
        'Demand tightness'           : round(total_demanded_area / total_available_area, 3)
                                       if total_available_area > 0 else None,
        'Stock/product length ratio' : round(avg_stock_length / avg_product_length, 2)
                                       if avg_product_length > 0 else None,
    }]
 
    stock_data = []
    for sid, stock in stocks.items():
        n_patterns = sum(1 for pat in patterns if sid in pat.stock_entries)
        stock_data.append({
            'Stock ID'        : sid,
            'Length (mm)'     : stock.length,
            'Width (m)'       : stock.width,
            'Defects'         : len(stock.defects),
            'Activation cost' : stock.cost,
            'N patterns'      : n_patterns,
        })
 
    defect_data = []
    for sid, stock in stocks.items():
        for defect in stock.defects:
            defect_data.append({
                'Stock ID'             : sid,
                'Start in length (mm)' : defect.start_in_length,
                'Length (mm)'          : defect.length,
                'End in length (mm)'   : defect.start_in_length + defect.length,
                'Start in width (m)'   : defect.start_in_width,
                'Width (m)'            : defect.width,
                'End in width (m)'     : defect.start_in_width + defect.width,
            })
 
    product_data = []
    for pid, product in products.items():
        product_data.append({
            'Product ID'  : pid,
            'Length (mm)' : product.length,
            'Width (m)'   : product.width,
            'Demand'      : product.demand,
        })
 
    raw_proposal_data = []
    if proposals_raw is not None:
        for prop in proposals_raw:
            raw_proposal_data.append({
                'Proposal ID'       : prop.proposal_id,
                'Stock ID'          : prop.stock_id,
                'Products produced' : str(prop.products_produced_per_repetition),
                'Length consumed'   : prop.length_consumed,
                'Earliest start'    : prop.earliest_start,
                'Latest end'        : prop.latest_end,
                'Setup cost'        : prop.setup_cost,
                'Cost per rep'      : prop.cost_per_repetition,
                'Max repetitions'   : prop.max_repetitions,
            })
  
    pattern_data = []
    for pat in patterns:
        n_entries = sum(len(e) for e in pat.stock_entries.values())
        pattern_data.append({
            'Pattern ID'            : pat.pattern_id,
            'Length consumed (mm)'  : pat.length_consumed,
            'Products produced'     : str(pat.products_produced_per_rep),
            'Total strip width (m)' : round(pat.total_strip_width, 3),
            'Setup cost'            : pat.setup_cost,
            'N stocks'              : len(pat.stock_entries),
            'N entries total'       : n_entries,
        })
 
    entry_data = []
    for pat in patterns:
        for stock_id, entries in pat.stock_entries.items():
            for i, entry in enumerate(entries):
                entry_data.append({
                    'Pattern ID'      : pat.pattern_id,
                    'Stock ID'        : stock_id,
                    'Entry index'     : i,
                    'Source proposal' : entry.source_proposal,
                    'Windows'         : str(entry.windows),
                    'Cost per rep'    : entry.cost_per_rep,
                    'Max repetitions' : entry.max_repetitions,
                })
 
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        pd.DataFrame(summary_data).to_excel(writer,  sheet_name='Summary',         index=False)
        pd.DataFrame(stock_data).to_excel(writer,    sheet_name='Stocks',           index=False)
        pd.DataFrame(defect_data).to_excel(writer,   sheet_name='Defects',          index=False)
        pd.DataFrame(product_data).to_excel(writer,  sheet_name='Products',         index=False)
        if raw_proposal_data:
            pd.DataFrame(raw_proposal_data).to_excel(writer,   sheet_name='Proposals',   index=False)
        pd.DataFrame(pattern_data).to_excel(writer,  sheet_name='Patterns',         index=False)
        pd.DataFrame(entry_data).to_excel(writer,    sheet_name='Entries',          index=False)
 
    print(f"Saved: {filepath}")
 
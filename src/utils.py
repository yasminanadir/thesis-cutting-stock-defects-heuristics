"""
load_n_from_dir         -- loads multiple instances from a directory in one call
utils.py — Utility functions for solution display and analysis.
"""
 
from src.solution import Solution, decode, evaluate
from src.instance import Pattern, PatternStockEntry, Stock
from src.constructive import placement_info
from typing import Dict, List, Tuple

 
# ---------------------------------------------------------------------------
# Solution display
# ---------------------------------------------------------------------------
 
def print_solution_state(
    solution: Solution,
    stocks: Dict,
    products: Dict,
    label: str = ""
):
    """
    Prints a detailed view of the current solution state.
    Decodes and evaluates the solution internally.
    """
    placements, fully_placed = decode(solution, stocks, verbose=False)
    cost, unmet, overproduced = evaluate(solution, placements, stocks, products)
 
    print(f"\n{'='*50}")
    if label:
        print(f"  {label}")
    print(f"  Cost        : {cost:.4f}")
    print(f"  Unmet       : {unmet}")
    print(f"  Overproduced: {overproduced}")
    print(f"  Fully placed: {fully_placed}")
    print(f"  Stocks used : {len(solution.active)}")
    for stock_id, entries in solution.active.items():
        print(f"    {stock_id}:")
        for pat, entry_idx, start_pos in entries:
            end_pos = start_pos + pat.length_consumed
            print(f"      {pat.pattern_id}  entry={entry_idx}"
                  f"  [{start_pos:.0f} -> {end_pos:.0f}]"
                  f"  produces={pat.products_produced_per_rep}")
    print(f"{'='*50}")
 
 
def print_cost_breakdown(solution: Solution, stocks: Dict, products: Dict):
    """
    Prints a detailed cost breakdown per stock, showing activation,
    setup, and repetition costs separately.
    """
    placements, fully_placed = decode(solution, stocks)
 
    print(f"\n{'='*70}")
    print(f"  COST BREAKDOWN")
    print(f"{'='*70}")
 
    total_cost       = 0.0
    total_stock_cost = 0.0
    total_setup_cost = 0.0
    total_rep_cost   = 0.0
 
    for stock_id, stock_placements in placements.items():
        if not stock_placements:
            continue
 
        stock_cost = stocks[stock_id].cost
        total_cost       += stock_cost
        total_stock_cost += stock_cost
 
        print(f"\n  Stock {stock_id}  (activation cost = {stock_cost:.4f})")
        print(f"  {'─'*60}")
        print(f"    + Stock activation       : {stock_cost:.4f}")
 
        stock_total      = stock_cost
        prev_pattern_id  = None
        prev_entry_idx   = None
 
        for pattern, entry_idx, start, end in stock_placements:
            stock_entries = pattern.stock_entries.get(stock_id, [])
            rep_cost = stock_entries[entry_idx].cost_per_rep if entry_idx < len(stock_entries) else None
 
            # setup cost — charged per run
            is_new_run = (
                prev_pattern_id is None
                or pattern.pattern_id != prev_pattern_id
                or entry_idx != prev_entry_idx
            )
            if is_new_run:
                setup = pattern.setup_cost
                total_cost       += setup
                total_setup_cost += setup
                stock_total      += setup
                print(f"    + Setup  {pattern.pattern_id:12s} entry={entry_idx}  : {setup:.4f}")
 
            # repetition cost
            if rep_cost is not None:
                total_cost    += rep_cost
                total_rep_cost += rep_cost
                stock_total   += rep_cost
                print(f"    + Rep    {pattern.pattern_id:12s} entry={entry_idx}"
                      f"  [{start:.0f}->{end:.0f}] : {rep_cost:.4f}")
            else:
                print(f"    + Rep    {pattern.pattern_id:12s} entry={entry_idx}"
                      f"  [{start:.0f}->{end:.0f}] : NO COST ENTRY")
 
            prev_pattern_id = pattern.pattern_id
            prev_entry_idx  = entry_idx
 
        print(f"  {'─'*60}")
        print(f"    Stock {stock_id} subtotal         : {stock_total:.4f}")
 
    print(f"\n{'='*70}")
    print(f"  Stock activation costs   : {total_stock_cost:.4f}")
    print(f"  Setup costs              : {total_setup_cost:.4f}")
    print(f"  Repetition costs         : {total_rep_cost:.4f}")
    print(f"{'─'*70}")
    print(f"  TOTAL COST               : {total_cost:.4f}")
    print(f"{'='*70}")
    print(f"  Fully placed             : {fully_placed}")
 
    _, unmet, over = evaluate(solution, placements, stocks, products)
    print(f"  Unmet demand             : {unmet}")
    print(f"  Overproduced             : {over}")
    print(f"{'='*70}")
 
 
# ---------------------------------------------------------------------------
# Gap utilities — used by move generators
# ---------------------------------------------------------------------------
 
def max_reps_in_gap(entry: PatternStockEntry, length_consumed: float,
                    gap_start: float, gap_end: float) -> int:
    """
    Compute the maximum number of repetitions of a pattern that fit
    in the interval [gap_start, gap_end], respecting the entry's windows.
 
    Used by move generators to assess available space between placements.
    """
    if gap_end - gap_start < length_consumed - 1e-9:
        return 0
 
    reps, _, _ = placement_info(
        entry, length_consumed, gap_start,
        int((gap_end - gap_start) // length_consumed)
    )
    return reps

import json
import os
from src.solution import Solution


def save_solution_json(
    sol,
    cost,
    is_feasible,
    unmet,
    overprod,
    elapsed_sec,
    instance_name,
    method,
    start_method,
    time_limit,
    output_dir="../outputs/solutions",
):
    """
    Save a solution to a JSON file.
    Stores the active dict as plain (pattern_id, entry_index, start_pos) triples
    — no Python objects, no pickle fragility.

    Filename convention:
        {instance}_{method}_{start_method}_TL{time_limit}.json
    """
    os.makedirs(output_dir, exist_ok=True)

    fname = f"{instance_name}_{method}_{start_method}_TL{int(time_limit)}.json"
    fpath = os.path.join(output_dir, fname)

    active_serializable = {
        stock_id: [
            [pat.pattern_id, entry_idx, start_pos]
            for pat, entry_idx, start_pos in reps
        ]
        for stock_id, reps in sol.active.items()
    }

    payload = {
        "instance"    : instance_name,
        "method"      : method,
        "start_method": start_method,
        "time_limit"  : time_limit,
        "cost"        : round(cost, 6),
        "is_feasible" : is_feasible,
        "unmet"       : unmet,
        "overprod"    : overprod,
        "elapsed_sec" : elapsed_sec,
        "active"      : active_serializable,
    }

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return fpath


def load_solution_json(fpath):
    """
    Load a saved solution JSON file.
    Returns the raw payload dict — use reconstruct_solution() to get
    a proper Solution object back.
    """
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_solution(payload, patterns):
    """
    Rebuild a Solution object from a saved JSON payload.

    payload  : dict returned by load_solution_json()
    patterns : list of Pattern objects from the same instance
               (must be the same instance the solution was saved from)

    Returns a Solution object with the same active dict as the original.
    """
    # build a lookup dict pattern_id -> Pattern for fast access
    pattern_lookup = {pat.pattern_id: pat for pat in patterns}

    sol = Solution()
    for stock_id, reps in payload["active"].items():
        for pattern_id, entry_idx, start_pos in reps:
            pat = pattern_lookup.get(pattern_id)
            if pat is None:
                raise ValueError(
                    f"Pattern {pattern_id} not found in provided patterns list — "
                    f"make sure you are loading with the correct instance."
                )
            sol.add_repetition(stock_id, pat, entry_idx, start_pos)

    return sol


def solution_exists(instance_name, method, start_method, time_limit,
                    output_dir="../outputs/solutions"):
    """
    Check whether a solution file already exists for this combination.
    Useful for resuming a run that was interrupted — skip already completed ones.
    """
    fname = f"{instance_name}_{method}_{start_method}_TL{int(time_limit)}.json"
    return os.path.exists(os.path.join(output_dir, fname))
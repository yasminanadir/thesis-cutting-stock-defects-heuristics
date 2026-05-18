"""
constructive.py
---------------
Constructive heuristics


Functions used in experiments:
  greedy_pattern_first_scarcity2  -- two-phase scarcity-first greedy (main greedy)
  greedy_stock_first              -- stock-first greedy (used as fallback in multistart)
  multistart_greedy               -- multi-start greedy with perturbation
  grasp_construct                 -- single GRASP construction with RCL
  grasp                           -- full GRASP with construction loop and final SA

Additional methods (implemented but not used in experiments):
  greedy_pattern_first            -- basic pattern-first greedy without scarcity handling
  estimate_completion_cost        -- cost estimator used by beam search
  beam_search                     -- pattern-placement beam search
"""

import math
import random
import time
from typing import Dict, List, Tuple, Optional

from src.instance import Pattern, PatternStockEntry, Stock, Product
from src.solution import Solution
 
# ---------------------------------------------------------------------------
# Placement utilities
# ---------------------------------------------------------------------------
 
def placement_info(entry: PatternStockEntry, length_consumed: float,
                   cursor: float, max_reps: int) -> Tuple[int, float, float]:
    """
    Place up to max_reps repetitions of a pattern using a specific
    PatternStockEntry, starting from cursor.
 
    Returns (n_reps_placed, start_pos, end_pos).
    start_pos — where the first repetition actually starts
    end_pos   — cursor position after placing all repetitions
    """
    reps      = 0
    start_pos = None
    pos       = cursor
 
    for window_start, window_end in entry.windows:
        if reps == max_reps:
            break
        if pos > window_end:
            continue
        pos = max(pos, window_start)
        if start_pos is None:
            start_pos = pos
        while pos + length_consumed <= window_end and reps < max_reps:
            reps += 1
            pos  += length_consumed
 
    if start_pos is None:
        start_pos = cursor
    return reps, start_pos, pos
 
 
def reps_needed(pattern: Pattern, unmet_demand: Dict[str, int]) -> int:
    """
    Minimum number of repetitions needed to cover unmet demand
    for the products this pattern produces.
    """
    needed = 0
    for pid, qty in pattern.products_produced_per_rep.items():
        if qty > 0 and unmet_demand.get(pid, 0) > 0:
            needed = max(needed, math.ceil(unmet_demand[pid] / qty))
    return needed
 
 
def _score_entry(pattern: Pattern, entry: PatternStockEntry,
                 stock_cost: float, cursor: float,
                 unmet_demand: Dict[str, int],
                 perturb: bool = False,
                 perturb_scale: float = 0.1) -> Tuple[float, int, float]:
    """
    Score a (pattern, entry) pair on a given stock.
 
    Returns (ratio, n_reps, start_pos).
    Returns (-1.0, 0, 0.0) if the entry is not useful.
    """
    n_needed = reps_needed(pattern, unmet_demand)
    if n_needed == 0:
        return -1.0, 0, 0.0
 
    n_reps, start_pos, _ = placement_info(
        entry, pattern.length_consumed, cursor, n_needed
    )
    if n_reps == 0:
        return -1.0, 0, 0.0
 
    unmet_covered = sum(
        min(qty * n_reps, unmet_demand.get(pid, 0))
        for pid, qty in pattern.products_produced_per_rep.items()
    )
    if unmet_covered == 0:
        return -1.0, 0, 0.0
 
    total_cost = stock_cost + pattern.setup_cost + entry.cost_per_rep * n_reps
    if total_cost <= 0:
        return -1.0, 0, 0.0
 
    ratio = unmet_covered / total_cost
    if perturb:
        ratio *= random.uniform(1 - perturb_scale, 1 + perturb_scale)
 
    return ratio, n_reps, start_pos
 
 
# ---------------------------------------------------------------------------
# Greedy — pattern first
# ---------------------------------------------------------------------------
 


def greedy_pattern_first_scarcity2(
    patterns: List[Pattern],
    stocks: Dict[str, Stock],
    products: Dict[str, Product],
    scarcity_threshold: float = 0.1,
    perturb_ratio: bool = False,
    perturb_scale: float = 0.1,
) -> Solution:
    """
    Two-phase scarcity-first greedy:
    Phase 1 — restricted pool of patterns covering scarce products,
              scored by coverage/cost on scarce products only
    Phase 2 — full pool, standard coverage/cost greedy
    """
    solution         = Solution()
    unmet_demand     = {pid: p.demand for pid, p in products.items()}
    opened_stocks    = set()
    cursor_per_stock = {sid: 0.0 for sid in stocks}
    n_patterns       = len(patterns)

    # --- compute product scarcity ---
    n_patterns_per_product = {pid: 0 for pid in products}
    for pattern in patterns:
        for pid, qty in pattern.products_produced_per_rep.items():
            if qty > 0:
                n_patterns_per_product[pid] += 1

    scarce_products = {
        pid for pid in products
        if n_patterns_per_product.get(pid, 0) / max(n_patterns, 1)
        < scarcity_threshold
        and products[pid].demand > 0
    }

    print(f"  Scarce products ({scarcity_threshold*100:.0f}% threshold): "
          f"{sorted(scarce_products)}")

    def _run_greedy_loop(pattern_pool, target_products):
        """
        Greedy loop restricted to pattern_pool,
        scoring coverage only on target_products.
        """
        nonlocal unmet_demand, opened_stocks, cursor_per_stock

        while any(unmet_demand.get(pid, 0) > 0 for pid in target_products):

            # score patterns in pool by coverage of target products
            pattern_scores = []
            for pattern in pattern_pool:
                coverage = sum(
                    min(qty, unmet_demand.get(pid, 0))
                    for pid, qty in pattern.products_produced_per_rep.items()
                    if pid in target_products
                )
                if coverage > 0:
                    pattern_scores.append((coverage, pattern))

            if not pattern_scores:
                break

            pattern_scores.sort(key=lambda x: x[0], reverse=True)
            candidate_patterns = [p for _, p in pattern_scores]

            best_pattern   = None
            best_stock_id  = None
            best_entry_idx = None
            best_n_reps    = 0
            best_start_pos = 0.0
            best_ratio     = -1.0

            for pattern in candidate_patterns:
                for stock_id, entries in pattern.stock_entries.items():
                    stock      = stocks[stock_id]
                    stock_cost = stock.cost if stock_id not in opened_stocks else 0.0
                    cursor     = cursor_per_stock[stock_id]

                    for entry_idx, entry in enumerate(entries):
                        # score entry but only count target products in coverage
                        n_needed = max(
                            math.ceil(unmet_demand.get(pid, 0) / qty)
                            for pid, qty in pattern.products_produced_per_rep.items()
                            if qty > 0 and pid in target_products
                            and unmet_demand.get(pid, 0) > 0
                        ) if any(
                            qty > 0 and pid in target_products
                            and unmet_demand.get(pid, 0) > 0
                            for pid, qty in pattern.products_produced_per_rep.items()
                        ) else 0

                        if n_needed == 0:
                            continue

                        n_reps, start_pos, _ = placement_info(
                            entry, pattern.length_consumed, cursor, n_needed
                        )
                        if n_reps == 0:
                            continue

                        unmet_covered = sum(
                            min(qty * n_reps, unmet_demand.get(pid, 0))
                            for pid, qty in pattern.products_produced_per_rep.items()
                            if pid in target_products
                        )
                        if unmet_covered == 0:
                            continue

                        total_cost = stock_cost + pattern.setup_cost \
                                     + entry.cost_per_rep * n_reps
                        if total_cost <= 0:
                            continue

                        ratio = unmet_covered / total_cost
                        if perturb_ratio:
                            ratio *= random.uniform(
                                1 - perturb_scale, 1 + perturb_scale
                            )

                        if ratio > best_ratio:
                            best_ratio     = ratio
                            best_pattern   = pattern
                            best_stock_id  = stock_id
                            best_entry_idx = entry_idx
                            best_n_reps    = n_reps
                            best_start_pos = start_pos

            if best_pattern is None:
                break

            # place repetitions
            entry  = best_pattern.stock_entries[best_stock_id][best_entry_idx]
            pos    = best_start_pos
            placed = 0

            while placed < best_n_reps:
                next_pos = None
                for ws, we in entry.windows:
                    if pos > we:
                        continue
                    candidate = max(pos, ws)
                    if candidate + best_pattern.length_consumed <= we + 1e-9:
                        next_pos = candidate
                        break
                if next_pos is None:
                    break
                solution.add_repetition(
                    best_stock_id, best_pattern, best_entry_idx, next_pos
                )
                opened_stocks.add(best_stock_id)
                pos    = next_pos + best_pattern.length_consumed
                placed += 1

            if placed == 0:
                break

            cursor_per_stock[best_stock_id] = max(
                cursor_per_stock[best_stock_id], pos
            )
            for pid, qty in best_pattern.products_produced_per_rep.items():
                if pid in unmet_demand:
                    unmet_demand[pid] = max(0, unmet_demand[pid] - qty * placed)

    # --- Phase 1: restricted pool, scarce products only ---
    if scarce_products:
        scarce_pool = [
            p for p in patterns
            if any(
                qty > 0 and pid in scarce_products
                for pid, qty in p.products_produced_per_rep.items()
            )
        ]
        print(f"  Phase 1: {len(scarce_pool)} patterns in scarce pool")
        _run_greedy_loop(scarce_pool, scarce_products)
        remaining_scarce = {
            pid: unmet_demand[pid] for pid in scarce_products
            if unmet_demand.get(pid, 0) > 0
        }
        print(f"  Phase 1 done. Remaining scarce unmet: {remaining_scarce}")

    # --- Phase 2: full pool, all remaining demand ---
    remaining_products = {
        pid for pid, v in unmet_demand.items() if v > 0
    }
    if remaining_products:
        print(f"  Phase 2: covering remaining {len(remaining_products)} products")
        _run_greedy_loop(patterns, remaining_products)

    remaining = {pid: v for pid, v in unmet_demand.items() if v > 0}
    if remaining:
        print(f"WARNING: greedy could not satisfy all demand")
        print(f"  Unmet demand remaining: {remaining}")
        # after greedy fails, check total available capacity 
        # on all stocks for unmet products
        for pid in unmet_demand:
            total_capacity = 0
            for pattern in patterns:
                for pid2, qty in pattern.products_produced_per_rep.items():
                    if pid2 == pid and qty > 0:
                        for stock_id in pattern.stock_entries:
                            stock = stocks[stock_id]
                            remaining = stock.length - cursor_per_stock[stock_id]
                            reps_possible = int(remaining / pattern.length_consumed)
                            total_capacity += reps_possible * qty
            print(f"{pid}: unmet={unmet_demand[pid]}  remaining_capacity={total_capacity}")

    return solution
 
# ---------------------------------------------------------------------------
# Greedy — stock first
# ---------------------------------------------------------------------------
 
def greedy_stock_first(
    patterns: List[Pattern],
    stocks: Dict[str, Stock],
    products: Dict[str, Product],
    shuffle_stocks: bool = False,
) -> Solution:
    """
    Stock-first greedy:
    1. Rank stocks by length * width / cost
    2. For each stock in order, repeatedly pick the best (pattern, entry) by ratio
    3. Place repetitions one by one, update demand and cursor
    4. Move to next stock when no useful (pattern, entry) remains
    5. Stop when all demand is met
    """
    solution         = Solution()
    unmet_demand     = {pid: p.demand for pid, p in products.items()}
    opened_stocks    = set()
    cursor_per_stock = {sid: 0.0 for sid in stocks}
 
    ranked_stocks = sorted(
        stocks.values(),
        key=lambda s: (s.length * s.width) / s.cost,
        reverse=True
    )
    if shuffle_stocks:
        random.shuffle(ranked_stocks)
 
    for stock in ranked_stocks:
        if not any(v > 0 for v in unmet_demand.values()):
            break
 
        stock_id   = stock.stock_id
        stock_cost = stock.cost if stock_id not in opened_stocks else 0.0
 
        # collect all (pattern, entry_idx, entry) available on this stock
        stock_candidates = []
        for pattern in patterns:
            if stock_id not in pattern.stock_entries:
                continue
            for entry_idx, entry in enumerate(pattern.stock_entries[stock_id]):
                stock_candidates.append((pattern, entry_idx, entry))
 
        while True:
            best_pattern   = None
            best_entry_idx = None
            best_n_reps    = 0
            best_start_pos = 0.0
            best_ratio     = -1.0
            cursor         = cursor_per_stock[stock_id]
 
            for pattern, entry_idx, entry in stock_candidates:
                ratio, n_reps, start_pos = _score_entry(
                    pattern, entry, stock_cost, cursor, unmet_demand
                )
                if ratio < 0:
                    continue
 
                if ratio > best_ratio:
                    best_ratio     = ratio
                    best_pattern   = pattern
                    best_entry_idx = entry_idx
                    best_n_reps    = n_reps
                    best_start_pos = start_pos
 
            if best_pattern is None:
                break
 
            # place repetitions one by one
            entry  = best_pattern.stock_entries[stock_id][best_entry_idx]
            pos    = best_start_pos
            placed = 0
 
            while placed < best_n_reps:
                next_pos = None
                for ws, we in entry.windows:
                    if pos > we:
                        continue
                    candidate = max(pos, ws)
                    if candidate + best_pattern.length_consumed <= we + 1e-9:
                        next_pos = candidate
                        break
 
                if next_pos is None:
                    break
 
                solution.add_repetition(
                    stock_id, best_pattern, best_entry_idx, next_pos
                )
                opened_stocks.add(stock_id)
                stock_cost = 0.0
                pos    = next_pos + best_pattern.length_consumed
                placed += 1
 
            cursor_per_stock[stock_id] = max(cursor_per_stock[stock_id], pos)
 
            for pid, qty in best_pattern.products_produced_per_rep.items():
                if pid in unmet_demand:
                    unmet_demand[pid] = max(0, unmet_demand[pid] - qty * placed)
 
    if any(v > 0 for v in unmet_demand.values()):
        print("WARNING: greedy could not satisfy all demand")
        print(f"  Unmet demand remaining: {unmet_demand}")
 
    return solution
 
 
# ---------------------------------------------------------------------------
# Multi-start
# ---------------------------------------------------------------------------
 
def multistart_greedy(
    patterns: List[Pattern],
    stocks: Dict[str, Stock],
    products: Dict[str, Product],
    n_starts: int = 20,
    seed: int = None,
    verbose: bool = False,
    repair_time_limit: float = 60.0,
) -> Solution:
    """
    Run n_starts randomized pattern-first greedy solutions.
    If at least one is feasible, return the best by cost.
    If none are feasible, fall back to n_starts stock-first solutions.
    If still none feasible, repair the least-unmet solution and return it.
    """
    from src.solution import decode, evaluate
    from src.metaheuristic import SAInfeasible, FirstImprovement


    if seed is not None:
        random.seed(seed)

    def _run_pool(method, label):
        best_feasible     = None
        best_feasible_cost = float('inf')
        best_infeasible    = None
        best_infeasible_unmet = float('inf')

        for i in range(n_starts):
            sol = (
                greedy_pattern_first_scarcity2(patterns, stocks, products, perturb_ratio=True)
                if method == 'PF'
                else greedy_stock_first(patterns, stocks, products, shuffle_stocks=True)
            )
            placements, _ = decode(sol, stocks)
            cost, unmet, _ = evaluate(sol, placements, stocks, products)
            total_unmet = sum(unmet.values())

            if verbose:
                print(f"  {label} [{i+1:2d}]  cost={cost:.2f}  unmet={total_unmet}"
                      + ("  ✓" if total_unmet == 0 else ""))

            if total_unmet == 0:
                if cost < best_feasible_cost:
                    best_feasible      = sol
                    best_feasible_cost = cost
            else:
                if total_unmet < best_infeasible_unmet:
                    best_infeasible       = sol
                    best_infeasible_unmet = total_unmet

        return best_feasible, best_infeasible, best_infeasible_unmet

    # --- step 1: try PF ---
    if verbose:
        print(f"  --- Pattern-first pool ({n_starts} starts) ---")
    pf_feasible, pf_best_infeasible, pf_best_unmet = _run_pool('PF', 'PF')

    if pf_feasible is not None:
        if verbose:
            print(f"  PF found feasible solution ✓")
        return pf_feasible

    # --- step 2: fallback to SF ---
    if verbose:
        print(f"  PF found no feasible solution — falling back to SF")
        print(f"  --- Stock-first pool ({n_starts} starts) ---")
    sf_feasible, sf_best_infeasible, sf_best_unmet = _run_pool('SF', 'SF')

    if sf_feasible is not None:
        if verbose:
            print(f"  SF found feasible solution ✓")
        return sf_feasible

    # --- step 3: repair least unmet across PF and SF ---
    
    if verbose:
        print(f"  SF found no feasible solution — repairing least unmet")

    best_infeasible = (
        pf_best_infeasible if pf_best_unmet <= sf_best_unmet else sf_best_infeasible
    )

    fi = FirstImprovement(stocks, products, patterns)
    repaired, _ = fi.repair(best_infeasible, time_limit=60.0)

    if verbose:
        placements, _ = decode(repaired, stocks)
        _, unmet, _   = evaluate(repaired, placements, stocks, products)
        total_unmet   = sum(unmet.values())
        print(f"  Repair result: unmet={total_unmet}"
              + ("  ✓" if total_unmet == 0 else "  ✗ still infeasible"))
    
    return repaired
        
    """
    #Version if you want SA with soft demand as the repair function
    # --- step 3: repair least unmet across PF and SF ---
    if verbose:
        print(f"  SF found no feasible solution — running SAInfeasible repair")

    best_infeasible = (
        pf_best_infeasible if pf_best_unmet <= sf_best_unmet else sf_best_infeasible
    )

    sa_inf = SAInfeasible(
        stocks, products, patterns,
        T_init               = None,
        T_min                = 1e-3,
        max_iterations       = 10_000_000,
        time_limit           = 60.0,
        penalty              = 1000.0,
        neighborhood_weights = {
            "remove"              : 1.0,
            "swap"                : 2.0,
            "relocate"            : 4.0,
            "stock_reset"         : 5.0,
            "pattern_replace_all" : 1.0,
            "insert"              : 3.0,
            "stock_open"          : 5.0,
            "close_open"          : 1.0,
        },
        verbose = verbose,
    )
    repaired, _, _, _ = sa_inf.run(best_infeasible)

    if verbose:
        placements, _ = decode(repaired, stocks)
        _, unmet, _   = evaluate(repaired, placements, stocks, products)
        total_unmet   = sum(unmet.values())
        print(f"  SAInfeasible result: unmet={total_unmet}"
            + ("  ✓" if total_unmet == 0 else "  ✗ still infeasible"))
    return repaired
    """
    

def grasp_construct(
    patterns     : List[Pattern],
    stocks       : Dict[str, Stock],
    products     : Dict[str, Product],
    alpha        : float = 0.3,
    verbose      : bool  = False,
) -> Solution:
    """
    Single GRASP construction — randomized greedy with RCL.
    
    At each step:
    1. Score all (pattern, stock, entry) candidates by ratio
    2. Build RCL: candidates with score >= score_max - alpha*(score_max - score_min)
    3. Pick randomly from RCL
    4. Place repetitions, update demand
    """
    solution         = Solution()
    unmet_demand     = {pid: p.demand for pid, p in products.items()}
    opened_stocks    = set()
    cursor_per_stock = {sid: 0.0 for sid in stocks}
    step             = 0

    while any(v > 0 for v in unmet_demand.values()):
        step += 1

        # --- score all candidates ---
        scored_candidates = []

        for pattern in patterns:
            coverage = sum(
                min(qty, unmet_demand.get(pid, 0))
                for pid, qty in pattern.products_produced_per_rep.items()
            )
            if coverage == 0:
                continue

            for stock_id, entries in pattern.stock_entries.items():
                stock      = stocks[stock_id]
                stock_cost = stock.cost if stock_id not in opened_stocks else 0.0
                cursor     = cursor_per_stock[stock_id]

                for entry_idx, entry in enumerate(entries):
                    ratio, n_reps, start_pos = _score_entry(
                        pattern, entry, stock_cost, cursor, unmet_demand
                    )
                    if ratio < 0:
                        continue
                    scored_candidates.append(
                        (ratio, pattern, stock_id, entry_idx, n_reps, start_pos)
                    )

        if not scored_candidates:
            print("WARNING: GRASP construction could not satisfy all demand")
            print(f"  Unmet demand remaining: {unmet_demand}")
            break

        # --- build RCL ---
        score_max = max(c[0] for c in scored_candidates)
        score_min = min(c[0] for c in scored_candidates)
        threshold = score_max - alpha * (score_max - score_min)
        rcl       = [c for c in scored_candidates if c[0] >= threshold]

        # --- pick randomly from RCL ---
        ratio, best_pattern, best_stock_id, best_entry_idx, best_n_reps, best_start_pos = \
            random.choice(rcl)

        if verbose:
            print(f"  Step {step:3d}  "
                  f"candidates={len(scored_candidates):5d}  "
                  f"rcl={len(rcl):4d}  "
                  f"pattern={best_pattern.pattern_id:>10}  "
                  f"stock={best_stock_id:>5}  "
                  f"n_reps={best_n_reps}  "
                  f"unmet={sum(unmet_demand.values())}")

        # --- place repetitions one by one ---
        entry  = best_pattern.stock_entries[best_stock_id][best_entry_idx]
        pos    = best_start_pos
        placed = 0

        while placed < best_n_reps:
            next_pos = None
            for ws, we in entry.windows:
                if pos > we:
                    continue
                candidate = max(pos, ws)
                if candidate + best_pattern.length_consumed <= we + 1e-9:
                    next_pos = candidate
                    break
            if next_pos is None:
                break
            solution.add_repetition(
                best_stock_id, best_pattern, best_entry_idx, next_pos
            )
            opened_stocks.add(best_stock_id)
            pos    = next_pos + best_pattern.length_consumed
            placed += 1

        cursor_per_stock[best_stock_id] = max(
            cursor_per_stock[best_stock_id], pos
        )

        for pid, qty in best_pattern.products_produced_per_rep.items():
            if pid in unmet_demand:
                unmet_demand[pid] = max(0, unmet_demand[pid] - qty * placed)

    if verbose:
        print(f"  Construction done: {step} steps  "
              f"unmet={sum(unmet_demand.values())}  "
              f"stocks_opened={len(opened_stocks)}")

    return solution


def grasp(
    patterns              : List[Pattern],
    stocks                : Dict[str, Stock],
    products              : Dict[str, Product],
    n_restarts            : int   = 10000,
    alpha                 : float = 0.3,
    run_local_search      : bool  = True,
    sd_time_limit         : float = 30.0,
    fi_time_limit         : float = 60.0,
    repair_time_limit     : float = 60.0,
    sd_active_moves       : set   = None,
    fi_neighborhood_weights: dict = None,
    outer_method          : str   = "FI",       
    T_init                : float = 100.0,       
    alpha_sa              : float = 0.991595,   
    sa_neighborhood_weights: dict = None,      
    seed                  : int   = None,
    time_limit            : float = None,
    verbose               : bool  = False,
    convergence_csv_path  : str   = None,  
    log_interval          : float = 0.1,
) -> Solution:
    """
    GRASP — Greedy Randomized Adaptive Search Procedure.
    
    alpha=0 → pure greedy (same solution every restart)
    alpha=1 → pure random
    alpha=0.3 → top 30% of candidates randomly chosen

    Pipeline:
    1. n_restarts of randomized greedy construction
    2. If no feasible solution → repair infeasibles one by one until one is feasible
    3. Run SD-relocate on every feasible solution
    4. Run FI on best SD solution
    """
    from src.solution import decode, evaluate

    if seed is not None:
        random.seed(seed)

    feasible_solutions   = []  # list of (cost, sol)
    infeasible_solutions = []  # list of (total_unmet, sol)

    # --- construction phase ---
    start_time         = time.time()
    construction_limit = (time_limit - fi_time_limit) \
                        if time_limit else float('inf')
    restart            = 0
    while restart < n_restarts and \
            (time.time() - start_time) < construction_limit:
        
        restart += 1
        sol = grasp_construct(patterns, stocks, products, alpha=alpha)

        placements, _ = decode(sol, stocks)
        cost, unmet, _ = evaluate(sol, placements, stocks, products)
        total_unmet    = sum(unmet.values())

        if verbose:
            print(f"  GRASP [{restart+1:2d}/{n_restarts}]  "
                  f"cost={cost:.2f}  unmet={total_unmet}"
                  + ("  ✓" if total_unmet == 0 else ""))
            if total_unmet > 0:
                print(f"    unmet detail: {unmet}")
                print(f"    open stocks : {len(sol.active)}")
                # check production per product
                produced = {}
                for sid, entries in sol.active.items():
                    for pat, eidx, _ in entries:
                        for pid, qty in pat.products_produced_per_rep.items():
                            produced[pid] = produced.get(pid, 0) + qty
                for pid, demand in products.items():
                    prod = produced.get(pid, 0)
                    dem  = demand.demand
                    print(f"    {pid}: produced={prod}  demand={dem}  "
                        f"{'OK' if prod >= dem else 'UNMET'}")

        if total_unmet == 0:
            feasible_solutions.append((cost, sol))
        else:
            infeasible_solutions.append((total_unmet, sol))

    # --- if no feasible solution found — repair infeasibles one by one ---
    if not feasible_solutions:
        if verbose:
            print(f"\n  No feasible solution found — repairing infeasible solutions")
        from src.metaheuristic import FirstImprovement

        infeasible_solutions.sort(key=lambda x: x[0])
    
        repair_start = time.time()
        for total_unmet, sol in infeasible_solutions:
            elapsed   = time.time() - repair_start
            remaining = repair_time_limit - elapsed
            if remaining <= 0:
                if verbose:
                    print(f"  Repair time limit reached — stopping")
                break

            if verbose:
                print(f"  Repairing solution with unmet={total_unmet}  "
                    f"remaining={remaining:.1f}s")

            fi_repair = FirstImprovement(stocks, products, patterns,
                                        max_iterations = 999999,
                                        time_limit     = repair_time_limit,
                                        verbose        = verbose)
            repaired, _ = fi_repair.repair(sol,
                                        max_repair_iterations = 999999,
                                        time_limit            = repair_time_limit)
            # SAInfeasible repair (if want to use sa with soft demand constraint as repair — uncomment to use)
            # from src.metaheuristic import SAInfeasible
            # sa_inf = SAInfeasible(
            #     stocks, products, patterns,
            #     T_init               = None,
            #     T_min                = 1e-3,
            #     max_iterations       = 10_000_000,
            #     time_limit           = remaining,
            #     penalty              = 1000.0,
            #     neighborhood_weights = {
            #         "remove"              : 1.0,
            #         "swap"                : 2.0,
            #         "relocate"            : 4.0,
            #         "stock_reset"         : 5.0,
            #         "pattern_replace_all" : 1.0,
            #         "insert"              : 3.0,
            #         "stock_open"          : 5.0,
            #         "close_open"          : 1.0,
            #     },
            #     verbose = verbose,
            # )
            # repaired, _, _, _ = sa_inf.run(sol)
            placements, _ = decode(repaired, stocks)
            cost, unmet, _ = evaluate(repaired, placements, stocks, products)
            total_unmet_after = sum(unmet.values())

            if verbose:
                print(f"  After repair: cost={cost:.2f}  unmet={total_unmet_after}"
                    + ("  ✓" if total_unmet_after == 0 else "  ✗"))

            if total_unmet_after == 0:
                feasible_solutions.append((cost, repaired))
                break

    # --- if still no feasible solution ---
    if not feasible_solutions:
        if verbose:
            print(f"  WARNING: could not find feasible solution after repair")
        best_unmet_sol = min(infeasible_solutions, key=lambda x: x[0])
        return best_unmet_sol[1], float('inf')

    # --- run local search ---
    if run_local_search:
        from src.metaheuristic import FirstImprovement, SteepestImprovement

        if verbose:
            print(f"\n  Running SD-relocate on {len(feasible_solutions)} "
                  f"feasible solutions  time_limit={sd_time_limit}s")

        sd_improved = []
        for cost_init, sol in feasible_solutions:
            sd = SteepestImprovement(
                stocks, products, patterns,
                max_iterations = 999,
                time_limit     = sd_time_limit,
                active_moves   = sd_active_moves,
                verbose        = False,
            )
            sol_sd, _, _, _, _ = sd.run(sol)
            placements, _ = decode(sol_sd, stocks)
            cost_sd, unmet_sd, _ = evaluate(sol_sd, placements, stocks, products)
            total_unmet_sd = sum(unmet_sd.values())

            if verbose:
                print(f"  init={cost_init:.2f}  after_sd={cost_sd:.2f}  "
                    f"unmet={total_unmet_sd}")

            sd_improved.append((cost_sd, sol_sd, total_unmet_sd))
        
        feasible_sd   = [(c, s, u) for c, s, u in sd_improved if u == 0]
        infeasible_sd = [(c, s, u) for c, s, u in sd_improved if u > 0]

        if feasible_sd:
            best_cost_sd, best_sol_sd, _ = min(feasible_sd, key=lambda x: x[0])
            if verbose:
                print(f"\n  Best SD solution (feasible): cost={best_cost_sd:.2f}")
        else:
            best_cost_sd, best_sol_sd, _ = min(infeasible_sd, key=lambda x: x[2])
            if verbose:
                print(f"\n  No feasible SD solution — using least unmet: "
                    f"cost={best_cost_sd:.2f}")

        if verbose:
            print(f"\n  Running FI on best SD solution  "
                  f"cost={best_cost_sd:.2f}  time_limit={fi_time_limit}s")
        
        fi_time = min(
            fi_time_limit,
            (time_limit - (time.time() - start_time)) if time_limit else fi_time_limit
        )

        if outer_method == "SA":
            from src.metaheuristic import SimulatedAnnealingAdaptiveAlpha
            if verbose:
                print(f"\n  Running SA on best SD solution  "
                      f"cost={best_cost_sd:.2f}  time_limit={fi_time:.1f}s")
            safe_outer_weights = sa_neighborhood_weights or {
                "remove"      : 1.0,
                "swap"        : 2.0,
                "relocate"    : 1.0,
                "stock_reset" : 3.0,
                "pattern_replace_all" : 1.0, 
                "insert"      : 1.0
            }
            outer_ls = SimulatedAnnealingAdaptiveAlpha(
                stocks, products, patterns,
                time_limit           = max(1.0, fi_time),
                T_init               = T_init,
                T_min                = 1e-3,
                neighborhood_weights = safe_outer_weights,
                verbose              = False,
            )
        else:  # default FI
            if verbose:
                print(f"\n  Running FI on best SD solution  "
                      f"cost={best_cost_sd:.2f}  time_limit={fi_time:.1f}s")
            safe_outer_weights = fi_neighborhood_weights or {
                "remove"             : 1.0,
                "swap"               : 3.0,
                "relocate"           : 2.0,
                "stock_reset"        : 2.0,
                "pattern_replace_all": 1.0,
            }
            outer_ls = FirstImprovement(
                stocks, products, patterns,
                max_iterations       = 999999,
                time_limit           = max(1.0, fi_time),
                neighborhood_weights = safe_outer_weights,
                verbose              = False,
            )
        outer_ls.convergence_csv_path = convergence_csv_path  # ← ADD
        outer_ls.log_interval = log_interval 
        outer_ls.time_offset = time.time() - start_time
        sol_out, _, _, _ = outer_ls.run(best_sol_sd)
        placements, _ = decode(sol_out, stocks)
        best_cost, _, _ = evaluate(sol_out, placements, stocks, products)
        best_sol = sol_out

        if verbose:
            print(f"  after_{outer_method}={best_cost:.2f}")

    else:
        best_cost, best_sol = min(feasible_solutions, key=lambda x: x[0])

    return best_sol, best_cost

# Additional method created but not implemented in the end

def greedy_pattern_first(
    patterns: List[Pattern],
    stocks: Dict[str, Stock],
    products: Dict[str, Product],
    perturb_ratio: bool = False,
    perturb_scale: float = 0.1,
) -> Solution:
    """
    Pattern-first greedy:
    1. Score all patterns by total unmet demand covered in one repetition
    2. Keep only the top_patterns most useful ones
    3. For each candidate pattern, iterate over all (stock, entry) pairs
    4. Pick the best (pattern, stock, entry) combination by ratio
    5. Place repetitions one by one, update demand and cursor, repeat
    """
    solution          = Solution()
    unmet_demand      = {pid: p.demand for pid, p in products.items()}
    opened_stocks     = set()
    cursor_per_stock  = {sid: 0.0 for sid in stocks}
 
    while any(v > 0 for v in unmet_demand.values()):
 
        # score all patterns by coverage of unmet demand in one rep
        pattern_scores = []
        for pattern in patterns:
            coverage = sum(
                min(qty, unmet_demand.get(pid, 0))
                for pid, qty in pattern.products_produced_per_rep.items()
            )
            if coverage > 0:
                pattern_scores.append((coverage, pattern))
 
        if not pattern_scores:
            print("WARNING: greedy could not satisfy all demand")
            print(f"  Unmet demand remaining: {unmet_demand}")
            break
 
        pattern_scores.sort(key=lambda x: x[0], reverse=True)
        candidate_patterns = [p for _, p in pattern_scores]
 
        best_pattern   = None
        best_stock_id  = None
        best_entry_idx = None
        best_n_reps    = 0
        best_start_pos = 0.0
        best_ratio     = -1.0
 
        for pattern in candidate_patterns:
            for stock_id, entries in pattern.stock_entries.items():
                stock      = stocks[stock_id]
                stock_cost = stock.cost if stock_id not in opened_stocks else 0.0
                cursor     = cursor_per_stock[stock_id]
 
                for entry_idx, entry in enumerate(entries):
                    ratio, n_reps, start_pos = _score_entry(
                        pattern, entry, stock_cost, cursor,
                        unmet_demand, perturb_ratio, perturb_scale
                    )
                    if ratio < 0:
                        continue
 
                    if ratio > best_ratio:
                        best_ratio     = ratio
                        best_pattern   = pattern
                        best_stock_id  = stock_id
                        best_entry_idx = entry_idx
                        best_n_reps    = n_reps
                        best_start_pos = start_pos
 
        if best_pattern is None:
            print("WARNING: greedy could not satisfy all demand")
            print(f"  Unmet demand remaining: {unmet_demand}")
            print(f"  Active stocks: {list(opened_stocks)}")
            print(f"  Cursors: {cursor_per_stock}")
            if best_pattern is None:
            # check if any pattern covers unmet at all
                for pattern in patterns:
                    coverage = sum(min(qty, unmet_demand.get(pid, 0))
                                for pid, qty in pattern.products_produced_per_rep.items())
                    if coverage > 0:
                        print(f"  pattern {pattern.pattern_id} covers unmet but was not selected")
                        for stock_id in list(pattern.stock_entries.keys())[:3]:
                            for eidx, entry in enumerate(pattern.stock_entries[stock_id]):
                                
                                ratio, n, sp = _score_entry(
                                    pattern, entry, stocks[stock_id].cost,
                                    cursor_per_stock[stock_id], unmet_demand
                                )
                                print(f"    stock={stock_id} entry={eidx} "
                                    f"ratio={ratio:.4f} n={n} cursor={cursor_per_stock[stock_id]:.1f}")
                        break
                break

             # debug — sample scores on inactive stocks
            print("  Sample scores on inactive stocks:")
            inactive = [sid for sid in stocks if sid not in opened_stocks][:3]
            for pat in candidate_patterns[:3]:
                for sid in inactive:
                    if sid not in pat.stock_entries:
                        continue
                    for eidx, entry in enumerate(pat.stock_entries[sid]):
                        ratio, n, sp = _score_entry(pat, entry, stocks[sid].cost,
                                                    0.0, unmet_demand)
                        print(f"    {pat.pattern_id} on {sid} e{eidx}: "
                            f"ratio={ratio:.4f} n={n} sp={sp:.0f}")

    
            break
 
        # place repetitions one by one
        entry  = best_pattern.stock_entries[best_stock_id][best_entry_idx]
        pos    = best_start_pos
        placed = 0
 
        while placed < best_n_reps:
            next_pos = None
            for ws, we in entry.windows:
                if pos > we:
                    continue
                candidate = max(pos, ws)
                if candidate + best_pattern.length_consumed <= we + 1e-9:
                    next_pos = candidate
                    break
 
            if next_pos is None:
                break
 
            solution.add_repetition(
                best_stock_id, best_pattern, best_entry_idx, next_pos
            )
            opened_stocks.add(best_stock_id)
            pos    = next_pos + best_pattern.length_consumed
            placed += 1
        
        if placed==0:
            break
 
        cursor_per_stock[best_stock_id] = max(cursor_per_stock[best_stock_id], pos)
 
        for pid, qty in best_pattern.products_produced_per_rep.items():
            if pid in unmet_demand:
                unmet_demand[pid] = max(0, unmet_demand[pid] - qty * placed)
 
    return solution


def estimate_completion_cost(unmet, opened_stocks, stocks, patterns):
    estimated_cost = 0.0
    unmet_copy     = dict(unmet)
    
    remaining_stocks = [s for sid, s in stocks.items()
                        if sid not in opened_stocks]
    
    for stock in remaining_stocks:
        if not any(v > 0 for v in unmet_copy.values()):
            break  # demand satisfied, stop
        
        stock_id   = stock.stock_id
        stock_cost = stock.cost
        cursor     = 0.0
        opened     = False

        # fill this stock as much as possible
        while True:
            best_ratio   = -1
            best_pattern = None
            best_entry_idx = None
            best_n_reps  = 0
            best_start   = 0.0

            for pattern in patterns:
                if stock_id not in pattern.stock_entries:
                    continue
                for entry_idx, pse in enumerate(pattern.stock_entries[stock_id]):
                    ratio, n_reps, start_pos = _score_entry(
                        pattern, pse,
                        stock_cost if not opened else 0.0,
                        cursor, unmet_copy
                    )
                    if ratio > best_ratio:
                        best_ratio     = ratio
                        best_pattern   = pattern
                        best_entry_idx = entry_idx
                        best_n_reps    = n_reps
                        best_start     = start_pos

            if best_pattern is None:
                break  # nothing useful on this stock anymore

            # update cost
            if not opened:
                estimated_cost += stock_cost
                opened          = True
            estimated_cost += best_pattern.setup_cost
            estimated_cost += best_pattern.stock_entries[stock_id][best_entry_idx].cost_per_rep * best_n_reps

            # update unmet and cursor
            for pid, qty in best_pattern.products_produced_per_rep.items():
                unmet_copy[pid] = max(0, unmet_copy.get(pid, 0) - qty * best_n_reps)
            cursor += best_n_reps * best_pattern.length_consumed

    # penalize if still unmet after all remaining stocks
    if any(v > 0 for v in unmet_copy.values()):
        estimated_cost += sum(unmet_copy.values()) * 99999

    return estimated_cost


def beam_search(
    patterns     : List[Pattern],
    stocks       : Dict[str, Stock],
    products     : Dict[str, Product],
    beam_width   : int   = 5,
    top_m        : int   = 20,
    score_fn     : str   = 'cost_per_demand',
    perturb      : bool  = False,
    perturb_scale: float = 0.2,
    verbose      : bool  = False,
) -> Solution:

    # --- initialize beam ---
    beam = [{
        'solution'      : Solution(),
        'unmet'         : {pid: p.demand for pid, p in products.items()},
        'cost'          : 0.0,
        'demand_covered': 0,
        'opened_stocks' : set(),
        'cursors'       : {sid: 0.0 for sid in stocks},
    }]

    step = 0
    while True:
        step += 1

        if not any(
            any(v > 0 for v in entry['unmet'].values())
            for entry in beam
        ):
            break

        # --- generate all candidates cheaply ---
        candidates = []

        for i, entry in enumerate(beam):

            if not any(v > 0 for v in entry['unmet'].values()):
                candidates.append({
                    'parent_idx'    : i,
                    'pattern'       : None,
                    'stock_id'      : None,
                    'entry_idx'     : None,
                    'start_pos'     : None,
                    'n_reps'        : 0,
                    'cost'          : entry['cost'],
                    'demand_covered': entry['demand_covered'],
                    'unmet'         : entry['unmet'],
                    'opened_stocks' : entry['opened_stocks'],
                    'cursors'       : entry['cursors'],
                    'score'         : entry['cost'] / max(1, entry['demand_covered']),
                })
                continue

            found_any = False

            for pattern in patterns:
                for stock_id, pse_list in pattern.stock_entries.items():
                    stock      = stocks[stock_id]
                    stock_cost = stock.cost if stock_id not in entry['opened_stocks'] else 0.0
                    cursor     = entry['cursors'][stock_id]

                    for entry_idx, pse in enumerate(pse_list):
                        ratio, n_reps, start_pos = _score_entry(
                            pattern, pse, stock_cost, cursor,
                            entry['unmet'], perturb, perturb_scale
                        )
                        if ratio < 0:
                            continue

                        found_any = True

                        new_cost     = entry['cost']
                        new_covered  = entry['demand_covered']
                        new_unmet    = dict(entry['unmet'])
                        new_opened   = set(entry['opened_stocks'])
                        new_cursors  = dict(entry['cursors'])

                        pos    = start_pos
                        placed = 0
                        for _ in range(n_reps):
                            next_pos = None
                            for ws, we in pse.windows:
                                if pos > we:
                                    continue
                                candidate_pos = max(pos, ws)
                                if candidate_pos + pattern.length_consumed <= we + 1e-9:
                                    next_pos = candidate_pos
                                    break
                            if next_pos is None:
                                break
                            pos    = next_pos + pattern.length_consumed
                            placed += 1

                        if placed == 0:
                            continue

                        if stock_id not in new_opened:
                            new_cost   += stock_cost
                            new_opened.add(stock_id)
                        new_cost += pattern.setup_cost
                        new_cost += pse.cost_per_rep * placed

                        for pid, qty in pattern.products_produced_per_rep.items():
                            reduction      = min(qty * placed, new_unmet.get(pid, 0))
                            new_unmet[pid] = max(0, new_unmet.get(pid, 0) - qty * placed)
                            new_covered   += reduction

                        new_cursors[stock_id] = max(new_cursors[stock_id], pos)

                        # --- fast score for all methods ---
                        if score_fn == 'greedy_ratio':
                            marginal_cost    = new_cost - entry['cost']
                            marginal_covered = new_covered - entry['demand_covered']
                            fast_score = -marginal_covered / max(1e-9, marginal_cost)
                        else:
                            # cost_per_demand and cost_plus_estimate both use this as fast pre-filter
                            fast_score = new_cost / max(1, new_covered)

                        candidates.append({
                            'parent_idx'    : i,
                            'pattern'       : pattern,
                            'stock_id'      : stock_id,
                            'entry_idx'     : entry_idx,
                            'start_pos'     : start_pos,
                            'n_reps'        : placed,
                            'cost'          : new_cost,
                            'demand_covered': new_covered,
                            'unmet'         : new_unmet,
                            'opened_stocks' : new_opened,
                            'cursors'       : new_cursors,
                            'score'         : fast_score,
                        })

            if not found_any:
                candidates.append({
                    'parent_idx'    : i,
                    'pattern'       : None,
                    'stock_id'      : None,
                    'entry_idx'     : None,
                    'start_pos'     : None,
                    'n_reps'        : 0,
                    'cost'          : entry['cost'],
                    'demand_covered': entry['demand_covered'],
                    'unmet'         : entry['unmet'],
                    'opened_stocks' : entry['opened_stocks'],
                    'cursors'       : entry['cursors'],
                    'score'         : entry['cost'] / max(1, entry['demand_covered']),
                })

        if not candidates:
            break

        # --- sort by fast score ---
        candidates.sort(key=lambda x: x['score'])

        # --- deduplicate by (pattern, stock) choice ---
        seen  = set()
        top_m_list = []
        for c in candidates:
            choice = (
                c['pattern'].pattern_id if c['pattern'] else None,
                c['stock_id']
            )
            if choice not in seen:
                seen.add(choice)
                top_m_list.append(c)
            if len(top_m_list) == max(top_m, beam_width):
                break

        # --- expensive estimation only on top-m for cost_plus_estimate ---
        if score_fn == 'cost_plus_estimate':
            for c in top_m_list:
                c['score'] = c['cost'] + estimate_completion_cost(
                    c['unmet'], c['opened_stocks'], stocks, patterns
                )
            top_m_list.sort(key=lambda x: x['score'])

        # --- keep top-k ---
        top_k = top_m_list[:beam_width]

        if verbose:
            print(f"\n  Step {step}  candidates={len(candidates)}"
                  f"  unique={len(seen)}  kept={len(top_k)}"
                  f"  total_unmet={sum(sum(e['unmet'].values()) for e in beam)}")
            print(f"  {'─'*60}")
            for rank, c in enumerate(top_k):
                print(f"    [{rank+1}]"
                      f"  pattern={c['pattern'].pattern_id if c['pattern'] else 'None':>10}"
                      f"  stock={c['stock_id'] if c['stock_id'] else 'None':>5}"
                      f"  n_reps={c['n_reps']}"
                      f"  cost={c['cost']:>10.2f}"
                      f"  covered={c['demand_covered']:>4}"
                      f"  unmet={sum(c['unmet'].values()):>3}"
                      f"  score={c['score']:>10.4f}")

        # --- materialize only top-k via deep copy ---
        new_beam = []
        for c in top_k:
            parent  = beam[c['parent_idx']]
            new_sol = copy_solution(parent['solution'])

            if c['pattern'] is not None:
                pos    = c['start_pos']
                placed = 0
                pse    = c['pattern'].stock_entries[c['stock_id']][c['entry_idx']]

                for _ in range(c['n_reps']):
                    next_pos = None
                    for ws, we in pse.windows:
                        if pos > we:
                            continue
                        candidate_pos = max(pos, ws)
                        if candidate_pos + c['pattern'].length_consumed <= we + 1e-9:
                            next_pos = candidate_pos
                            break
                    if next_pos is None:
                        break
                    new_sol.add_repetition(
                        c['stock_id'], c['pattern'], c['entry_idx'], next_pos
                    )
                    pos    = next_pos + c['pattern'].length_consumed
                    placed += 1

            new_beam.append({
                'solution'      : new_sol,
                'unmet'         : c['unmet'],
                'cost'          : c['cost'],
                'demand_covered': c['demand_covered'],
                'opened_stocks' : c['opened_stocks'],
                'cursors'       : c['cursors'],
            })

        beam = new_beam

        if not beam:
            break

    # --- return best feasible solution ---
    feasible = [e for e in beam if not any(v > 0 for v in e['unmet'].values())]

    if feasible:
        return min(feasible, key=lambda e: e['cost'])['solution']

    # fallback — repair least unmet
    best_entry  = min(beam, key=lambda e: sum(e['unmet'].values()))
    fi          = FirstImprovement(stocks, products, patterns)
    repaired, _ = fi.repair(best_entry['solution'], time_limit=60.0)
    return repaired
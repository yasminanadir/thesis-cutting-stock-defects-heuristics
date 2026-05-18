"""
metaheuristic.py — Local search for the 1.5D CSP with defects.
Pattern-based approach.

Two base variants:
    - SteepestImprovement : evaluate all moves, apply the best improving one
    - FirstImprovement    : randomly sample moves, apply first improving one

Both variants share:
    - repair() : phase 1 — fix feasibility before cost optimisation

Designed to be extended to:
    - TabuSearch         : extends SteepestImprovement with a tabu list
    - SimulatedAnnealing : extends FirstImprovement with probabilistic acceptance
"""

import copy
import math
import time
import random
from typing import Dict, List, Optional, Tuple

from src.instance import Pattern, PatternStockEntry, Stock, Product
from src.solution import Solution, decode, evaluate, copy_solution
from src.constructive import placement_info, reps_needed
from src.moves import feasible_insertions, _gaps, _first_feasible_in_gap, pattern_replace_all, merge_stocks
from src.utils import print_solution_state


# ---------------------------------------------------------------------------
# Sliding helpers
# ---------------------------------------------------------------------------

def _slide_left(entry: PatternStockEntry, length_consumed: float,
                current_start: float, min_start: float) -> Optional[float]:
    """
    Try to slide a repetition as early as possible within its entry windows,
    but not before min_start (the end of the previous repetition).

    Returns the earliest feasible start position, or None if unchanged.
    """
    for ws, we in entry.windows:
        candidate = max(min_start, ws)
        if candidate + length_consumed <= we + 1e-9 and candidate < current_start - 1e-9:
            return candidate
    return None


def _slide_right(entry: PatternStockEntry, length_consumed: float,
                 current_start: float, max_end: float) -> Optional[float]:
    """
    Try to slide a repetition as late as possible within its entry windows,
    but ensuring its end does not exceed max_end (the start of the next repetition).

    Returns the latest feasible start position, or None if unchanged.
    """
    best = None
    for ws, we in entry.windows:
        # latest start within this window such that end <= max_end
        latest = min(we, max_end) - length_consumed
        candidate = max(current_start, ws)
        if candidate <= latest + 1e-9:
            pos = latest
            if pos > current_start + 1e-9:
                best = pos
    return best


def _try_create_gap(solution: Solution, stock_id: str,
                    gap_idx: int, stocks: Dict) -> None:
    """
    Maximize the gap at gap_idx by cascading slides:
        - All repetitions to the LEFT of the gap are slid as far left
          as possible, starting from index 0 and propagating rightward.
        - All repetitions to the RIGHT of the gap are slid as far right
          as possible, starting from the last index and propagating leftward.

    gap_idx follows the same convention as _gaps:
        gap 0 = before entry 0
        gap i = between entry i-1 and entry i
        gap n = after last entry

    Modifies solution in place.
    """
    entries      = solution.active[stock_id]
    n            = len(entries)
    stock_length = stocks[stock_id].length

    # cascade left: slide entries 0..gap_idx-1 as far left as possible
    for i in range(gap_idx):
        pat, eidx, start = entries[i]
        entry     = pat.stock_entries[stock_id][eidx]
        min_start = entries[i - 1][2] + entries[i - 1][0].length_consumed \
                    if i > 0 else 0.0
        new_start = _slide_left(entry, pat.length_consumed, start, min_start)
        if new_start is not None:
            entries[i] = (pat, eidx, new_start)

    # cascade right: slide entries gap_idx..n-1 as far right as possible
    for i in range(n - 1, gap_idx - 1, -1):
        pat, eidx, start = entries[i]
        entry   = pat.stock_entries[stock_id][eidx]
        max_end = entries[i + 1][2] if i + 1 < n else stock_length
        new_start = _slide_right(entry, pat.length_consumed, start, max_end)
        if new_start is not None:
            entries[i] = (pat, eidx, new_start)


# ---------------------------------------------------------------------------
# LocalSearch base class
# ---------------------------------------------------------------------------

class LocalSearch:

    def __init__(
        self,
        stocks: Dict,
        products: Dict,
        patterns: List[Pattern],
        max_iterations: int = 1000,
        verbose: bool = False
    ):
        self.stocks         = stocks
        self.products       = products
        self.patterns       = patterns
        self.max_iterations = max_iterations
        self.verbose        = verbose
        self.convergence_log      = []
        self.log_interval         = 0.1
        self.convergence_csv_path = None
        self.time_offset          = 0.0

        # patterns available per stock — precomputed for efficiency
        self.patterns_by_stock: Dict[str, List[Tuple[Pattern, int, PatternStockEntry]]] = {}
        for sid in stocks:
            self.patterns_by_stock[sid] = []
        for pattern in patterns:
            for stock_id, entries in pattern.stock_entries.items():
                for entry_idx, entry in enumerate(entries):
                    self.patterns_by_stock[stock_id].append((pattern, entry_idx, entry))

    def _random_active_stock(self, solution: Solution, min_reps: int = 1) -> Optional[str]:
        eligible = [
            sid for sid, entries in solution.active.items()
            if len(entries) >= min_reps
        ]
        return random.choice(eligible) if eligible else None
    
    def _log_convergence(self, elapsed, current_cost, greedy_ref=None):
        import csv, os
        adjusted_elapsed = elapsed + self.time_offset
        self.convergence_log.append((round(adjusted_elapsed, 2), current_cost))
        if self.convergence_csv_path is not None:
            impr = ((greedy_ref - current_cost) / greedy_ref * 100
                    if greedy_ref and greedy_ref > 0 else None)
            write_header = not os.path.exists(self.convergence_csv_path)
            with open(self.convergence_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['elapsed_sec', 'best_cost', 'improvement_pct'])
                writer.writerow([
                    round(adjusted_elapsed, 2),
                    round(current_cost, 4),
                    round(impr, 4) if impr is not None else ''
                ])

    def _fill_stock(self, solution: Solution, stock_id: str,
                    unmet_target: Dict) -> None:
        """
        Fill a stock greedily with patterns that cover unmet demand.
        Patterns are scored by unmet demand covered per unit cost.
        Modifies solution in place.
        """
        stock  = self.stocks[stock_id]
        cursor = 0.0

        # keep placing until no more useful (pattern, entry) fits
        while True:
            best_pattern   = None
            best_entry_idx = None
            best_entry     = None
            best_n_reps    = 0
            best_start_pos = 0.0
            best_ratio     = -1.0

            for pattern, entry_idx, entry in self.patterns_by_stock[stock_id]:
                n_needed = reps_needed(pattern, unmet_target)
                if n_needed == 0:
                    continue

                n_reps, start_pos, _ = placement_info(
                    entry, pattern.length_consumed, cursor, n_needed
                )
                if n_reps == 0:
                    continue

                unmet_covered = sum(
                    min(qty * n_reps, unmet_target.get(pid, 0))
                    for pid, qty in pattern.products_produced_per_rep.items()
                )
                if unmet_covered == 0:
                    continue

                total_cost = pattern.setup_cost + entry.cost_per_rep * n_reps
                if total_cost <= 0:
                    continue

                ratio = unmet_covered / total_cost
                if ratio > best_ratio:
                    best_ratio     = ratio
                    best_pattern   = pattern
                    best_entry_idx = entry_idx
                    best_entry     = entry
                    best_n_reps    = n_reps
                    best_start_pos = start_pos

            if best_pattern is None:
                break

            # place repetitions one by one
            pos    = best_start_pos
            placed = 0
            while placed < best_n_reps:
                next_pos = None
                for ws, we in best_entry.windows:
                    if pos > we:
                        continue
                    candidate = max(pos, ws)
                    if candidate + best_pattern.length_consumed <= we + 1e-9:
                        next_pos = candidate
                        break
                if next_pos is None:
                    break
                solution.add_repetition(stock_id, best_pattern, best_entry_idx, next_pos)
                pos    = next_pos + best_pattern.length_consumed
                placed += 1

            cursor = max(cursor, pos)

            for pid, qty in best_pattern.products_produced_per_rep.items():
                if pid in unmet_target:
                    unmet_target[pid] = max(0, unmet_target[pid] - qty * placed)

    def _fill_stock_randomized(self, solution: Solution, stock_id: str,
                                unmet_target: Dict,
                                perturb_scale: float = 0.2) -> None:
        """
        Same as _fill_stock but applies a random perturbation to each ratio,
        producing different fills on repeated calls. Used by stock_reset to
        explore different configurations of the same stock.
        """
        stock  = self.stocks[stock_id]
        cursor = 0.0

        while True:
            best_pattern   = None
            best_entry_idx = None
            best_entry     = None
            best_n_reps    = 0
            best_start_pos = 0.0
            best_ratio     = -1.0

            for pattern, entry_idx, entry in self.patterns_by_stock[stock_id]:
                n_needed = reps_needed(pattern, unmet_target)
                if n_needed == 0:
                    continue

                n_reps, start_pos, _ = placement_info(
                    entry, pattern.length_consumed, cursor, n_needed
                )
                if n_reps == 0:
                    continue

                unmet_covered = sum(
                    min(qty * n_reps, unmet_target.get(pid, 0))
                    for pid, qty in pattern.products_produced_per_rep.items()
                )
                if unmet_covered == 0:
                    continue

                total_cost = pattern.setup_cost + entry.cost_per_rep * n_reps
                if total_cost <= 0:
                    continue

                ratio = (unmet_covered / total_cost) * random.uniform(
                    1 - perturb_scale, 1 + perturb_scale
                )
                if ratio > best_ratio:
                    best_ratio     = ratio
                    best_pattern   = pattern
                    best_entry_idx = entry_idx
                    best_entry     = entry
                    best_n_reps    = n_reps
                    best_start_pos = start_pos

            if best_pattern is None:
                break

            pos    = best_start_pos
            placed = 0
            while placed < best_n_reps:
                next_pos = None
                for ws, we in best_entry.windows:
                    if pos > we:
                        continue
                    candidate = max(pos, ws)
                    if candidate + best_pattern.length_consumed <= we + 1e-9:
                        next_pos = candidate
                        break
                if next_pos is None:
                    break
                solution.add_repetition(stock_id, best_pattern, best_entry_idx, next_pos)
                pos    = next_pos + best_pattern.length_consumed
                placed += 1

            cursor = max(cursor, pos)

            for pid, qty in best_pattern.products_produced_per_rep.items():
                if pid in unmet_target:
                    unmet_target[pid] = max(0, unmet_target[pid] - qty * placed)

    # ---------------------------------------------------------------------------
    # Move samplers
    # ---------------------------------------------------------------------------

    def sample_insert_move(self, solution: Solution,
                           stock_id: str = None) -> Optional[tuple]:
        """
        Sample an insert move: pick a random (pattern, entry_idx) and find
        a feasible gap on a stock. Tries to slide adjacent repetitions to
        create space if no direct gap exists.
        """
        if stock_id is None:
            stock_id = self._random_active_stock(solution)
        if stock_id is None:
            return None

        stock   = self.stocks[stock_id]
        entries = solution.active[stock_id]

        # shuffle candidates for randomness
        candidates = list(self.patterns_by_stock[stock_id])
        random.shuffle(candidates)

        for pattern, entry_idx, entry in candidates:
            # try direct insertion first
            positions = feasible_insertions(pattern, entry_idx, stock_id, stock, entries)
            if positions:
                return ("insert", stock_id, pattern, entry_idx, random.choice(positions))

            # try with sliding — attempt each gap
            gaps = _gaps(entries, stock.length, pattern.length_consumed)
            for gap_i, (gs, ge) in enumerate(gaps):
                sol_copy = copy_solution(solution)
                _try_create_gap(sol_copy, stock_id, gap_i, self.stocks)
                new_entries  = sol_copy.active[stock_id]
                new_gaps     = _gaps(new_entries, stock.length, pattern.length_consumed)
                if gap_i < len(new_gaps):
                    new_gs, new_ge = new_gaps[gap_i]
                    pos = _first_feasible_in_gap(entry, pattern.length_consumed, new_gs, new_ge)
                    if pos is not None:
                        return ("insert_with_slide", stock_id, pattern, entry_idx,
                                pos, gap_i)

        return None



    def sample_remove_move(self, solution: Solution) -> Optional[tuple]:
        stock_id = self._random_active_stock(solution)
        if stock_id is None:
            return None
        entries = solution.active[stock_id]
        index   = random.randrange(len(entries))
        return ("remove", stock_id, index)

    def sample_shift_move(self, solution: Solution) -> Optional[tuple]:
        """
        Sample a shift move: pick a random repetition and move it to a
        different feasible position within the same gap.
        """
        stock_id = self._random_active_stock(solution)
        if stock_id is None:
            return None

        stock   = self.stocks[stock_id]
        entries = solution.active[stock_id]
        index   = random.randrange(len(entries))

        pat, eidx, start = entries[index]
        entry = pat.stock_entries[stock_id][eidx]

        left_end    = entries[index - 1][2] + entries[index - 1][0].length_consumed \
                      if index > 0 else 0.0
        right_start = entries[index + 1][2] if index + 1 < len(entries) else stock.length

        # find all feasible positions in this gap different from current
        feasible = []
        for ws, we in entry.windows:
            pos_start = max(left_end, ws)
            pos_end   = min(right_start, we) - pat.length_consumed
            if pos_start <= pos_end + 1e-9:
                # sample a few positions in this range
                step = max(1.0, (pos_end - pos_start) / 5)
                pos  = pos_start
                while pos <= pos_end + 1e-9:
                    if abs(pos - start) > 1e-6:
                        feasible.append(pos)
                    pos += step

        if not feasible:
            return None

        new_pos = random.choice(feasible)
        return ("shift", stock_id, index, new_pos)

    def sample_stock_open_move(self, solution: Solution,
                               unmet: Dict = None) -> Optional[tuple]:
        inactive = [sid for sid in self.stocks if sid not in solution.active]
        if not inactive:
            return None

        if unmet:
            useful = [
                sid for sid in inactive
                if any(
                    any(unmet.get(pid, 0) > 0 for pid in pat.products_produced_per_rep)
                    for pat, _, _ in self.patterns_by_stock[sid]
                )
            ]
            if useful:
                return ("stock_open", random.choice(useful))

        return ("stock_open", random.choice(inactive))

    def sample_stock_close_move(self, solution: Solution) -> Optional[tuple]:
        if not solution.active:
            return None
        return ("stock_close", random.choice(list(solution.active.keys())))

    def sample_close_open_move(self, solution: Solution) -> Optional[tuple]:
        if not solution.active:
            return None
        inactive = [sid for sid in self.stocks if sid not in solution.active]
        if not inactive:
            return None
        stock_id_to_close = random.choice(list(solution.active.keys()))
        stock_id_to_open  = random.choice(inactive)
        return ("close_open", stock_id_to_close, stock_id_to_open)



    def sample_swap_move(self, solution: Solution) -> Optional[tuple]:
        """Unfiltered — used by SA and repair."""
        stock_id = self._random_active_stock(solution)
        if stock_id is None:
            return None

        stock   = self.stocks[stock_id]
        entries = solution.active[stock_id]
        index   = random.randrange(len(entries))

        old_pat, old_eidx, old_start = entries[index]

        left_end    = entries[index - 1][2] + entries[index - 1][0].length_consumed \
                      if index > 0 else 0.0
        right_start = entries[index + 1][2] if index + 1 < len(entries) else stock.length
        gap_start   = left_end
        gap_end     = right_start

        candidates = [
            (pattern, entry_idx, entry)
            for pattern, entry_idx, entry in self.patterns_by_stock[stock_id]
            if not (pattern.pattern_id == old_pat.pattern_id and entry_idx == old_eidx)
        ]

        if not candidates:
            return None

        random.shuffle(candidates)

        for pattern, entry_idx, entry in candidates:
            pos = _first_feasible_in_gap(entry, pattern.length_consumed,
                                         gap_start, gap_end)
            if pos is not None:
                return ("swap", stock_id, index, pattern, entry_idx, pos)

            sol_copy = copy_solution(solution)
            sol_copy.remove_repetition(stock_id, index)
            _try_create_gap(sol_copy, stock_id, index, self.stocks)
            new_entries     = sol_copy.active.get(stock_id, [])
            new_left_end    = new_entries[index-1][2] + new_entries[index-1][0].length_consumed \
                              if index > 0 and index <= len(new_entries) else 0.0
            new_right_start = new_entries[index][2] if index < len(new_entries) else stock.length
            pos = _first_feasible_in_gap(entry, pattern.length_consumed,
                                         new_left_end, new_right_start)
            if pos is not None:
                return ("swap_with_slide", stock_id, index, pattern, entry_idx,
                        pos, index)

        return None

    def sample_swap_move_guided(self, solution: Solution,
                                current_unmet: Dict = None,
                                current_overproduced: Dict = None) -> Optional[tuple]:
        """Filtered — used by FI. Only considers candidates that reduce cost,
        reduce unmet, or increase overproduction."""
        stock_id = self._random_active_stock(solution)
        if stock_id is None:
            return None

        stock   = self.stocks[stock_id]
        entries = solution.active[stock_id]
        index   = random.randrange(len(entries))

        old_pat, old_eidx, old_start = entries[index]

        left_end    = entries[index - 1][2] + entries[index - 1][0].length_consumed \
                      if index > 0 else 0.0
        right_start = entries[index + 1][2] if index + 1 < len(entries) else stock.length
        gap_start   = left_end
        gap_end     = right_start

        eps = 1e-6

        if current_unmet is None or current_overproduced is None:
            placements, _ = decode(solution, self.stocks)
            _, current_unmet, current_overproduced = evaluate(
                solution, placements, self.stocks, self.products
            )

        current_cost_per_rep          = old_pat.stock_entries[stock_id][old_eidx].cost_per_rep
        current_unmet_contribution    = sum(
            qty for pid, qty in old_pat.products_produced_per_rep.items()
            if pid in current_unmet
        )
        current_overprod_contribution = sum(
            qty for pid, qty in old_pat.products_produced_per_rep.items()
            if pid in current_overproduced
        )

        candidates = [
            (pattern, entry_idx, entry)
            for pattern, entry_idx, entry in self.patterns_by_stock[stock_id]
            if not (pattern.pattern_id == old_pat.pattern_id and entry_idx == old_eidx)
            and (
                # reduces cost
                entry.cost_per_rep < current_cost_per_rep - eps
                or
                # reduces unmet
                sum(qty for pid, qty in pattern.products_produced_per_rep.items()
                    if pid in current_unmet) > current_unmet_contribution
                or
                # increases overproduction
                sum(qty for pid, qty in pattern.products_produced_per_rep.items()
                    if pid in current_overproduced) > current_overprod_contribution
            )
        ]

        if not candidates:
            return None

        random.shuffle(candidates)

        for pattern, entry_idx, entry in candidates:
            pos = _first_feasible_in_gap(entry, pattern.length_consumed,
                                         gap_start, gap_end)
            if pos is not None:
                return ("swap", stock_id, index, pattern, entry_idx, pos)

            sol_copy = copy_solution(solution)
            sol_copy.remove_repetition(stock_id, index)
            _try_create_gap(sol_copy, stock_id, index, self.stocks)
            new_entries     = sol_copy.active.get(stock_id, [])
            new_left_end    = new_entries[index-1][2] + new_entries[index-1][0].length_consumed \
                              if index > 0 and index <= len(new_entries) else 0.0
            new_right_start = new_entries[index][2] if index < len(new_entries) else stock.length
            pos = _first_feasible_in_gap(entry, pattern.length_consumed,
                                         new_left_end, new_right_start)
            if pos is not None:
                return ("swap_with_slide", stock_id, index, pattern, entry_idx,
                        pos, index)

        return None

    def sample_relocate_move(self, solution: Solution) -> Optional[tuple]:
        """Unfiltered — used by SA and repair."""
        if len(solution.active) < 1:
            return None

        stock_id_from = self._random_active_stock(solution)
        if stock_id_from is None:
            return None

        entries_from           = solution.active[stock_id_from]
        index                  = random.randrange(len(entries_from))
        pattern, entry_idx, start = entries_from[index]

        other_stocks = [sid for sid in self.stocks if sid != stock_id_from]
        if not other_stocks:
            return None
        random.shuffle(other_stocks)

        for stock_id_to in other_stocks:
            if stock_id_to not in pattern.stock_entries:
                continue

            stock_to   = self.stocks[stock_id_to]
            entries_to = solution.active.get(stock_id_to, [])

            for eidx_to, entry_to in enumerate(pattern.stock_entries[stock_id_to]):
                positions = feasible_insertions(
                    pattern, eidx_to, stock_id_to, stock_to, entries_to
                )
                if positions:
                    return ("relocate", stock_id_from, index,
                            stock_id_to, pattern, eidx_to, random.choice(positions))

                gaps = _gaps(entries_to, stock_to.length, pattern.length_consumed)
                for gap_i, (gs, ge) in enumerate(gaps):
                    sol_copy = copy_solution(solution)
                    if stock_id_to in sol_copy.active:
                        _try_create_gap(sol_copy, stock_id_to, gap_i, self.stocks)
                        new_entries = sol_copy.active[stock_id_to]
                    else:
                        new_entries = []
                    new_gaps = _gaps(new_entries, stock_to.length, pattern.length_consumed)
                    if gap_i < len(new_gaps):
                        new_gs, new_ge = new_gaps[gap_i]
                        pos = _first_feasible_in_gap(
                            entry_to, pattern.length_consumed, new_gs, new_ge
                        )
                        if pos is not None:
                            return ("relocate_with_slide", stock_id_from, index,
                                    stock_id_to, pattern, eidx_to, pos, gap_i)

        return None

    def sample_relocate_move_guided(self, solution: Solution,
                                    current_unmet: Dict = None,
                                    current_overproduced: Dict = None) -> Optional[tuple]:
        """Filtered — used by FI. Only considers target stocks where cost_delta < 0,
        accounting for activation costs."""
        if len(solution.active) < 1:
            return None

        stock_id_from = self._random_active_stock(solution)
        if stock_id_from is None:
            return None

        entries_from              = solution.active[stock_id_from]
        index                     = random.randrange(len(entries_from))
        pattern, entry_idx, start = entries_from[index]

        eps = 1e-6

        current_cost_per_rep = pattern.stock_entries[stock_id_from][entry_idx].cost_per_rep
        saves_activation     = self.stocks[stock_id_from].cost \
                               if len(entries_from) == 1 else 0.0

        other_stocks = [sid for sid in self.stocks if sid != stock_id_from]
        if not other_stocks:
            return None
        random.shuffle(other_stocks)

        for stock_id_to in other_stocks:
            if stock_id_to not in pattern.stock_entries:
                continue

            pays_activation = self.stocks[stock_id_to].cost \
                              if stock_id_to not in solution.active else 0.0
            stock_to        = self.stocks[stock_id_to]
            entries_to      = solution.active.get(stock_id_to, [])

            for eidx_to, entry_to in enumerate(pattern.stock_entries[stock_id_to]):
                cost_delta = (entry_to.cost_per_rep + pays_activation) \
                             - (current_cost_per_rep + saves_activation)
                if cost_delta >= -eps:
                    continue

                positions = feasible_insertions(
                    pattern, eidx_to, stock_id_to, stock_to, entries_to
                )
                if positions:
                    return ("relocate", stock_id_from, index,
                            stock_id_to, pattern, eidx_to, random.choice(positions))

                gaps = _gaps(entries_to, stock_to.length, pattern.length_consumed)
                for gap_i, (gs, ge) in enumerate(gaps):
                    sol_copy = copy_solution(solution)
                    if stock_id_to in sol_copy.active:
                        _try_create_gap(sol_copy, stock_id_to, gap_i, self.stocks)
                        new_entries = sol_copy.active[stock_id_to]
                    else:
                        new_entries = []
                    new_gaps = _gaps(new_entries, stock_to.length, pattern.length_consumed)
                    if gap_i < len(new_gaps):
                        new_gs, new_ge = new_gaps[gap_i]
                        pos = _first_feasible_in_gap(
                            entry_to, pattern.length_consumed, new_gs, new_ge
                        )
                        if pos is not None:
                            return ("relocate_with_slide", stock_id_from, index,
                                    stock_id_to, pattern, eidx_to, pos, gap_i)

        return None

    def sample_pattern_replace_all_move(self, solution: Solution) -> Optional[tuple]:
        """Unfiltered — used by SA and repair."""
        stock_id = self._random_active_stock(solution)
        if stock_id is None:
            return None

        entries       = solution.active[stock_id]
        used_patterns = list({pat.pattern_id: pat for pat, _, _ in entries}.values())
        if not used_patterns:
            return None
        old_pattern = random.choice(used_patterns)

        candidates = [
            pat for pat, _, _ in self.patterns_by_stock[stock_id]
            if pat.length_consumed == old_pattern.length_consumed
            and pat.pattern_id != old_pattern.pattern_id
        ]
        if not candidates:
            return None

        new_pattern = random.choice(candidates)
        return ("pattern_replace_all", stock_id, old_pattern, new_pattern)

    def sample_pattern_replace_all_move_guided(self, solution: Solution,
                                               current_unmet: Dict = None,
                                               current_overproduced: Dict = None) -> Optional[tuple]:
        """Filtered — used by FI. Only considers replacements that reduce cost,
        reduce unmet, or increase overproduction."""
        stock_id = self._random_active_stock(solution)
        if stock_id is None:
            return None

        entries       = solution.active[stock_id]
        used_patterns = list({pat.pattern_id: pat for pat, _, _ in entries}.values())
        if not used_patterns:
            return None
        old_pattern = random.choice(used_patterns)

        eps = 1e-6

        if current_unmet is None or current_overproduced is None:
            placements, _ = decode(solution, self.stocks)
            _, current_unmet, current_overproduced = evaluate(
                solution, placements, self.stocks, self.products
            )

        old_cost_per_rep          = min(e.cost_per_rep
                                        for e in old_pattern.stock_entries[stock_id])
        old_unmet_contribution    = sum(
            qty for pid, qty in old_pattern.products_produced_per_rep.items()
            if pid in current_unmet
        )
        old_overprod_contribution = sum(
            qty for pid, qty in old_pattern.products_produced_per_rep.items()
            if pid in current_overproduced
        )

        candidates = [
            pat for pat, _, _ in self.patterns_by_stock[stock_id]
            if pat.length_consumed == old_pattern.length_consumed
            and pat.pattern_id != old_pattern.pattern_id
            and (
                # reduces cost
                min(e.cost_per_rep for e in pat.stock_entries[stock_id]) < old_cost_per_rep - eps
                or
                # reduces unmet
                sum(qty for pid, qty in pat.products_produced_per_rep.items()
                    if pid in current_unmet) > old_unmet_contribution
                or
                # increases overproduction
                sum(qty for pid, qty in pat.products_produced_per_rep.items()
                    if pid in current_overproduced) > old_overprod_contribution
            )
        ]
        if not candidates:
            return None

        new_pattern = random.choice(candidates)
        return ("pattern_replace_all", stock_id, old_pattern, new_pattern)

    def sample_stock_reset_move(self, solution: Solution) -> Optional[tuple]:
        """
        Sample a stock reset move: pick a random active stock.
        The stock will be cleared and refilled with randomized greedy.
        """
        if not solution.active:
            return None
        return ("stock_reset", random.choice(list(solution.active.keys())))



    def sample_merge_stocks_move(self, solution: Solution) -> Optional[tuple]:
        """
        Sample a merge_stocks move: pick a donor stock and a compatible receiver.
        A receiver is compatible only if ALL patterns on the donor have a
        stock_entry on the receiver — otherwise merge_stocks will always return None.
        Prefers smaller stocks as donors to encourage consolidation.
        """
        if len(solution.active) < 2:
            return None

        active = list(solution.active.keys())

        # sort donors by number of repetitions ascending — prefer smaller stocks
        donors = sorted(active, key=lambda sid: len(solution.active[sid]))

        for donor in donors:
            donor_patterns = {pat.pattern_id: pat
                              for pat, _, _ in solution.active[donor]}

            # find receivers where all donor patterns have a stock entry
            compatible = [
                sid for sid in active
                if sid != donor
                and all(sid in pat.stock_entries
                        for pat in donor_patterns.values())
            ]
            if compatible:
                receiver = random.choice(compatible)
                return ("merge_stocks", donor, receiver)

        return None

    # ---------------------------------------------------------------------------
    # Apply move
    # ---------------------------------------------------------------------------

    def apply_move(self, solution: Solution, move: tuple,
                   inplace: bool = False) -> Solution:
        if not inplace:
            solution = copy_solution(solution)

        move_type = move[0]

        if move_type == "remove":
            _, stock_id, index = move
            solution.remove_repetition(stock_id, index)

        elif move_type == "insert":
            _, stock_id, pattern, entry_idx, start_pos = move
            solution.add_repetition(stock_id, pattern, entry_idx, start_pos)

        elif move_type == "insert_with_slide":
            _, stock_id, pattern, entry_idx, start_pos, gap_i = move
            _try_create_gap(solution, stock_id, gap_i, self.stocks)
            solution.add_repetition(stock_id, pattern, entry_idx, start_pos)

        elif move_type == "swap":
            _, stock_id, index, new_pattern, new_entry_idx, new_start_pos = move
            solution.replace_repetition(stock_id, index, new_pattern,
                                        new_entry_idx, new_start_pos)

        elif move_type == "swap_with_slide":
            _, stock_id, index, new_pattern, new_entry_idx, new_start_pos, gap_i = move
            solution.remove_repetition(stock_id, index)
            _try_create_gap(solution, stock_id, gap_i, self.stocks)
            solution.add_repetition(stock_id, new_pattern, new_entry_idx, new_start_pos)

        elif move_type == "shift":
            _, stock_id, index, new_start_pos = move
            pattern, entry_idx, _ = solution.get_repetition(stock_id, index)
            solution.replace_repetition(stock_id, index, pattern,
                                        entry_idx, new_start_pos)

        elif move_type == "stock_open":
            _, stock_id = move
            placements, _ = decode(solution, self.stocks)
            _, unmet_target, _ = evaluate(solution, placements,
                                          self.stocks, self.products)
            # if solution already feasible force meaningful fill
            # using full demand so stock always gets populated
            if sum(unmet_target.values()) == 0:
                unmet_target = {pid: p.demand
                                for pid, p in self.products.items()}
            solution.add_stock(stock_id)
            self._fill_stock(solution, stock_id, unmet_target)
            if not solution.active.get(stock_id):
                solution.remove_stock(stock_id)

        elif move_type == "stock_close":
            _, stock_id = move
            solution.remove_stock(stock_id)

        elif move_type == "close_open":
            _, stock_id_to_close, stock_id_to_open = move
            placements, _ = decode(solution, self.stocks)
            _, unmet_before, _ = evaluate(solution, placements,
                                          self.stocks, self.products)
            solution.remove_stock(stock_id_to_close)
            solution.add_stock(stock_id_to_open)
            self._fill_stock(solution, stock_id_to_open, unmet_before)
            if not solution.active.get(stock_id_to_open):
                solution.remove_stock(stock_id_to_open)

        elif move_type == "relocate":
            _, stock_id_from, index, stock_id_to, pattern, eidx_to, start_pos = move
            # add to destination first
            solution.add_repetition(stock_id_to, pattern, eidx_to, start_pos)
            # only remove from source if destination now has the repetition
            entries_to = solution.active.get(stock_id_to, [])
            if any(p.pattern_id == pattern.pattern_id and e == eidx_to 
                    for p, e, _ in entries_to):
                solution.remove_repetition(stock_id_from, index)
            else:
                return None  # placement failed — don't remove from source

        elif move_type == "relocate_with_slide":
            _, stock_id_from, index, stock_id_to, pattern, eidx_to, start_pos, gap_i = move
            # add to destination first
            if stock_id_to in solution.active:
                _try_create_gap(solution, stock_id_to, gap_i, self.stocks)
            solution.add_repetition(stock_id_to, pattern, eidx_to, start_pos)
            entries_to = solution.active.get(stock_id_to, [])
            if any(p.pattern_id == pattern.pattern_id and e == eidx_to
                    for p, e, _ in entries_to):
                solution.remove_repetition(stock_id_from, index)
            else:
                return None

        elif move_type == "stock_reset":
            _, stock_id = move

            # record current patterns before clearing
            previous = {
                (pat.pattern_id, eidx)
                for pat, eidx, _ in solution.active.get(stock_id, [])
            }

            # compute unmet after removing this stock
            sol_without = copy_solution(solution)
            sol_without.remove_stock(stock_id)
            placements, _ = decode(sol_without, self.stocks)
            _, unmet_after_remove, _ = evaluate(sol_without, placements,
                                                self.stocks, self.products)

            # clear and refill
            solution.remove_stock(stock_id)
            solution.add_stock(stock_id)
            self._fill_stock_randomized(solution, stock_id, unmet_after_remove)

            # if result is identical to previous configuration — treat as no-op
            new_patterns = {
                (pat.pattern_id, eidx)
                for pat, eidx, _ in solution.active.get(stock_id, [])
            }
            if new_patterns == previous:
                return None

            if not solution.active.get(stock_id):
                solution.remove_stock(stock_id)

        elif move_type == "pattern_replace_all":
            _, stock_id, old_pattern, new_pattern = move
            result = pattern_replace_all(
                solution, stock_id, old_pattern, new_pattern,
                self.stocks, inplace=True
            )
            if result is None:
                return None

        elif move_type == "merge_stocks":
            _, stock_id_donor, stock_id_receiver = move
            result = merge_stocks(
                solution, stock_id_donor, stock_id_receiver,
                self.stocks, inplace=True
            )
            if result is None:
                return None

        else:
            raise ValueError(f"Unknown move type: {move_type}")

        return solution

    # ---------------------------------------------------------------------------
    # Repair — Phase 1
    # ---------------------------------------------------------------------------

    def repair(self, solution: Solution,
               max_repair_iterations: int = 5000,
               time_limit: float = 60.0) -> Tuple[Solution, Dict]:
        """
        Phase 1 — Feasibility repair.
        Runs until all demand is met or stopping criteria reached.
        Accepts moves that strictly reduce total unmet demand.
        """
        move_counts = {}
        placements, _ = decode(solution, self.stocks)
        _, current_unmet, _ = evaluate(solution, placements,
                                       self.stocks, self.products)
        current_total_unmet = sum(current_unmet.values())

        if current_total_unmet == 0:
            return solution, move_counts

        print(f"  Repair start: unmet={current_unmet}")

        repair_samplers = {
            "stock_open"  : lambda s: self.sample_stock_open_move(s, unmet=current_unmet),
            "close_open"  : self.sample_close_open_move,
            "insert"      : self.sample_insert_move,
            "swap"        : self.sample_swap_move,
            "stock_reset" : self.sample_stock_reset_move,
            "relocate"     : self.sample_relocate_move
        }
        neighborhoods = list(repair_samplers.keys())
        weights       = [5, 1, 3, 1, 3, 1]

        start_time    = time.time()
        non_improving = 0

        while current_total_unmet > 0:
            if time.time() - start_time > time_limit:
                print(f"  Repair: time limit reached  unmet={current_unmet}")
                break
            if non_improving >= max_repair_iterations:
                print(f"  Repair: no improving move found  unmet={current_unmet}")
                break

            # disable stock_open if no inactive stocks remain
            inactive_exists = any(sid not in solution.active for sid in self.stocks)
            w = list(weights)
            if not inactive_exists:
                w[0] = 0

            neighborhood = random.choices(neighborhoods, weights=w, k=1)[0]
            move         = repair_samplers[neighborhood](solution)

            if move is None:
                non_improving += 1
                continue

            candidate = self.apply_move(solution, move, inplace=False)
            if candidate is None:
                non_improving += 1
                continue
            placements, fully_placed = decode(candidate, self.stocks)
            if not fully_placed:
                non_improving += 1
                continue

            _, candidate_unmet, _ = evaluate(candidate, placements,
                                             self.stocks, self.products)
            candidate_total_unmet = sum(candidate_unmet.values())

            if candidate_total_unmet < current_total_unmet:
                solution            = candidate
                current_unmet       = candidate_unmet
                current_total_unmet = candidate_total_unmet
                non_improving       = 0
                move_counts[move[0]] = move_counts.get(move[0], 0) + 1
                print(f"  Repair: unmet={current_unmet}  move={move[0]}")
            else:
                non_improving += 1

        if current_total_unmet == 0:
            print("  Repair: feasible solution found")

        return solution, move_counts

    def run(self, solution: Solution):
        raise NotImplementedError("Use SteepestImprovement or FirstImprovement")
    
    def _is_improving(self, current_cost, current_unmet, current_overproduced,
                    candidate_cost, candidate_unmet, candidate_overproduced,
                    eps=1e-6) -> bool:
        current_total_unmet   = sum(current_unmet.values())
        candidate_total_unmet = sum(candidate_unmet.values())

        # in _is_improving — add this as first check
        if candidate_total_unmet > current_total_unmet:
            return False  # never accept increased unmet

        if candidate_total_unmet < current_total_unmet:
            return True

        if candidate_total_unmet == current_total_unmet \
                and candidate_cost < current_cost - eps:
            return True

        if candidate_total_unmet == current_total_unmet \
                and abs(candidate_cost - current_cost) <= eps \
                and sum(candidate_overproduced.values()) > sum(current_overproduced.values()):
            return True

        return False

    def _setup_cost_for_entries(self, entries):
        total       = 0.0
        prev_pat_id = None
        prev_eidx   = None
        for pat, eidx, _ in entries:
            if pat.pattern_id != prev_pat_id or eidx != prev_eidx:
                total += pat.setup_cost
            prev_pat_id = pat.pattern_id
            prev_eidx   = eidx
        return total

    def _delta_relocate(self, solution, stock_id_from, index,
                        stock_id_to, eidx_to, new_pos):
        pat, eidx, _ = solution.active[stock_id_from][index]
        entries_from  = solution.active[stock_id_from]
        entries_to    = solution.active.get(stock_id_to, [])

        # source stock before
        old_rep_cost_from   = pat.stock_entries[stock_id_from][eidx].cost_per_rep
        old_setup_from      = self._setup_cost_for_entries(entries_from)
        old_activation_from = self.stocks[stock_id_from].cost

        # source stock after
        new_entries_from    = [e for i, e in enumerate(entries_from) if i != index]
        new_setup_from      = self._setup_cost_for_entries(new_entries_from)
        new_activation_from = self.stocks[stock_id_from].cost if new_entries_from else 0.0

        # target stock before
        old_setup_to        = self._setup_cost_for_entries(entries_to)
        old_activation_to   = self.stocks[stock_id_to].cost if entries_to else 0.0

        # target stock after
        new_entry           = (pat, eidx_to, new_pos)
        new_entries_to      = sorted(entries_to + [new_entry], key=lambda x: x[2])
        new_rep_cost_to     = pat.stock_entries[stock_id_to][eidx_to].cost_per_rep
        new_setup_to        = self._setup_cost_for_entries(new_entries_to)
        new_activation_to   = self.stocks[stock_id_to].cost

        # delta
        old_cost = (old_activation_from + old_rep_cost_from + old_setup_from +
                    old_activation_to   + old_setup_to)
        new_cost = (new_activation_from + new_setup_from +
                    new_activation_to   + new_rep_cost_to + new_setup_to)

        return new_cost - old_cost
    
    def _production_feasible(self, prod_delta, current_unmet):
        for pid, delta in prod_delta.items():
            if delta < 0 and current_unmet.get(pid, 0) > 0:
                return False
        return True

    def _delta_remove(self, solution, stock_id, index):
        pat, eidx, _ = solution.active[stock_id][index]
        entries       = solution.active[stock_id]

        old_rep_cost   = pat.stock_entries[stock_id][eidx].cost_per_rep
        old_setup      = self._setup_cost_for_entries(entries)
        old_activation = self.stocks[stock_id].cost

        new_entries    = [e for i, e in enumerate(entries) if i != index]
        new_setup      = self._setup_cost_for_entries(new_entries)
        new_activation = self.stocks[stock_id].cost if new_entries else 0.0

        cost_delta = (new_activation + new_setup) - \
                    (old_activation + old_rep_cost + old_setup)

        prod_delta = {
            pid: -qty
            for pid, qty in pat.products_produced_per_rep.items()
        }
        return cost_delta, prod_delta

    def _delta_swap(self, solution, stock_id, index, new_pattern, new_eidx, new_pos):
        pat, eidx, _  = solution.active[stock_id][index]
        entries        = solution.active[stock_id]

        old_setup      = self._setup_cost_for_entries(entries)
        old_rep_cost   = pat.stock_entries[stock_id][eidx].cost_per_rep

        new_entries    = list(entries)
        new_entries[index] = (new_pattern, new_eidx, new_pos)
        new_entries.sort(key=lambda x: x[2])  # re-sort by start position
        new_setup      = self._setup_cost_for_entries(new_entries)
        new_rep_cost   = new_pattern.stock_entries[stock_id][new_eidx].cost_per_rep

        cost_delta = (new_rep_cost + new_setup) - (old_rep_cost + old_setup)

        all_pids   = set(pat.products_produced_per_rep) | \
                 set(new_pattern.products_produced_per_rep)
        prod_delta = {
            pid: new_pattern.products_produced_per_rep.get(pid, 0) -
                pat.products_produced_per_rep.get(pid, 0)
            for pid in all_pids
            if new_pattern.products_produced_per_rep.get(pid, 0) !=
            pat.products_produced_per_rep.get(pid, 0)
        }
        return cost_delta, prod_delta

    def _delta_pattern_replace_all(self, solution, stock_id,
                                    old_pattern, new_pattern, new_eidx):
        entries  = solution.active[stock_id]
        n_reps   = sum(1 for pat, _, _ in entries
                    if pat.pattern_id == old_pattern.pattern_id)

        old_setup    = self._setup_cost_for_entries(entries)
        old_rep_cost = sum(
            pat.stock_entries[stock_id][eidx].cost_per_rep
            for pat, eidx, _ in entries
            if pat.pattern_id == old_pattern.pattern_id
        )

        new_entries  = [
            (new_pattern, new_eidx, start)
            if pat.pattern_id == old_pattern.pattern_id
            else (pat, eidx, start)
            for pat, eidx, start in entries
        ]
        new_setup    = self._setup_cost_for_entries(new_entries)
        new_rep_cost = new_pattern.stock_entries[stock_id][new_eidx].cost_per_rep * n_reps

        cost_delta = (new_rep_cost + new_setup) - (old_rep_cost + old_setup)

        all_pids   = set(old_pattern.products_produced_per_rep) | \
                    set(new_pattern.products_produced_per_rep)
        prod_delta = {
            pid: (new_pattern.products_produced_per_rep.get(pid, 0) -
                old_pattern.products_produced_per_rep.get(pid, 0)) * n_reps
            for pid in all_pids
            if new_pattern.products_produced_per_rep.get(pid, 0) !=
            old_pattern.products_produced_per_rep.get(pid, 0)
        }
        return cost_delta, prod_delta
    
    def _delta_insert(self, solution, stock_id, pattern, eidx, new_pos):
        entries        = solution.active.get(stock_id, [])

        # stock before
        old_setup      = self._setup_cost_for_entries(entries)
        old_activation = self.stocks[stock_id].cost if entries else 0.0

        # stock after
        new_entry      = (pattern, eidx, new_pos)
        new_entries    = sorted(entries + [new_entry], key=lambda x: x[2])
        new_setup      = self._setup_cost_for_entries(new_entries)
        new_rep_cost   = pattern.stock_entries[stock_id][eidx].cost_per_rep
        new_activation = self.stocks[stock_id].cost

        cost_delta = (new_activation + new_rep_cost + new_setup) - \
                    (old_activation + old_setup)

        prod_delta = {
            pid: qty
            for pid, qty in pattern.products_produced_per_rep.items()
        }
        return cost_delta, prod_delta


# ---------------------------------------------------------------------------
# SteepestImprovement
# ---------------------------------------------------------------------------

class SteepestImprovement(LocalSearch):
    """
    Evaluates all candidate moves and applies the best improving one.
    Phase 1: repair. Phase 2: steepest descent on cost.
    """

    def __init__(self, stocks, products, patterns,
                 max_iterations=100, time_limit=300.0, active_moves=None, verbose=False, seed= None):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.time_limit = time_limit
        self.active_moves = active_moves #None = use all moves
        self.seed= seed

    def _generate_improving_moves(self, solution: Solution,current_unmet, current_overproduced: Dict, active_moves: set = None, eps: float = 1e-6):
        """Generate only promising candidate moves for steepest descent."""
        inactive = [sid for sid in self.stocks if sid not in solution.active]
        active_moves = self.active_moves

        for stock_id, entries in solution.active.items():
            stock = self.stocks[stock_id]

            # remove moves
            if active_moves is None or "remove" in active_moves:
                for i in range(len(entries)):
                    pat, eidx, _ = entries[i]
                    # only yield if removing doesn't create unmet demand
                    removable = True
                    for pid, qty in pat.products_produced_per_rep.items():
                        if current_unmet.get(pid, 0) > 0:
                            removable = False
                            break
                        if current_overproduced.get(pid, 0) < qty:
                            removable = False
                            break
                    if removable:
                        yield ("remove", stock_id, i)

            # swap moves — only yield if reduces cost OR increases overproduction
            if active_moves is None or "swap" in active_moves:
                for i, (pat, eidx, start) in enumerate(entries):
                    left_end    = entries[i-1][2] + entries[i-1][0].length_consumed \
                                if i > 0 else 0.0
                    right_start = entries[i+1][2] if i+1 < len(entries) else stock.length
                    gap_start   = left_end
                    gap_end     = right_start

                    current_cost_per_rep = pat.stock_entries[stock_id][eidx].cost_per_rep
                    current_overprod_contribution = sum(
                        qty for pid, qty in pat.products_produced_per_rep.items()
                        if pid in current_overproduced
                    )

                    for pattern, entry_idx, entry in self.patterns_by_stock[stock_id]:
                        if pattern.pattern_id == pat.pattern_id and entry_idx == eidx:
                            continue

                        candidate_overprod_contribution = sum(
                            qty for pid, qty in pattern.products_produced_per_rep.items()
                            if pid in current_overproduced
                        )

                        reduces_cost       = entry.cost_per_rep < current_cost_per_rep - eps
                        increases_overprod = candidate_overprod_contribution > current_overprod_contribution

                        if not reduces_cost and not increases_overprod:
                            continue

                        # direct swap
                        pos = _first_feasible_in_gap(entry, pattern.length_consumed,
                                                    gap_start, gap_end)
                        if pos is not None:
                            yield ("swap", stock_id, i, pattern, entry_idx, pos)
                        else:
                            # swap with slide
                            sol_copy = copy_solution(solution)
                            sol_copy.remove_repetition(stock_id, i)
                            _try_create_gap(sol_copy, stock_id, i, self.stocks)
                            new_entries     = sol_copy.active.get(stock_id, [])
                            new_left_end    = new_entries[i-1][2] + new_entries[i-1][0].length_consumed \
                                if i > 0 and i <= len(new_entries) else 0.0
                            new_right_start = new_entries[i][2] if i < len(new_entries) else stock.length
                            pos = _first_feasible_in_gap(entry, pattern.length_consumed,
                                            new_left_end, new_right_start)
                            if pos is not None:
                                yield ("swap_with_slide", stock_id, i, pattern, entry_idx, pos, i)

                                            
            # relocate moves — only yield if cost delta is negative
            if active_moves is None or "relocate" in active_moves:
                for i, (pat, eidx, start) in enumerate(entries):
                    current_cost_per_rep = pat.stock_entries[stock_id][eidx].cost_per_rep
                    saves_activation     = self.stocks[stock_id].cost if len(entries) == 1 else 0.0

                    for stock_id_to in list(solution.active.keys()) + inactive:
                        if stock_id_to == stock_id:
                            continue
                        if stock_id_to not in pat.stock_entries:
                            continue

                        pays_activation = self.stocks[stock_id_to].cost \
                                        if stock_id_to not in solution.active else 0.0
                        entries_to      = list(solution.active.get(stock_id_to, []))
                        stock_to        = self.stocks[stock_id_to]

                        for eidx_to, entry_to in enumerate(pat.stock_entries[stock_id_to]):
                            cost_delta = (entry_to.cost_per_rep + pays_activation) \
                                        - (current_cost_per_rep + saves_activation)
                            if cost_delta >= -eps:
                                continue

                            positions = feasible_insertions(
                                pat, eidx_to, stock_id_to, stock_to, entries_to
                            )
                            if positions:
                                yield ("relocate", stock_id, i,
                                    stock_id_to, pat, eidx_to, positions[0])
                            if not positions:
                                # try with cascade sliding — save/restore instead of deepcopy
                                # original_entries_to = copy_solution(solution.active.get(stock_id_to, []))
                                gaps_to = _gaps(entries_to, stock_to.length,
                                            pat.length_consumed)
                                for gap_i, (gs, ge) in enumerate(gaps_to):
                                    original_entries_to = list(entries_to)
                                    if stock_id_to in solution.active:
                                        _try_create_gap(solution, stock_id_to,
                                                    gap_i, self.stocks)
                                    new_entries_to = solution.active.get(stock_id_to, [])
                                    new_gaps    = _gaps(new_entries_to, stock_to.length,
                                            pat.length_consumed)
                                    if gap_i < len(new_gaps):
                                        new_gs, new_ge = new_gaps[gap_i]
                                        slide_pos = _first_feasible_in_gap(
                                            entry_to, pat.length_consumed,
                                            new_gs, new_ge
                                        )
                                        if slide_pos is not None:
                                            yield ("relocate_with_slide", stock_id, i,
                                                stock_id_to, pat, eidx_to,
                                                slide_pos, gap_i)
                                    # restore original entries
                                    if stock_id_to in solution.active:
                                        solution.active[stock_id_to] = original_entries_to


            # stock_reset move
            if active_moves is None or "stock_reset" in active_moves:
                yield ("stock_reset", stock_id)

            # pattern_replace_all — replace all reps of one pattern with another
            if active_moves is None or "pattern_replace_all" in active_moves:
                entries = solution.active[stock_id]
                used_patterns = list({pat.pattern_id: pat for pat, _, _ in entries}.values())
                for old_pattern in used_patterns:
                    old_cost_per_rep = min(e.cost_per_rep for e in old_pattern.stock_entries[stock_id])
                    old_overprod_contribution = sum(
                        qty for pid, qty in old_pattern.products_produced_per_rep.items()
                        if pid in current_overproduced
                    )
                    for new_pat, _, _ in self.patterns_by_stock[stock_id]:
                        if new_pat.length_consumed != old_pattern.length_consumed:
                            continue
                        if new_pat.pattern_id == old_pattern.pattern_id:
                            continue

                        new_cost_per_rep = min(e.cost_per_rep for e in new_pat.stock_entries[stock_id])
                        new_overprod_contribution = sum(
                            qty for pid, qty in new_pat.products_produced_per_rep.items()
                            if pid in current_overproduced
                        )

                        reduces_cost       = new_cost_per_rep < old_cost_per_rep - eps
                        increases_overprod = new_overprod_contribution > old_overprod_contribution

                        if not reduces_cost and not increases_overprod:
                            continue

                        yield ("pattern_replace_all", stock_id, old_pattern, new_pat)

    def _generate_all_moves(self, solution: Solution, current_overproduced: Dict, eps: float = 1e-6):
        """Generate all candidate moves for steepest descent."""
        inactive = [sid for sid in self.stocks if sid not in solution.active]
       

        for stock_id, entries in solution.active.items():
            stock = self.stocks[stock_id]

            # remove moves
            for i in range(len(entries)):
                yield ("remove", stock_id, i)

            # insert moves — direct first, then with slide
            gaps = _gaps(entries, stock.length, 0.0)
            for gap_i, (gs, ge) in enumerate(gaps):
                for pattern, entry_idx, entry in self.patterns_by_stock[stock_id]:
                    pos = _first_feasible_in_gap(
                        entry, pattern.length_consumed, gs, ge
                    )
                    if pos is not None:
                        yield ("insert", stock_id, pattern, entry_idx, pos)
                    else:
                        # try with cascade sliding
                        sol_copy = copy_solution(solution)
                        _try_create_gap(sol_copy, stock_id, gap_i, self.stocks)
                        new_entries = sol_copy.active[stock_id]
                        new_gaps    = _gaps(new_entries, stock.length,
                                           pattern.length_consumed)
                        if gap_i < len(new_gaps):
                            new_gs, new_ge = new_gaps[gap_i]
                            slide_pos = _first_feasible_in_gap(
                                entry, pattern.length_consumed, new_gs, new_ge
                            )
                            if slide_pos is not None:
                                yield ("insert_with_slide", stock_id, pattern,
                                       entry_idx, slide_pos, gap_i)
            
            # swap moves — replace one repetition with a different (pattern, entry)
            for i, (pat, eidx, start) in enumerate(entries):
                left_end    = entries[i-1][2] + entries[i-1][0].length_consumed \
                            if i > 0 else 0.0
                right_start = entries[i+1][2] if i+1 < len(entries) else stock.length
                gap_start   = left_end
                gap_end     = right_start

                for pattern, entry_idx, entry in self.patterns_by_stock[stock_id]:
                    if pattern.pattern_id == pat.pattern_id and entry_idx == eidx:
                        continue

                    # direct swap
                    pos = _first_feasible_in_gap(entry, pattern.length_consumed,
                                                gap_start, gap_end)
                    if pos is not None:
                        yield ("swap", stock_id, i, pattern, entry_idx, pos)
                    else:
                        # swap with slide
                        sol_copy = copy_solution(solution)
                        sol_copy.remove_repetition(stock_id, i)
                        _try_create_gap(sol_copy, stock_id, i, self.stocks)
                        new_entries     = sol_copy.active.get(stock_id, [])
                        new_left_end    = new_entries[i-1][2] + new_entries[i-1][0].length_consumed \
                            if i > 0 and i <= len(new_entries) else 0.0
                        new_right_start = new_entries[i][2] if i < len(new_entries) else stock.length
                        pos = _first_feasible_in_gap(entry, pattern.length_consumed,
                                        new_left_end, new_right_start)
                        if pos is not None:
                            yield ("swap_with_slide", stock_id, i, pattern, entry_idx, pos, i)

            # shift moves — try multiple positions within the gap
            for i, (pat, eidx, start) in enumerate(entries):
                entry       = pat.stock_entries[stock_id][eidx]
                left_end    = entries[i-1][2] + entries[i-1][0].length_consumed \
                            if i > 0 else 0.0
                right_start = entries[i+1][2] if i+1 < len(entries) else stock.length
                for ws, we in entry.windows:
                    pos_start = max(left_end, ws)
                    pos_end   = min(right_start, we) - pat.length_consumed
                    if pos_start <= pos_end + 1e-9:
                        step = max(1.0, (pos_end - pos_start) / 5)
                        pos  = pos_start
                        while pos <= pos_end + 1e-9:
                            if abs(pos - start) > 1e-6:
                                yield ("shift", stock_id, i, pos)
                            pos += step

            # relocate moves — move one repetition to a different stock
            for i, (pat, eidx, start) in enumerate(entries):
                for stock_id_to in list(solution.active.keys()) + inactive:
                    if stock_id_to == stock_id:
                        continue
                    if stock_id_to not in pat.stock_entries:
                        continue
                    entries_to = solution.active.get(stock_id_to, [])
                    stock_to   = self.stocks[stock_id_to]
                    for eidx_to, entry_to in enumerate(pat.stock_entries[stock_id_to]):
                        positions = feasible_insertions(
                            pat, eidx_to, stock_id_to, stock_to, entries_to
                        )
                        for pos in positions:
                            yield ("relocate", stock_id, i,
                                   stock_id_to, pat, eidx_to, pos)
                        if not positions:
                            # try with cascade sliding
                            gaps_to = _gaps(entries_to, stock_to.length,
                                           pat.length_consumed)
                            for gap_i, (gs, ge) in enumerate(gaps_to):
                                sol_copy = copy_solution(solution)
                                if stock_id_to in sol_copy.active:
                                    _try_create_gap(sol_copy, stock_id_to,
                                                   gap_i, self.stocks)
                                new_entries = sol_copy.active.get(stock_id_to, [])
                                new_gaps    = _gaps(new_entries, stock_to.length,
                                                   pat.length_consumed)
                                if gap_i < len(new_gaps):
                                    new_gs, new_ge = new_gaps[gap_i]
                                    slide_pos = _first_feasible_in_gap(
                                        entry_to, pat.length_consumed,
                                        new_gs, new_ge
                                    )
                                    if slide_pos is not None:
                                        yield ("relocate_with_slide", stock_id, i,
                                               stock_id_to, pat, eidx_to,
                                               slide_pos, gap_i)

            # stock_reset move
            yield ("stock_reset", stock_id)

            # pattern_replace_all — replace all reps of one pattern with another
            entries = solution.active[stock_id]
            used_patterns = list({pat.pattern_id: pat for pat, _, _ in entries}.values())
            for old_pattern in used_patterns:
                for new_pat, _, _ in self.patterns_by_stock[stock_id]:
                    if (new_pat.length_consumed == old_pattern.length_consumed
                            and new_pat.pattern_id != old_pattern.pattern_id):
                        yield ("pattern_replace_all", stock_id, old_pattern, new_pat)

        # merge_stocks — move all reps from donor onto receiver and close donor
        active_list = list(solution.active.keys())
        for stock_id_donor in active_list:
            for stock_id_receiver in active_list:
                if stock_id_donor != stock_id_receiver:
                    yield ("merge_stocks", stock_id_donor, stock_id_receiver)

        # stock-level moves
        for stock_id in list(solution.active.keys()):
            yield ("stock_close", stock_id)
            for sid_open in inactive:
                yield ("close_open", stock_id, sid_open)

        for sid in inactive:
            yield ("stock_open", sid)

    def run(self, solution: Solution):
        solution, repair_counts = self.repair(solution)
        start_time = time.time()
        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )

        if self.verbose:
            print_solution_state(solution, self.stocks, self.products,
                                 label="After repair")

        improvement_counts = {}

        # move stats — track per move type across all iterations
        move_stats = {}   # move_type -> {generated, none, infeasible, best_selected}

        def _update_stats(move_type, key):
            if move_type not in move_stats:
                move_stats[move_type] = {
                    "generated": 0, "none": 0,
                    "infeasible": 0, "best_selected": 0, "improving": 0
                }
            move_stats[move_type][key] += 1
        final_iteration = 0
        stop_reason= "max_iterations"
        self.convergence_log = []
        _next_log = start_time + self.log_interval
        for iteration in range(self.max_iterations):
            _now = time.time()
            if _now - start_time > self.time_limit:
                print(f"  SD: time limit reached at iteration {iteration}")
                stop_reason="time_limit"
                break
            if _now >= _next_log:
                self._log_convergence(_now - start_time, current_cost)
                _next_log = _now + self.log_interval

            best_move            = None
            best_cost            = current_cost
            best_overproduced    = current_overproduced
            best_unmet           = current_unmet
            best_candidate_sol   = None  # save candidate for stock_reset
            eps                  = 1e-6

            for move in self._generate_improving_moves(solution, current_unmet, current_overproduced, self.active_moves, eps):
                if time.time() - start_time > self.time_limit:
                    print(f"  SD: time limit reached mid-iteration {iteration}")
                    break
                move_type = move[0]
                _update_stats(move_type, "generated")

                if move_type in ("relocate", "relocate_with_slide"):
                    if move_type == "relocate":
                        _, stock_id_from, index, stock_id_to, pat, eidx_to, pos = move
                    else:
                        _, stock_id_from, index, stock_id_to, pat, eidx_to, pos, gap_i = move
                    cost_delta = self._delta_relocate(solution, stock_id_from, index,
                                                    stock_id_to, eidx_to, pos)
                    if current_cost + cost_delta >= best_cost - eps:
                        continue
                    candidate_cost         = current_cost + cost_delta
                    candidate_unmet        = current_unmet
                    candidate_overproduced = current_overproduced
                    candidate              = None

                else:
                    if move_type == "remove":
                        _, stock_id, index = move
                        cost_delta, prod_delta = self._delta_remove(solution, stock_id, index)
                        if current_cost + cost_delta >= best_cost - eps:
                            continue
                        if not self._production_feasible(prod_delta, current_unmet):
                            _update_stats(move_type, "infeasible")
                            continue
                    elif move_type in ("swap", "swap_with_slide"):
                        if move_type == "swap":
                            _, stock_id, index, new_pattern, new_eidx, new_pos = move
                        else:
                            _, stock_id, index, new_pattern, new_eidx, new_pos, gap_i = move
                        cost_delta, _ = self._delta_swap(solution, stock_id, index,
                                                        new_pattern, new_eidx, new_pos)
                        if current_cost + cost_delta >= best_cost - eps:
                            continue
                    elif move_type == "pattern_replace_all":
                        _, stock_id, old_pattern, new_pattern = move
                        new_eidx = min(
                            range(len(new_pattern.stock_entries[stock_id])),
                            key=lambda i: new_pattern.stock_entries[stock_id][i].cost_per_rep
                        )
                        cost_delta, _ = self._delta_pattern_replace_all(
                            solution, stock_id, old_pattern, new_pattern, new_eidx
                        )
                        if current_cost + cost_delta >= best_cost - eps:
                            continue

                    candidate = self.apply_move(solution, move, inplace=False)
                    if candidate is None:
                        _update_stats(move_type, "none")
                        continue
                    placements, fully_placed = decode(candidate, self.stocks)
                    if not fully_placed:
                        _update_stats(move_type, "infeasible")
                        continue
                    candidate_cost, candidate_unmet, candidate_overproduced = evaluate(
                        candidate, placements, self.stocks, self.products
                    )
                    # reject if unmet increases
                    if sum(candidate_unmet.values()) > sum(current_unmet.values()):
                        _update_stats(move_type, "infeasible")
                        continue

                if self._is_improving(best_cost, best_unmet, best_overproduced,
                                    candidate_cost, candidate_unmet, candidate_overproduced):
                    _update_stats(move_type, "improving")
                    best_move          = move
                    best_cost          = candidate_cost
                    best_overproduced  = candidate_overproduced
                    best_unmet         = candidate_unmet
                    best_candidate_sol = candidate  # None for relocate, solution for others
                    if move_type == "stock_reset":
                        print(f"  DEBUG stock_reset selected: "
                            f"candidate_unmet={sum(candidate_unmet.values())}  "
                            f"current_unmet={sum(current_unmet.values())}  "
                            f"candidate_cost={candidate_cost:.4f}")

            if best_move is None:
                print(f"  Local optimum at iteration {iteration}")
                stop_reason="local_optimum"
                break

            _update_stats(best_move[0], "best_selected")

            if best_candidate_sol is not None:
                solution = best_candidate_sol
            else:
                # debug — capture state before applying relocate
                if best_move[0] in ("relocate", "relocate_with_slide"):
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos = best_move[:7]
                    reps_from_before = [(p.pattern_id, e, s)
                                    for p, e, s in solution.active.get(stock_id_from, [])]
                    reps_to_before   = [(p.pattern_id, e, s)
                                    for p, e, s in solution.active.get(stock_id_to, [])]

                # relocate and other deterministic moves — apply inplace
                result = self.apply_move(solution, best_move, inplace=True)
                if result is None:
                    print(f"  Warning: best_move {best_move[0]} returned None — skipping")
                    print(f"  WARNING: best_cost_selected={best_cost:.4f}  "
                        f"best_unmet_selected={sum(best_unmet.values())}  "
                        f"used_candidate={'yes' if best_candidate_sol is not None else 'no'}")
                    continue
                solution = result

                # verify production unchanged after relocate
                if best_move[0] in ("relocate", "relocate_with_slide"):
                    placements, _ = decode(solution, self.stocks)
                    actual_cost, actual_unmet, actual_overprod = evaluate(
                        solution, placements, self.stocks, self.products
                    )
                    if sum(actual_unmet.values()) != sum(current_unmet.values()):
                        _, stock_id_from, index, stock_id_to, pat, eidx_to, pos = best_move[:7]
                        reps_from_after = [(p.pattern_id, e, s)
                                        for p, e, s in solution.active.get(stock_id_from, [])]
                        reps_to_after   = [(p.pattern_id, e, s)
                                        for p, e, s in solution.active.get(stock_id_to, [])]
                        print(f"  RELOCATE PRODUCTION CHANGE:")
                        print(f"    unmet before={current_unmet}  after={actual_unmet}")
                        print(f"    S_from={stock_id_from} before={reps_from_before}")
                        print(f"    S_from={stock_id_from} after ={reps_from_after}")
                        print(f"    S_to  ={stock_id_to}   before={reps_to_before}")
                        print(f"    S_to  ={stock_id_to}   after ={reps_to_after}")
                        print(f"    pat={pat.pattern_id}  pos={pos}")
            placements, _ = decode(solution, self.stocks)
            current_cost, current_unmet, current_overproduced = evaluate(
                solution, placements, self.stocks, self.products
            )
            improvement_counts[best_move[0]] = improvement_counts.get(best_move[0], 0) + 1
            if sum(current_unmet.values()) > 0:
                print(f"  WARNING: unmet after {best_move[0]}: {current_unmet}")
            print(f"  Iter {iteration}: cost={current_cost:.4f}  move={best_move[0]}")

            if self.verbose:
                print_solution_state(solution, self.stocks, self.products,
                                     label=f"After {best_move[0]}")
            final_iteration = iteration

        move_stats["_meta"] = {
            "final_iteration" : final_iteration,
            "stop_reason"     : stop_reason,
            "elapsed_sec"     : round(time.time() - start_time, 2),
        }

        print(f"\n=== MOVE STATISTICS (SD) ===")
        for move_type, stats in sorted(move_stats.items()):
            if move_type == "_meta":   # ← add this
                continue
            total = stats['generated']
            sel   = stats['best_selected']
            sel_rate = sel / total * 100 if total > 0 else 0
            print(f"  {move_type:25s}: generated={total:6d}  "
                  f"none={stats['none']:5d}  "
                  f"infeasible={stats['infeasible']:5d}  "
                  f"improving={stats['improving']:5d}  "
                  f"best_selected={sel:4d}  "
                  f"selection_rate={sel_rate:.2f}%")

        return solution, repair_counts, improvement_counts, move_stats, final_iteration
    

# ---------------------------------------------------------------------------
# FirstImprovement
# ---------------------------------------------------------------------------

class FirstImprovement(LocalSearch):
    """
    Randomly samples moves and applies the first improving one found.
    Phase 1: repair. Phase 2: first improvement on cost.
    """

    def __init__(self, stocks, products, patterns,
                 max_iterations=1000, time_limit=60.0,
                 neighborhood_weights: Dict[str, float] = None,
                 verbose=False, 
                 seed= None):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.time_limit = time_limit
        self.seed= seed


        default_weights = {
            "remove"             : 1.0,
            "swap"               : 2.0,
            "relocate"           : 4.0,
            "stock_reset"        : 0.5,
            "pattern_replace_all": 0.5,
        }
        if neighborhood_weights:
            # only use neighborhoods explicitly defined in the custom weights
            active_weights = neighborhood_weights
        else:
            active_weights = default_weights

        self.neighborhoods        = list(active_weights.keys())
        self.neighborhood_weights = list(active_weights.values())

    def run(self, solution: Solution):
        if self.seed is not None:
            random.seed(self.seed)
        solution, repair_counts = self.repair(solution)

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )

        if self.verbose:
            print_solution_state(solution, self.stocks, self.products,
                                 label="After repair")

        samplers = {
            "remove"              : self.sample_remove_move,
            #"insert"              : self.sample_insert_move,
            "swap"                : lambda s: self.sample_swap_move_guided(s, current_unmet, current_overproduced),
            #"shift"               : self.sample_shift_move,
            #"stock_open"          : self.sample_stock_open_move,
            #"stock_close"         : self.sample_stock_close_move,
            #"close_open"          : self.sample_close_open_move,
            "relocate"            : lambda s: self.sample_relocate_move_guided(s, current_unmet, current_overproduced),
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : lambda s: self.sample_pattern_replace_all_move_guided(s, current_unmet, current_overproduced),
            #"merge_stocks"        : self.sample_merge_stocks_move,
        }

        move_stats = {
            n: {"sampled": 0, "improving": 0, "none": 0, "infeasible": 0}
            for n in self.neighborhoods
        }

        start_time    = time.time()
        self.convergence_log = []
        _next_log     = start_time + self.log_interval
        non_improving = 0
        eps           = 1e-6

        while non_improving < self.max_iterations:
            _now = time.time()
            if _now - start_time > self.time_limit:
                print("  Time limit reached")
                break
            if _now >= _next_log:
                self._log_convergence(_now - start_time, current_cost)
                _next_log = _now + self.log_interval

            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)

            if move is None:
                move_stats[neighborhood]["none"] += 1
                continue

            move_stats[neighborhood]["sampled"] += 1
           
            move_type = move[0]

            if move_type in ("relocate", "relocate_with_slide"):
                if move_type == "relocate":
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos = move
                else:
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos, gap_i = move
                cost_delta = self._delta_relocate(solution, stock_id_from, index,
                                                stock_id_to, eidx_to, pos)
                if current_cost + cost_delta >= current_cost - eps:
                    non_improving += 1
                    continue
                candidate_cost         = current_cost + cost_delta
                candidate_unmet        = current_unmet
                candidate_overproduced = current_overproduced
                candidate              = None

            else:
                # all other moves — delta pre-screen if available, then full eval
                if move_type == "remove":
                    _, stock_id, index = move
                    cost_delta, _ = self._delta_remove(solution, stock_id, index)
                    if current_cost + cost_delta >= current_cost - eps:
                        non_improving += 1
                        continue
                elif move_type in ("swap", "swap_with_slide"):
                    if move_type == "swap":
                        _, stock_id, index, new_pattern, new_eidx, new_pos = move
                    else:
                        _, stock_id, index, new_pattern, new_eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_swap(solution, stock_id, index,
                                          new_pattern, new_eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        non_improving += 1
                        continue
                elif move_type == "pattern_replace_all":
                    _, stock_id, old_pattern, new_pattern = move
                    new_eidx = min(
                        range(len(new_pattern.stock_entries[stock_id])),
                        key=lambda i: new_pattern.stock_entries[stock_id][i].cost_per_rep
                    )
                    cost_delta, _ = self._delta_pattern_replace_all(
                        solution, stock_id, old_pattern, new_pattern, new_eidx
                    )
                    if current_cost + cost_delta >= current_cost - eps:
                        non_improving += 1
                        continue
                elif move_type in ("insert", "insert_with_slide"):
                    if move_type == "insert":
                        _, stock_id, pattern, eidx, new_pos = move
                    else:
                        _, stock_id, pattern, eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_insert(solution, stock_id, pattern, eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        non_improving += 1
                        continue

                # full eval for all non-relocate moves that pass delta pre-screen
                candidate = self.apply_move(solution, move, inplace=False)
                if candidate is None:
                    move_stats[neighborhood]["infeasible"] += 1
                    non_improving += 1
                    continue
                placements, fully_placed = decode(candidate, self.stocks)
                if not fully_placed:
                    move_stats[neighborhood]["infeasible"] += 1
                    non_improving += 1
                    continue
                candidate_cost, candidate_unmet, candidate_overproduced = evaluate(
                    candidate, placements, self.stocks, self.products
                )

            if self._is_improving(current_cost, current_unmet, current_overproduced,
                                candidate_cost, candidate_unmet, candidate_overproduced):
                if candidate is None:
                    result = self.apply_move(solution, move, inplace=True)
                    if result is None:
                        non_improving += 1
                        continue
                    solution = result
                else:
                    solution = candidate
                placements, _ = decode(solution, self.stocks)
                current_cost, current_unmet, current_overproduced = evaluate(
                    solution, placements, self.stocks, self.products
                )

                if sum(current_unmet.values()) > 0:
                    print(f"  WARNING: unmet after {move[0]}: {current_unmet}")

                non_improving = 0
                move_stats[neighborhood]["improving"] += 1
                print(f"  Improvement: cost={current_cost:.4f}  move={move[0]}")
                if self.verbose:
                    print_solution_state(solution, self.stocks, self.products,
                                        label=f"After {move[0]}")
            else:
                non_improving += 1

        print(f"  Stopped after {non_improving} non-improving attempts")
        print("\n=== MOVE STATISTICS ===")
        for n, stats in move_stats.items():
            hit_rate = (stats['improving'] / stats['sampled'] * 100
                        if stats['sampled'] > 0 else 0)
            print(f"  {n:15s}: sampled={stats['sampled']:5d}  "
                  f"improving={stats['improving']:4d}  "
                  f"infeasible={stats['infeasible']:4d}  "
                  f"none={stats['none']:5d}  "
                  f"hit_rate={hit_rate:.1f}%")

        improvement_counts = {
            n: s["improving"] for n, s in move_stats.items() if s["improving"] > 0
        }

        

        return solution, repair_counts, improvement_counts, move_stats


# ---------------------------------------------------------------------------
# Tabu Search
# ---------------------------------------------------------------------------

class TabuSearch(LocalSearch):
    """
    Tabu Search with active_moves filtering.
    Uses the same exhaustive move generation as SD but accepts
    non-improving moves, preventing cycling via a tabu list.
    """

    def __init__(self, stocks, products, patterns,
                 tabu_tenure    : int   = 7,
                 max_iterations : int   = 9999,
                 time_limit     : float = 60.0,
                 active_moves   : set   = None,
                 seed           : int   = None,
                 verbose        : bool  = False):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.tabu_tenure  = tabu_tenure
        self.time_limit   = time_limit
        self.active_moves = active_moves if active_moves is not None \
                            else {"relocate", "stock_reset"}
        self.seed= seed

    def _generate_ts_moves(self, solution: Solution):
        """
        Generate all candidate moves filtered by active_moves.
        Reuses the same move generation logic as SD but only yields
        moves whose type is in self.active_moves.
        """
        inactive = [sid for sid in self.stocks
                    if sid not in solution.active]
        eps = 1e-6

        placements, _ = decode(solution, self.stocks)
        _, _, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )

        for stock_id, entries in solution.active.items():
            stock = self.stocks[stock_id]

            if "remove" in self.active_moves:
                for i in range(len(entries)):
                    yield ("remove", stock_id, i)

            if "swap" in self.active_moves:
                for i, (pat, eidx, start) in enumerate(entries):
                    left_end    = entries[i-1][2] + entries[i-1][0].length_consumed \
                                  if i > 0 else 0.0
                    right_start = entries[i+1][2] \
                                  if i+1 < len(entries) else stock.length
                    gap_start   = left_end
                    gap_end     = right_start
                    current_cost_per_rep = pat.stock_entries[stock_id][eidx].cost_per_rep
                    for pattern, entry_idx, entry in self.patterns_by_stock[stock_id]:
                        if pattern.pattern_id == pat.pattern_id and entry_idx == eidx:
                            continue
                        pos = _first_feasible_in_gap(
                            entry, pattern.length_consumed, gap_start, gap_end
                        )
                        if pos is not None:
                            yield ("swap", stock_id, i, pattern, entry_idx, pos)

            if "relocate" in self.active_moves:
                for i, (pat, eidx, start) in enumerate(entries):
                    current_cost_per_rep = pat.stock_entries[stock_id][eidx].cost_per_rep
                    saves_activation     = self.stocks[stock_id].cost \
                                          if len(entries) == 1 else 0.0
                    for stock_id_to in list(solution.active.keys()) + inactive:
                        if stock_id_to == stock_id:
                            continue
                        if stock_id_to not in pat.stock_entries:
                            continue
                        pays_activation = self.stocks[stock_id_to].cost \
                                        if stock_id_to not in solution.active else 0.0
                        entries_to = list(solution.active.get(stock_id_to, []))
                        stock_to   = self.stocks[stock_id_to]
                        for eidx_to, entry_to in enumerate(
                            pat.stock_entries[stock_id_to]
                        ):
                            # no cost_delta filter — TS considers all moves
                            positions = feasible_insertions(
                                pat, eidx_to, stock_id_to, stock_to, entries_to
                            )
                            if positions:
                                yield ("relocate", stock_id, i,
                                       stock_id_to, pat, eidx_to, positions[0])
                            if not positions:
                                # cascade sliding — same as SD
                                gaps_to = _gaps(entries_to, stock_to.length,
                                            pat.length_consumed)
                                for gap_i, (gs, ge) in enumerate(gaps_to):
                                    original_entries_to = list(entries_to)
                                    if stock_id_to in solution.active:
                                        _try_create_gap(solution, stock_id_to,
                                                    gap_i, self.stocks)
                                    new_entries_to = solution.active.get(stock_id_to, [])
                                    new_gaps = _gaps(new_entries_to, stock_to.length,
                                            pat.length_consumed)
                                    if gap_i < len(new_gaps):
                                        new_gs, new_ge = new_gaps[gap_i]
                                        slide_pos = _first_feasible_in_gap(
                                            entry_to, pat.length_consumed,
                                            new_gs, new_ge
                                        )
                                        if slide_pos is not None:
                                            yield ("relocate_with_slide", stock_id, i,
                                                stock_id_to, pat, eidx_to,
                                                slide_pos, gap_i)
                                    if stock_id_to in solution.active:
                                        solution.active[stock_id_to] = original_entries_to

            if "stock_reset" in self.active_moves:
                yield ("stock_reset", stock_id)

            if "pattern_replace_all" in self.active_moves:
                used_patterns = list(
                    {pat.pattern_id: pat for pat, _, _ in entries}.values()
                )
                for old_pattern in used_patterns:
                    old_cpr = min(e.cost_per_rep for e in
                                  old_pattern.stock_entries[stock_id])
                    for new_pat, _, _ in self.patterns_by_stock[stock_id]:
                        if new_pat.length_consumed != old_pattern.length_consumed:
                            continue
                        if new_pat.pattern_id == old_pattern.pattern_id:
                            continue
                        yield ("pattern_replace_all", stock_id,
                                   old_pattern, new_pat)

    def _reverse_attributes(self, move: tuple, solution: Solution) -> tuple:
        move_type = move[0]
        if move_type in ("relocate", "relocate_with_slide"):
            _, stock_from, index, stock_to, pattern, eidx_to, *_ = move
            return ('relocate', stock_to, stock_from, pattern.pattern_id)
        elif move_type in ("swap", "swap_with_slide"):
            _, stock_id, index, new_pattern, new_eidx, *_ = move
            old_pat, old_eidx, _ = solution.active[stock_id][index]
            return ('swap', stock_id, new_pattern.pattern_id, old_pat.pattern_id)
        elif move_type == "remove":
            _, stock_id, index = move
            pat, eidx, _ = solution.active[stock_id][index]
            return ('insert', stock_id, pat.pattern_id)
        elif move_type == "stock_reset":
            _, stock_id = move
            return ('stock_reset', stock_id)
        elif move_type == "pattern_replace_all":
            _, stock_id, old_pat, new_pat = move
            return ('pattern_replace_all', stock_id,
                    new_pat.pattern_id, old_pat.pattern_id)
        else:
            return (move_type,)

    def run(self, solution: Solution):
        if self.seed is not None:
            random.seed(self.seed)
        solution, repair_counts = self.repair(solution)

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, _ = evaluate(
            solution, placements, self.stocks, self.products
        )
        """
        if sum(current_unmet.values()) > 0:
            print(f"  TS: repair failed — unmet={current_unmet}")
            move_stats = {"_meta": {"final_iteration": 0, 
                            "stop_reason": "repair_failed",
                            "elapsed_sec": 0.0}}
            return solution, repair_counts, {}, move_stats, 0
        """
        
        best_solution = copy_solution(solution)
        best_cost     = current_cost

        print(f"  TS start: cost={current_cost:.4f}  "
              f"tenure={self.tabu_tenure}  max_iter={self.max_iterations}  "
              f"active_moves={self.active_moves}")

        from collections import deque
        tabu_list          = deque(maxlen=self.tabu_tenure)
        improvement_counts = {}
        aspiration_count   = 0
        start_time         = time.time()
        eps                = 1e-6

        final_iteration = 0
        stop_reason     = "max_iterations"
        self.convergence_log = []
        _next_log = start_time + self.log_interval

        for iteration in range(self.max_iterations):
            _now = time.time()
            if _now - start_time > self.time_limit:
                print(f"  TS: time limit reached at iteration {iteration}")
                stop_reason = "time_limit"
                break
            if _now >= _next_log:
                self._log_convergence(_now - start_time, current_cost)
                _next_log = _now + self.log_interval

            best_move      = None
            best_candidate = None
            best_move_cost = float('inf')
            best_candidate_unmet = {}
            best_aspirated = False

            for move in self._generate_ts_moves(solution):
                if time.time() - start_time > self.time_limit:  
                    break
                reverse = self._reverse_attributes(move, solution)
                is_tabu = reverse in tabu_list

                candidate = self.apply_move(solution, move, inplace=False)
                if candidate is None:
                    continue

                placements, fully_placed = decode(candidate, self.stocks)
                if not fully_placed:
                    continue

                candidate_cost, candidate_unmet, _ = evaluate(
                    candidate, placements, self.stocks, self.products
                )

                if sum(candidate_unmet.values()) > sum(current_unmet.values()):
                    continue

                aspirated = candidate_cost < best_cost - eps

                if is_tabu and not aspirated:
                    continue

                if candidate_cost < best_move_cost:
                    best_move_cost = candidate_cost
                    best_move      = move
                    best_candidate = candidate
                    best_candidate_unmet= candidate_unmet
                    best_aspirated = aspirated

            if best_move is None:
                print(f"  TS: no valid move at iteration {iteration} — stopping")
                stop_reason = "no_valid_move"
                break

            # apply best move
            new_solution = best_candidate
            if new_solution is None:
                if self.verbose:
                    print(f"  TS [{iteration}]: apply_move returned None — skipping")
                reverse = self._reverse_attributes(best_move, solution)
                tabu_list.append(reverse)
                continue
            reverse = self._reverse_attributes(best_move, solution)  
            tabu_list.append(reverse)  
            solution     = new_solution
            current_cost = best_move_cost
            candidate_unmet= best_candidate_unmet

            if current_cost < best_cost - eps:
                best_cost     = current_cost
                best_solution = copy_solution(solution)
                improvement_counts[best_move[0]] = \
                    improvement_counts.get(best_move[0], 0) + 1
                aspiration_count += int(best_aspirated)
                if self.verbose:
                    print(f"  TS [{iteration}]: new best  "
                          f"cost={best_cost:.4f}  move={best_move[0]}  "
                          f"aspirated={best_aspirated}")
            else:
                if self.verbose and iteration % 10 == 0:
                    print(f"  TS [{iteration}]: cost={current_cost:.4f}  "
                          f"best={best_cost:.4f}")
                    
            final_iteration= iteration
        
        move_stats = {
            "_meta": {
                "final_iteration" : final_iteration,
                "stop_reason"     : stop_reason,
                "elapsed_sec"     : round(time.time() - start_time, 2),
            }
        }

        print(f"  TS finished: best={best_cost:.4f}  "
              f"iterations={iteration+1}  aspirations={aspiration_count}")
        print(f"  Improving moves: {improvement_counts}")

        return best_solution, repair_counts, improvement_counts, move_stats, final_iteration


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

class SimulatedAnnealing(LocalSearch):
        
    def __init__(self, stocks, products, patterns,
             T_init          : float = None,
             T_min           : float = 1e-3,
             alpha           : float = 0.995,
             max_iterations  : int   = 10000,
             time_limit      : float = 60.0,
             neighborhood_weights: Dict[str, float] = None,
             verbose         : bool  = False, 
             seed=None):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.T_init     = T_init
        self.T_min      = T_min
        self.alpha      = alpha
        self.time_limit = time_limit
        self.seed = seed

        default_weights = {
            "remove"              : 1.0,
            "swap"                : 1.0,
            "relocate"            : 1.0,
            "stock_reset"         : 1.0,
            "pattern_replace_all" : 1.0,
            "insert"              : 1.0,
            "stock_open"          : 1.0,
            "close_open"          : 1.0,
        }
        if neighborhood_weights:
            active_weights = neighborhood_weights
        else:
            active_weights = default_weights

        self.neighborhoods        = list(active_weights.keys())
        self.neighborhood_weights = list(active_weights.values())

    def _initial_temperature(self, solution: Solution) -> float:
        """
        Estimate initial temperature by sampling random moves and computing
        the average cost increase of worsening moves.
        T_init is set so that a worsening move of average size is accepted
        with probability ~0.8 at the start.
        """
        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }

        placements, _ = decode(solution, self.stocks)
        current_cost, _, _ = evaluate(solution, placements, self.stocks, self.products)

        deltas = []
        for _ in range(50):
            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)
            if move is None:
                continue
            candidate = self.apply_move(solution, move, inplace=False)
            if candidate is None:
                continue
            placements, fully_placed = decode(candidate, self.stocks)
            if not fully_placed:
                continue
            candidate_cost, _, _ = evaluate(candidate, placements,
                                            self.stocks, self.products)
            delta = candidate_cost - current_cost
            if delta > 0:
                deltas.append(delta)

        if not deltas:
            return 100.0

        avg_delta = sum(deltas) / len(deltas)
        T = -avg_delta / math.log(0.8)
        return max(T, 1.0)

    def run(self, solution: Solution):
        if self.seed is not None:
            random.seed(self.seed)
        # --- Phase 1: repair ---
        solution, repair_counts = self.repair(solution)

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )
        """
        if sum(current_unmet.values()) > 0:
            print(f"  SA: repair failed — unmet={current_unmet}")
            return solution, repair_counts, {}, {}
        """
        if self.verbose:
            print_solution_state(solution, self.stocks, self.products,
                             label="After repair")

        # --- Phase 2: SA ---
        T = self.T_init if self.T_init is not None \
            else self._initial_temperature(solution)

        print(f"  SA start: cost={current_cost:.4f}  T_init={T:.4f}  "
            f"alpha={self.alpha}  T_min={self.T_min}")

        best_solution = copy_solution(solution)
        best_cost     = current_cost

        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }

        move_stats = {
            n: {"sampled": 0, "improving": 0, "accepted_worse": 0,
                "infeasible": 0, "none": 0, "rejected": 0}
            for n in self.neighborhoods
        }

        accepted_worse = 0
        rejected       = 0
        start_time     = time.time()
        self.convergence_log = []
        _next_log      = start_time + self.log_interval
        iteration      = 0
        eps            = 1e-6

        while T > self.T_min and iteration < self.max_iterations:
            _now = time.time()
            if _now - start_time > self.time_limit:
                print(f"  SA: time limit reached at iteration {iteration}")
                break
            if _now >= _next_log:
                self._log_convergence(_now - start_time, current_cost)
                _next_log = _now + self.log_interval


            # sample a random move
            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)

            if move is None:
                move_stats[neighborhood]["none"] += 1
                continue

            move_stats[neighborhood]["sampled"] += 1

            move_type = move[0]

            if move_type in ("relocate", "relocate_with_slide"):
                if move_type == "relocate":
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos = move
                else:
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos, gap_i = move
                cost_delta             = self._delta_relocate(solution, stock_id_from, index,
                                                            stock_id_to, eidx_to, pos)
                candidate_cost         = current_cost + cost_delta
                candidate_unmet        = current_unmet
                candidate_overproduced = current_overproduced
                candidate              = None

            else:
                if move_type == "remove":
                    _, stock_id, index = move
                    cost_delta, _ = self._delta_remove(solution, stock_id, index)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= self.alpha
                        continue
                elif move_type in ("swap", "swap_with_slide"):
                    if move_type == "swap":
                        _, stock_id, index, new_pattern, new_eidx, new_pos = move
                    else:
                        _, stock_id, index, new_pattern, new_eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_swap(solution, stock_id, index,
                                                    new_pattern, new_eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= self.alpha
                        continue
                elif move_type == "pattern_replace_all":
                    _, stock_id, old_pattern, new_pattern = move
                    new_eidx = min(
                        range(len(new_pattern.stock_entries[stock_id])),
                        key=lambda i: new_pattern.stock_entries[stock_id][i].cost_per_rep
                    )
                    cost_delta, _ = self._delta_pattern_replace_all(
                        solution, stock_id, old_pattern, new_pattern, new_eidx
                    )
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= self.alpha
                        continue
                elif move_type in ("insert", "insert_with_slide"):
                    if move_type == "insert":
                        _, stock_id, pattern, eidx, new_pos = move
                    else:
                        _, stock_id, pattern, eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_insert(solution, stock_id, pattern, eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= self.alpha
                        continue    
                    

                candidate = self.apply_move(solution, move, inplace=False)
                if candidate is None:
                    move_stats[neighborhood]["infeasible"] += 1
                    iteration += 1
                    T *= self.alpha
                    continue
                placements, fully_placed = decode(candidate, self.stocks)
                if not fully_placed:
                    move_stats[neighborhood]["infeasible"] += 1
                    iteration += 1
                    T *= self.alpha
                    continue
                candidate_cost, candidate_unmet, candidate_overproduced = evaluate(
                    candidate, placements, self.stocks, self.products
                )

            current_total_unmet   = sum(current_unmet.values())
            candidate_total_unmet = sum(candidate_unmet.values())
        
            # --- acceptance criterion ---
            if candidate_total_unmet < current_total_unmet:
                # always accept if reduces unmet
                accept = True
                improving = True

            elif candidate_total_unmet == current_total_unmet \
                    and candidate_cost < current_cost - eps:
                # always accept if same unmet and reduces cost
                accept    = True
                improving = True

            elif candidate_total_unmet == current_total_unmet \
                    and candidate_cost >= current_cost - eps:
                # probabilistic acceptance for neutral or worsening cost
                delta  = candidate_cost - current_cost
                prob   = math.exp(-delta / T) if delta > 0 else 1.0
                accept = random.random() < prob
                improving = False

            else:
                # never accept if increases unmet
                accept    = False
                improving = False

            if accept:
                if candidate is None:
                    result = self.apply_move(solution, move, inplace=True)
                    if result is None:
                        move_stats[neighborhood]["none"] += 1
                        T *= self.alpha
                        iteration += 1
                        continue
                    solution = result
                else:
                    solution = candidate
                placements, _ = decode(solution, self.stocks)
                current_cost, current_unmet, current_overproduced = evaluate(
                    solution, placements, self.stocks, self.products
                )
                if sum(current_unmet.values()) > 0:
                    print(f"  WARNING: unmet after {move[0]}: {current_unmet}")

                if improving:
                    move_stats[neighborhood]["improving"] += 1
                    if current_cost < best_cost - eps:
                        best_cost     = current_cost
                        best_solution = copy_solution(solution)
                        print(f"  SA [{iteration}]: ✓ new best  "
                            f"cost={best_cost:.4f}  T={T:.4f}  move={move[0]}")
                    else:
                        print(f"  SA [{iteration}]: improving  "
                            f"cost={current_cost:.4f}  T={T:.4f}  move={move[0]}")
                else:
                    move_stats[neighborhood]["accepted_worse"] += 1
                    accepted_worse += 1
                    print(f"  SA [{iteration}]: accepted worse  "
                        f"cost={current_cost:.4f}  delta={candidate_cost - best_cost:.4f}  "
                        f"prob={prob:.4f}  T={T:.4f}  move={move[0]}")
            else:
                move_stats[neighborhood]["rejected"] += 1
                rejected += 1

            # cool temperature
            T *= self.alpha
            iteration += 1

        print(f"  SA finished: best={best_cost:.4f}  "
            f"iterations={iteration}  T_final={T:.6f}  "
            f"accepted_worse={accepted_worse}  rejected={rejected}")

        print(f"\n=== MOVE STATISTICS (SA) ===")
        for n, stats in move_stats.items():
            hit_rate = (stats['improving'] / stats['sampled'] * 100
                        if stats['sampled'] > 0 else 0)
            print(f"  {n:25s}: sampled={stats['sampled']:5d}  "
                f"improving={stats['improving']:4d}  "
                f"accepted_worse={stats['accepted_worse']:4d}  "
                f"rejected={stats['rejected']:4d}  "
                f"infeasible={stats['infeasible']:4d}  "
                f"none={stats['none']:5d}  "
                f"hit_rate={hit_rate:.1f}%")

        improvement_counts = {
            n: s["improving"] for n, s in move_stats.items() if s["improving"] > 0
        }
        return best_solution, repair_counts, improvement_counts, move_stats
    
class SimulatedAnnealingAdaptiveAlpha(LocalSearch):
    """
    SA with fixed T_init=100 but adaptive alpha computed so that
    T reaches T_min exactly at the time limit.
    Everything else identical to SimulatedAnnealing.
    """

    def __init__(self, stocks, products, patterns,
                 T_init              : float = 100.0,
                 T_min               : float = 1e-3,
                 max_iterations      : int   = 10_000_000,
                 time_limit          : float = 60.0,
                 neighborhood_weights: Dict[str, float] = None,
                 verbose             : bool  = False,
                 seed                : int   = None):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.T_init     = T_init
        self.T_min      = T_min
        self.time_limit = time_limit
        self.seed       = seed

        default_weights = {
            "remove"              : 1.0,
            "swap"                : 1.0,
            "relocate"            : 1.0,
            "stock_reset"         : 1.0,
            "pattern_replace_all" : 1.0,
            "insert"              : 1.0,
            "stock_open"          : 1.0,
            "close_open"          : 1.0,
        }
        active_weights = neighborhood_weights if neighborhood_weights \
                         else default_weights
        self.neighborhoods        = list(active_weights.keys())
        self.neighborhood_weights = list(active_weights.values())

    def run(self, solution: Solution):
        if self.seed is not None:
            random.seed(self.seed)

        solution, repair_counts = self.repair(solution)

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )
        """
        if sum(current_unmet.values()) > 0:
            print(f"  SA_AdaptiveAlpha: repair failed — unmet={current_unmet}")
            return solution, repair_counts, {}, {}
        """

        T = self.T_init

  
        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }

        t_probe = time.time()
        for _ in range(100):
            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            samplers[neighborhood](solution)
        t_per_iter = (time.time() - t_probe) / 100
        estimated_iterations = max(100, self.time_limit / t_per_iter)
        alpha = (self.T_min / T) ** (1.0 / estimated_iterations)
        # ← END CHANGE

        print(f"  SA_AdaptiveAlpha start: cost={current_cost:.4f}  "
              f"T_init={T:.4f}  alpha={alpha:.8f}  "
              f"estimated_iterations={estimated_iterations:.0f}")

        best_solution = copy_solution(solution)
        best_cost     = current_cost

        move_stats = {
            n: {"sampled": 0, "improving": 0, "accepted_worse": 0,
                "infeasible": 0, "none": 0, "rejected": 0}
            for n in self.neighborhoods
        }

        accepted_worse = 0
        rejected       = 0
        start_time     = time.time()
        self.convergence_log = []
        _next_log      = start_time + self.log_interval
        iteration      = 0
        eps            = 1e-6

        while T > self.T_min and iteration < self.max_iterations:
            _now = time.time()
            if _now - start_time > self.time_limit:
                print(f"  SA_AdaptiveAlpha: time limit reached at iteration {iteration}")
                break
            if _now >= _next_log:
                self._log_convergence(_now - start_time, current_cost)
                _next_log = _now + self.log_interval

            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)

            if move is None:
                move_stats[neighborhood]["none"] += 1
                continue

            move_stats[neighborhood]["sampled"] += 1
            move_type = move[0]

            if move_type in ("relocate", "relocate_with_slide"):
                if move_type == "relocate":
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos = move
                else:
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos, gap_i = move
                cost_delta             = self._delta_relocate(solution, stock_id_from, index,
                                                            stock_id_to, eidx_to, pos)
                candidate_cost         = current_cost + cost_delta
                candidate_unmet        = current_unmet
                candidate_overproduced = current_overproduced
                candidate              = None

            else:
                if move_type == "remove":
                    _, stock_id, index = move
                    cost_delta, _ = self._delta_remove(solution, stock_id, index)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= alpha  # ← use local alpha not self.alpha
                        continue
                elif move_type in ("swap", "swap_with_slide"):
                    if move_type == "swap":
                        _, stock_id, index, new_pattern, new_eidx, new_pos = move
                    else:
                        _, stock_id, index, new_pattern, new_eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_swap(solution, stock_id, index,
                                                    new_pattern, new_eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= alpha  # ← use local alpha not self.alpha
                        continue
                elif move_type == "pattern_replace_all":
                    _, stock_id, old_pattern, new_pattern = move
                    new_eidx = min(
                        range(len(new_pattern.stock_entries[stock_id])),
                        key=lambda i: new_pattern.stock_entries[stock_id][i].cost_per_rep
                    )
                    cost_delta, _ = self._delta_pattern_replace_all(
                        solution, stock_id, old_pattern, new_pattern, new_eidx
                    )
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= alpha  # ← use local alpha not self.alpha
                        continue
                elif move_type in ("insert", "insert_with_slide"):
                    if move_type == "insert":
                        _, stock_id, pattern, eidx, new_pos = move
                    else:
                        _, stock_id, pattern, eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_insert(solution, stock_id, pattern, eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= alpha  # ← use local alpha not self.alpha
                        continue

                candidate = self.apply_move(solution, move, inplace=False)
                if candidate is None:
                    move_stats[neighborhood]["infeasible"] += 1
                    iteration += 1
                    T *= alpha  # ← use local alpha not self.alpha
                    continue
                placements, fully_placed = decode(candidate, self.stocks)
                if not fully_placed:
                    move_stats[neighborhood]["infeasible"] += 1
                    iteration += 1
                    T *= alpha  # ← use local alpha not self.alpha
                    continue
                candidate_cost, candidate_unmet, candidate_overproduced = evaluate(
                    candidate, placements, self.stocks, self.products
                )

            current_total_unmet   = sum(current_unmet.values())
            candidate_total_unmet = sum(candidate_unmet.values())

            if candidate_total_unmet < current_total_unmet:
                accept    = True
                improving = True
            elif candidate_total_unmet == current_total_unmet \
                    and candidate_cost < current_cost - eps:
                accept    = True
                improving = True
            elif candidate_total_unmet == current_total_unmet \
                    and candidate_cost >= current_cost - eps:
                delta  = candidate_cost - current_cost
                prob   = math.exp(-delta / T) if delta > 0 else 1.0
                accept = random.random() < prob
                improving = False
            else:
                accept    = False
                improving = False

            if accept:
                if candidate is None:
                    result = self.apply_move(solution, move, inplace=True)
                    if result is None:
                        move_stats[neighborhood]["none"] += 1
                        T *= alpha  # ← use local alpha not self.alpha
                        iteration += 1
                        continue
                    solution = result
                else:
                    solution = candidate
                placements, _ = decode(solution, self.stocks)
                current_cost, current_unmet, current_overproduced = evaluate(
                    solution, placements, self.stocks, self.products
                )
                if sum(current_unmet.values()) > 0:
                    print(f"  WARNING: unmet after {move[0]}: {current_unmet}")

                if improving:
                    move_stats[neighborhood]["improving"] += 1
                    if current_cost < best_cost - eps:
                        best_cost     = current_cost
                        best_solution = copy_solution(solution)
                        print(f"  SA_AdaptiveAlpha [{iteration}]: ✓ new best  "
                              f"cost={best_cost:.4f}  T={T:.4f}  move={move[0]}")
                    else:
                        print(f"  SA_AdaptiveAlpha [{iteration}]: improving  "
                              f"cost={current_cost:.4f}  T={T:.4f}  move={move[0]}")
                else:
                    move_stats[neighborhood]["accepted_worse"] += 1
                    accepted_worse += 1
                    print(f"  SA_AdaptiveAlpha [{iteration}]: accepted worse  "
                          f"cost={current_cost:.4f}  delta={candidate_cost - best_cost:.4f}  "
                          f"prob={prob:.4f}  T={T:.4f}  move={move[0]}")
            else:
                move_stats[neighborhood]["rejected"] += 1
                rejected += 1

            T *= alpha  # ← use local alpha not self.alpha
            iteration += 1

        print(f"  SA_AdaptiveAlpha finished: best={best_cost:.4f}  "
              f"iterations={iteration}  T_final={T:.6f}  "
              f"accepted_worse={accepted_worse}  rejected={rejected}")

        print(f"\n=== MOVE STATISTICS (SA_AdaptiveAlpha) ===")
        for n, stats in move_stats.items():
            hit_rate = (stats['improving'] / stats['sampled'] * 100
                        if stats['sampled'] > 0 else 0)
            print(f"  {n:25s}: sampled={stats['sampled']:5d}  "
                  f"improving={stats['improving']:4d}  "
                  f"accepted_worse={stats['accepted_worse']:4d}  "
                  f"rejected={stats['rejected']:4d}  "
                  f"infeasible={stats['infeasible']:4d}  "
                  f"none={stats['none']:5d}  "
                  f"hit_rate={hit_rate:.1f}%")

        improvement_counts = {
            n: s["improving"] for n, s in move_stats.items() if s["improving"] > 0
        }
        return best_solution, repair_counts, improvement_counts, move_stats


# ---------------------------------------------------------------------------
# ILS — Iterated Local Search
# ---------------------------------------------------------------------------

class IteratedLocalSearch(LocalSearch):
    """
    Iterated Local Search — alternates between local search and perturbation
    to escape local optima.

    Pipeline per ILS iteration:
        1. Apply local search on current solution → local optimum s*
        2. Perturb s* by closing k stocks → s' (possibly infeasible)
        3. Repair s' if infeasible
        4. Apply local search on s' → new local optimum s**
        5. Accept s** if better than best known, else revert to s*
        6. Repeat until time limit or max_ils_iterations reached

    Perturbation strength k is controlled by perturb_k (number of stocks
    to close). If no improvement is found after patience iterations,
    k is increased by 1 up to perturb_k_max.
    """

    def __init__(self, stocks, products, patterns,
                 max_ils_iterations      : int   = 9999,
                 local_search_iterations : int   = 1000,
                 local_search_time       : float = 30.0,
                 local_search_method     : str   = "FI",
                 init_ls_iterations      : int   = None,
                 init_ls_time            : float = None,
                 init_ls_method          : str   = None,
                 perturb_k               : int   = 1,
                 perturb_k_max           : int   = 3,
                 patience                : int   = 5,
                 time_limit              : float = 300.0,
                 active_moves=None,
                 T_init                  : float = 100.0,
                 alpha                   : float = 0.991595,
                 sa_neighborhood_weights : dict  = None,
                 verbose                 : bool  = False, 
                 seed= None):
        super().__init__(stocks, products, patterns,
                         max_ils_iterations, verbose)
        self.max_ils_iterations      = max_ils_iterations
        self.local_search_iterations = local_search_iterations
        self.local_search_time       = local_search_time
        self.local_search_method     = local_search_method
        self.init_ls_iterations      = init_ls_iterations or local_search_iterations
        self.init_ls_time            = init_ls_time       or local_search_time
        self.init_ls_method          = init_ls_method     or local_search_method
        self.perturb_k               = perturb_k
        self.perturb_k_max           = perturb_k_max
        self.patience                = patience
        self.time_limit              = time_limit
        self.active_moves              = active_moves
        self.T_init                  = T_init
        self.alpha                   = alpha
        self.sa_neighborhood_weights = sa_neighborhood_weights
        self.seed= seed
    def _make_initial_local_search(self, time_budget=None):
        ls_time = min(self.init_ls_time, time_budget) \
            if time_budget else self.init_ls_time
        if self.init_ls_method == "SD":
            ls = SteepestImprovement(
                self.stocks, self.products, self.patterns,
                max_iterations=self.init_ls_iterations,
                time_limit=ls_time,
                active_moves=self.active_moves,
                verbose=self.verbose
            )
        elif self.local_search_method == "SA":
            ls = SimulatedAnnealing(
                self.stocks, self.products, self.patterns,
                time_limit           = ls_time,
                T_init               = self.T_init,
                alpha                = self.alpha,
                neighborhood_weights = self.sa_neighborhood_weights,
                verbose              = False,
            )
        elif self.init_ls_method == "TS":
            ls = TabuSearch(
                self.stocks, self.products, self.patterns,
                max_iterations=self.init_ls_iterations,
                time_limit=ls_time,
                verbose=self.verbose
            )
        else:
            ls = FirstImprovement(
                self.stocks, self.products, self.patterns,
                max_iterations=self.init_ls_iterations,
                time_limit=ls_time,
                verbose=self.verbose
            )
        ls.convergence_csv_path = self.convergence_csv_path
        ls.log_interval = self.log_interval
        return ls

    def _perturb(self, solution: Solution, k: int) -> Solution:
        """
        Perturbation — close k randomly chosen active stocks.
        Returns a new solution that may be infeasible.
        The closed stocks are chosen randomly — no bias toward
        good or bad stocks, to ensure genuine diversification.
        """
        sol = copy_solution(solution)
        active = list(sol.active.keys())
        if not active:
            return sol

        k_actual = min(k, len(active))
        stocks_to_close = random.sample(active, k_actual)

        print(f"  Perturbation: closing {stocks_to_close}  (k={k_actual})")
        for stock_id in stocks_to_close:
            sol.remove_stock(stock_id)

        return sol

    def _make_local_search(self, time_budget=None):
        ls_time = min(self.local_search_time, time_budget) \
            if time_budget else self.local_search_time
        if self.local_search_method == "SD":
            ls = SteepestImprovement(
                self.stocks, self.products, self.patterns,
                max_iterations=self.local_search_iterations,
                time_limit=ls_time,
                active_moves=self.active_moves,
                verbose=False
            )
        elif self.local_search_method == "SA":
            ls = SimulatedAnnealing(
                self.stocks, self.products, self.patterns,
                max_iterations=self.local_search_iterations,
                time_limit=ls_time,
                T_init               = self.T_init,
                alpha                = self.alpha,
                neighborhood_weights = self.sa_neighborhood_weights,
                verbose=False
            )
        elif self.local_search_method == "TS":
            ls = TabuSearch(
                self.stocks, self.products, self.patterns,
                max_iterations=self.local_search_iterations,
                time_limit=ls_time,
                verbose=False
            )
        else:
            ls = FirstImprovement(
                self.stocks, self.products, self.patterns,
                max_iterations=self.local_search_iterations,
                time_limit=ls_time,
                verbose=False
            )
        ls.convergence_csv_path = self.convergence_csv_path
        ls.log_interval = self.log_interval
        return ls

    def run(self, solution: Solution):
        """
        Run ILS starting from the given solution.
        Returns (best_solution, repair_counts, improvement_counts).
        """
        if self.seed is not None: 
            random.seed(self.seed)
        start_time = time.time()
        self.start_time = start_time
        self.convergence_log = []

        # --- Phase 1: initial repair ---
        solution, repair_counts = self.repair(solution)

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, _ = evaluate(
            solution, placements, self.stocks, self.products
        )
        """
        if sum(current_unmet.values()) > 0:
            print(f"  ILS: initial repair failed — unmet={current_unmet}")
            return solution, repair_counts, {}
        """

        print(f"\n  {'='*55}")
        print(f"  ILS — initial local search ({self.init_ls_method})  "
              f"max_iter={self.init_ls_iterations}  time={self.init_ls_time}s")
        print(f"  {'='*55}")
        remaining = max(1.0, self.time_limit - (time.time() - start_time))
    
        ls = self._make_local_search(time_budget=remaining)
        result = ls.run(solution)
        if self.init_ls_method == "SD":
            solution, _, ls_counts, _, _ = result
        else:
            solution, _, ls_counts, _ = result

        placements, _ = decode(solution, self.stocks)
        best_cost, best_unmet, _ = evaluate(
            solution, placements, self.stocks, self.products
        )
        best_solution = copy_solution(solution)

        print(f"\n  {'='*55}")
        print(f"  ILS starting point: cost={best_cost:.4f}  "
              f"stocks={len(solution.active)}")
        print(f"  Config: max_iter={self.max_ils_iterations}  "
              f"k={self.perturb_k}→{self.perturb_k_max}  "
              f"patience={self.patience}")
        print(f"  Init LS : method={self.init_ls_method}  "
              f"iter={self.init_ls_iterations}  time={self.init_ls_time}s")
        print(f"  Loop LS : method={self.local_search_method}  "
              f"iter={self.local_search_iterations}  time={self.local_search_time}s")
        print(f"  {'='*55}")

        improvement_counts = dict(ls_counts)
        no_improve_count   = 0
        current_k          = self.perturb_k

        # --- ILS main loop ---
        for ils_iter in range(self.max_ils_iterations):
            if time.time() - start_time > self.time_limit:
                print(f"  ILS: time limit reached at iteration {ils_iter}")
                break

            print(f"\n  {'─'*55}")
            print(f"  ILS iteration {ils_iter+1}/{self.max_ils_iterations}  "
                  f"k={current_k}  no_improve={no_improve_count}/{self.patience}  "
                  f"best={best_cost:.4f}  method={self.local_search_method}")
            print(f"  {'─'*55}")

            # perturbation
            sol_perturbed = self._perturb(best_solution, current_k)
            print(f"  Perturbation: closed {current_k} stock(s)  "
                  f"active={len(sol_perturbed.active)}")

            placements, _ = decode(sol_perturbed, self.stocks)
            _, unmet_perturbed, _ = evaluate(
                sol_perturbed, placements, self.stocks, self.products
            )
            total_unmet = sum(unmet_perturbed.values())
            print(f"  After perturbation: unmet={total_unmet}")

            # repair if needed
            if total_unmet > 0:
                print(f"  Repairing...")
                sol_perturbed, iter_repair = self.repair(
                    sol_perturbed, time_limit=30.0
                )
                placements, _ = decode(sol_perturbed, self.stocks)
                _, unmet_after, _ = evaluate(
                    sol_perturbed, placements, self.stocks, self.products
                )
                total_unmet_after = sum(unmet_after.values())
                print(f"  After repair: unmet={total_unmet_after}  "
                      f"active={len(sol_perturbed.active)}")
                if total_unmet_after > 0:
                    print(f"  Repair failed — skipping this iteration")
                    no_improve_count += 1
                    if no_improve_count >= self.patience:
                        current_k = min(current_k + 1, self.perturb_k_max)
                        no_improve_count = 0
                        print(f"  Patience reached — increasing k to {current_k}")
                    continue
            else:
                print(f"  Already feasible after perturbation — skipping repair")

            # local search on perturbed solution
            placements, _ = decode(sol_perturbed, self.stocks)
            cost_before_ls, _, _ = evaluate(
                sol_perturbed, placements, self.stocks, self.products
            )
            print(f"  Running local search ({self.local_search_method})  "
                  f"start_cost={cost_before_ls:.4f}")

            remaining = max(1.0, self.time_limit - (time.time() - start_time))
            ls_iter = self._make_local_search(time_budget=remaining)
            result = ls_iter.run(sol_perturbed)
            if self.local_search_method == "SD":
                sol_improved, _, iter_ls_counts, _, _ = result
            else:
                sol_improved, _, iter_ls_counts, _ = result

            placements, _ = decode(sol_improved, self.stocks)
            iter_cost, iter_unmet, _ = evaluate(
                sol_improved, placements, self.stocks, self.products
            )
            print(f"  After local search: cost={iter_cost:.4f}  "
                  f"ls_improvement={cost_before_ls - iter_cost:.4f}  "
                  f"moves={iter_ls_counts}")

            # accumulate improvement counts
            for move_type, count in iter_ls_counts.items():
                improvement_counts[move_type] = \
                    improvement_counts.get(move_type, 0) + count

            # acceptance criterion — accept if better than best
            if sum(iter_unmet.values()) == 0 and iter_cost < best_cost - 1e-6:
                improvement = best_cost - iter_cost
                best_cost     = iter_cost
                best_solution = copy_solution(sol_improved)
                no_improve_count = 0
                current_k        = self.perturb_k
                print(f"  ✓ New best found: cost={best_cost:.4f}  "
                      f"improvement={improvement:.4f}  "
                      f"stocks={len(best_solution.active)}  "
                      f"k reset to {current_k}")
            else:
                no_improve_count += 1
                print(f"  ✗ No improvement: iter={iter_cost:.4f}  "
                      f"best={best_cost:.4f}  "
                      f"no_improve={no_improve_count}/{self.patience}")

                if no_improve_count >= self.patience:
                    current_k = min(current_k + 1, self.perturb_k_max)
                    no_improve_count = 0
                    print(f"  Patience reached — increasing k to {current_k}")

        print(f"\n  {'='*55}")
        print(f"  ILS finished: best_cost={best_cost:.4f}  "
              f"stocks={len(best_solution.active)}")
        print(f"  {'='*55}")

        return best_solution, repair_counts, improvement_counts

class VNS(LocalSearch):
    def __init__(self, stocks, products, patterns,
                 sd_time_limit   : float = 30.0,
                 fi_time_limit   : float = 60.0,
                 fi_max_iterations : int   = 1500,
                 fi_neighborhood_weights : dict  = None,
                 n2_method               : str   = "FI",   # "FI" or "SA"
                 T_init                  : float = 100.0,
                 alpha                   : float = 0.991595,
                 sa_neighborhood_weights : dict  = None,
                 sd_active_moves : set=None,
                 max_iterations  : int   = 100,  # safety cap only
                 time_limit      : float = 300.0,
                 verbose         : bool  = False, 
                 seed= None):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.sd_time_limit = sd_time_limit
        self.fi_time_limit = fi_time_limit
        self.fi_max_iterations = fi_max_iterations
        self.fi_neighborhood_weights = fi_neighborhood_weights
        self.n2_method               = n2_method
        self.T_init                  = T_init
        self.alpha                   = alpha
        self.sa_neighborhood_weights = sa_neighborhood_weights
        self.time_limit    = time_limit
        self.sd_active_moves = sd_active_moves or {"relocate"}
        self.seed= seed

    def run(self, solution: Solution):
        if self.seed is not None: 
            random.seed(self.seed)
        solution, repair_counts = self.repair(solution)

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )

        best_solution      = copy_solution(solution)
        best_cost          = current_cost
        improvement_counts = {}
        start_time         = time.time()
        self.start_time    = start_time
        self.convergence_log = []
        fi_max_iterations  = self.fi_max_iterations

        print(f"  VNS start: cost={best_cost:.4f}")

        def run_n1(sol):
            remaining = max(1.0, self.time_limit - (time.time() - start_time))
            sd = SteepestImprovement(
                self.stocks, self.products, self.patterns,
                max_iterations = 999,
                time_limit     = min(self.sd_time_limit, remaining),
                active_moves   = self.sd_active_moves,
                verbose        = False,
            )
            d.convergence_csv_path = self.convergence_csv_path
            sd.log_interval = self.log_interval
            sd.time_offset = time.time() - start_time
            sol_out, _, sd_counts, _, _ = sd.run(copy_solution(sol))
            placements, _ = decode(sol_out, self.stocks)
            cost_out, _, _ = evaluate(sol_out, placements, self.stocks, self.products)
            return sol_out, cost_out, sd_counts

        def run_n2(sol, max_iter):
            remaining = max(1.0, self.time_limit - (time.time() - start_time))
            # replace
            if self.n2_method == "SA":
                ls = SimulatedAnnealing(
                    self.stocks, self.products, self.patterns,
                    time_limit           = min(self.fi_time_limit, remaining),
                    T_init               = self.T_init,
                    alpha                = self.alpha,
                    neighborhood_weights = self.sa_neighborhood_weights,
                    verbose              = False,
                )
            else:  # default FI
                ls = FirstImprovement(
                    self.stocks, self.products, self.patterns,
                    max_iterations       = max_iter,
                    time_limit           = min(self.fi_time_limit, remaining),
                    neighborhood_weights = self.fi_neighborhood_weights,
                    verbose              = False,
                )
            ls.convergence_csv_path = self.convergence_csv_path
            ls.log_interval = self.log_interval
            ls.time_offset = time.time() - start_time
            sol_out, _, ls_counts, _ = ls.run(sol)
            placements, _ = decode(sol_out, self.stocks)
            cost_out, _, _ = evaluate(sol_out, placements, self.stocks, self.products)
            return sol_out, cost_out, ls_counts

        # --- Iteration 0 — run N1 then N2 regardless ---
        print(f"\n  {'─'*50}")
        print(f"  VNS iteration 0 (initial)  best={best_cost:.4f}")

        if time.time() - start_time > self.time_limit:
            print(f"  VNS: time limit reached before iteration 0")
            return best_solution, repair_counts, improvement_counts

        print(f"  N1: SD-relocate  time_limit={self.sd_time_limit}s")
        sol_n1, cost_n1, sd_counts = run_n1(best_solution)
        print(f"  After N1: cost={cost_n1:.4f}  improvement={best_cost - cost_n1:.4f}")
        for move, count in sd_counts.items():
            improvement_counts[move] = improvement_counts.get(move, 0) + count

        if time.time() - start_time > self.time_limit:
            print(f"  VNS: time limit reached after N1 iteration 0")
            return best_solution, repair_counts, improvement_counts

        print(f"  N2: {self.n2_method}  time_limit={self.fi_time_limit}s  "
            f"max_iter={fi_max_iterations}")
        sol_n2, cost_n2, fi_counts = run_n2(sol_n1, fi_max_iterations)
        print(f"  After N2: cost={cost_n2:.4f}  improvement={cost_n1 - cost_n2:.4f}")
        for move, count in fi_counts.items():
            improvement_counts[move] = improvement_counts.get(move, 0) + count

        if time.time() - start_time > self.time_limit:
            print(f"  VNS: time limit reached after N2 iteration 0")
            if cost_n2 < best_cost - 1e-6:
                best_cost     = cost_n2
                best_solution = copy_solution(sol_n2)
            return best_solution, repair_counts, improvement_counts

        if cost_n2 < best_cost - 1e-6:
            improvement_ratio = (best_cost - cost_n2) / best_cost
            best_cost         = cost_n2
            best_solution     = copy_solution(sol_n2)
            if improvement_ratio > 0.05:
                fi_max_iterations = self.fi_max_iterations
                print(f"  ✓ Large improvement ({improvement_ratio:.1%}) "
                    f"— reset fi_max_iter to {fi_max_iterations}")
            else:
                fi_max_iterations = max(100, int(fi_max_iterations * 0.7))
                print(f"  ✓ Small improvement ({improvement_ratio:.1%}) "
                    f"— reduce fi_max_iter to {fi_max_iterations}")
        else:
            print(f"  ✗ N2 found no improvement at iteration 0 — stopping")
            return best_solution, repair_counts, improvement_counts

        # --- Iterations > 0 ---
        for iteration in range(1, self.max_iterations):
            if time.time() - start_time > self.time_limit:
                print(f"  VNS: time limit reached at iteration {iteration}")
                break

            print(f"\n  {'─'*50}")
            print(f"  VNS iteration {iteration}  best={best_cost:.4f}  "
                f"fi_max_iter={fi_max_iterations}")

            # run N1
            print(f"  N1: SD-relocate  time_limit={self.sd_time_limit}s")
            sol_n1, cost_n1, sd_counts = run_n1(best_solution)
            print(f"  After N1: cost={cost_n1:.4f}  improvement={best_cost - cost_n1:.4f}")
            for move, count in sd_counts.items():
                improvement_counts[move] = improvement_counts.get(move, 0) + count

            if cost_n1 >= best_cost - 1e-6:
                print(f"  ✗ N1 found no improvement — stopping")
                break

            # N1 improved — run N2
            print(f"  N2: {self.n2_method}  time_limit={self.fi_time_limit}s  "
                f"max_iter={fi_max_iterations}")
            sol_n2, cost_n2, fi_counts = run_n2(sol_n1, fi_max_iterations)
            print(f"  After N2: cost={cost_n2:.4f}  improvement={cost_n1 - cost_n2:.4f}")
            for move, count in fi_counts.items():
                improvement_counts[move] = improvement_counts.get(move, 0) + count

            if cost_n2 < best_cost - 1e-6:
                improvement_ratio = (best_cost - cost_n2) / best_cost
                best_cost         = cost_n2
                best_solution     = copy_solution(sol_n2)
                if improvement_ratio > 0.05:
                    fi_max_iterations = self.fi_max_iterations
                    print(f"  ✓ Large improvement ({improvement_ratio:.1%}) "
                        f"— reset fi_max_iter to {fi_max_iterations}")
                else:
                    fi_max_iterations = max(100, int(fi_max_iterations * 0.7))
                    print(f"  ✓ Small improvement ({improvement_ratio:.1%}) "
                        f"— reduce fi_max_iter to {fi_max_iterations}")
            else:
                print(f"  ✗ N2 found no improvement — stopping")
                break

        print(f"\n  VNS finished: best={best_cost:.4f}")
        return best_solution, repair_counts, improvement_counts
    
class SAInfeasible(LocalSearch):
    """
    SA variant that accepts infeasible starting solutions.
    Uses penalized objective: obj = cost + penalty * total_unmet
    No repair phase — starts directly from any solution.
    Tracks best feasible solution separately.
    Returns best feasible if found, otherwise least-unmet solution.
    """

    def __init__(self, stocks, products, patterns,
                 T_init              : float = None,
                 T_min               : float = 1e-3,
                 max_iterations      : int   = 10_000_000,
                 time_limit          : float = 60.0,
                 neighborhood_weights: Dict[str, float] = None,
                 penalty             : float = 10000.0,
                 verbose             : bool  = False, 
                 seed= None):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.T_init     = T_init
        self.T_min      = T_min
        self.time_limit = time_limit
        self.penalty    = penalty
        self.seed= seed

        default_weights = {
            "remove"              : 1.0,
            "swap"                : 2.0,
            "relocate"            : 4.0,
            "stock_reset"         : 5.0,
            "pattern_replace_all" : 1.0,
            "insert"              : 3.0,
            "stock_open"          : 5.0,
            "close_open"          : 1.0,
        }
        active_weights = neighborhood_weights if neighborhood_weights \
                         else default_weights

        self.neighborhoods        = list(active_weights.keys())
        self.neighborhood_weights = list(active_weights.values())

    def _penalized_obj(self, cost: float, unmet: Dict) -> float:
        return cost + self.penalty * sum(unmet.values())

    def _initial_temperature(self, solution: Solution) -> float:
        """Estimate T_init from random move sample."""
        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, _ = evaluate(
            solution, placements, self.stocks, self.products
        )
        current_obj = self._penalized_obj(current_cost, current_unmet)

        deltas = []
        for _ in range(50):
            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)
            if move is None:
                continue
            candidate = self.apply_move(solution, move, inplace=False)
            if candidate is None:
                continue
            placements, fully_placed = decode(candidate, self.stocks)
            if not fully_placed:
                continue
            candidate_cost, candidate_unmet, _ = evaluate(
                candidate, placements, self.stocks, self.products
            )
            candidate_obj = self._penalized_obj(candidate_cost, candidate_unmet)
            delta = candidate_obj - current_obj
            if delta > 0:
                deltas.append(delta)

        if not deltas:
            return 100.0
        avg_delta = sum(deltas) / len(deltas)
        return max(-avg_delta / math.log(0.8), 1.0)

    def run(self, solution: Solution):
        if self.seed is not None: 
            random.seed(self.seed)
        # --- evaluate starting solution ---
        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )
        current_obj = self._penalized_obj(current_cost, current_unmet)

        print(f"  SAInfeasible start: cost={current_cost:.4f}  "
              f"unmet={sum(current_unmet.values())}  "
              f"penalized_obj={current_obj:.4f}  "
              f"penalty={self.penalty}")

        # --- initial temperature ---
        T = self.T_init if self.T_init is not None \
            else self._initial_temperature(solution)
        
        # --- tracking ---
        best_feasible      = None
        best_feasible_cost = float('inf')
        best_solution      = copy_solution(solution)
        best_obj           = current_obj

        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }

        t_probe = time.time()
        for _ in range(100):
            neighborhood = random.choices(self.neighborhoods, weights=self.neighborhood_weights, k=1)[0]
            samplers[neighborhood](solution)
        t_per_iter = (time.time() - t_probe) / 100
        estimated_iterations = max(100, self.time_limit / t_per_iter)
        alpha = (self.T_min / T) ** (1.0 / estimated_iterations)
        print(f"  SAInfeasible: T_init={T:.4f}  alpha={alpha:.8f}  "
              f"T_min={self.T_min}")



        move_stats = {
            n: {"sampled": 0, "improving": 0, "accepted_worse": 0,
                "infeasible": 0, "none": 0, "rejected": 0}
            for n in self.neighborhoods
        }

        accepted_worse = 0
        rejected       = 0
        start_time     = time.time()
        iteration      = 0
        eps            = 1e-6

        while T > self.T_min and iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                print(f"  SAInfeasible: time limit reached at iteration {iteration}")
                break

            # --- sample move ---
            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)

            if move is None:
                move_stats[neighborhood]["none"] += 1
                continue

            move_stats[neighborhood]["sampled"] += 1

            # --- apply move and evaluate ---
            candidate = self.apply_move(solution, move, inplace=False)
            if candidate is None:
                move_stats[neighborhood]["infeasible"] += 1
                iteration += 1
                T *= alpha
                continue

            placements, fully_placed = decode(candidate, self.stocks)
            if not fully_placed:
                move_stats[neighborhood]["infeasible"] += 1
                iteration += 1
                T *= alpha
                continue

            candidate_cost, candidate_unmet, candidate_overproduced = evaluate(
                candidate, placements, self.stocks, self.products
            )
            candidate_obj = self._penalized_obj(candidate_cost, candidate_unmet)

            # --- acceptance criterion ---
            delta = candidate_obj - current_obj

            if delta < -eps:
                accept    = True
                improving = True
            else:
                prob      = math.exp(-delta / T) if T > 1e-10 else 0.0
                accept    = random.random() < prob
                improving = False

            if accept:
                solution             = candidate
                current_cost         = candidate_cost
                current_unmet        = candidate_unmet
                current_overproduced = candidate_overproduced
                current_obj          = candidate_obj

                # track best feasible
                if sum(current_unmet.values()) == 0:
                    if current_cost < best_feasible_cost - eps:
                        best_feasible_cost = current_cost
                        best_feasible      = copy_solution(solution)
                        print(f"  SAInfeasible [{iteration}]: ✓ new best feasible "
                              f"cost={current_cost:.4f}  T={T:.6f}")
                    

                # track best overall by penalized obj
                if current_obj < best_obj - eps:
                    best_obj      = current_obj
                    best_solution = copy_solution(solution)
                    if self.verbose:
                        print(f"  SAInfeasible [{iteration}]: new best obj "
                              f"obj={current_obj:.4f}  "
                              f"unmet={sum(current_unmet.values())}  "
                              f"cost={current_cost:.4f}  T={T:.6f}  "
                              f"move={move[0]}")

                if improving:
                    move_stats[neighborhood]["improving"] += 1
                else:
                    move_stats[neighborhood]["accepted_worse"] += 1
                    accepted_worse += 1
            else:
                move_stats[neighborhood]["rejected"] += 1
                rejected += 1

            T *= alpha
            iteration += 1

        # --- return best feasible if found, else best overall ---
        final_solution = best_feasible if best_feasible is not None \
                         else best_solution
        final_feasible = best_feasible is not None

        placements, _ = decode(final_solution, self.stocks)
        final_cost, final_unmet, _ = evaluate(
            final_solution, placements, self.stocks, self.products
        )

        print(f"  SAInfeasible finished: "
              f"cost={final_cost:.4f}  "
              f"unmet={sum(final_unmet.values())}  "
              f"feasible={final_feasible}  "
              f"iterations={iteration}  "
              f"T_final={T:.6f}  "
              f"accepted_worse={accepted_worse}  "
              f"rejected={rejected}")

        print(f"\n=== MOVE STATISTICS (SAInfeasible) ===")
        for n, stats in move_stats.items():
            hit_rate = (stats['improving'] / stats['sampled'] * 100
                        if stats['sampled'] > 0 else 0)
            print(f"  {n:25s}: sampled={stats['sampled']:5d}  "
                  f"improving={stats['improving']:4d}  "
                  f"accepted_worse={stats['accepted_worse']:4d}  "
                  f"rejected={stats['rejected']:4d}  "
                  f"infeasible={stats['infeasible']:4d}  "
                  f"none={stats['none']:5d}  "
                  f"hit_rate={hit_rate:.1f}%")

        improvement_counts = {
            n: s["improving"] for n, s in move_stats.items()
            if s["improving"] > 0
        }
        return final_solution, {}, improvement_counts, move_stats
    

#------------------------------------------------------------------------------------------------------------
# METHODS created but not implemented in the end

class SimulatedAnnealingAdaptiveTinit(LocalSearch):
    """
    SA with instance-specific T_init estimated from cost scale,
    targeting 80% initial acceptance. Alpha stays fixed at tuned value.
    Everything else identical to SimulatedAnnealing.
    """

    def __init__(self, stocks, products, patterns,
                 T_min               : float = 1e-3,
                 alpha               : float = 0.98786,
                 max_iterations      : int   = 10_000_000,
                 time_limit          : float = 60.0,
                 neighborhood_weights: Dict[str, float] = None,
                 target_acceptance= 0.8,
                 verbose             : bool  = False,
                 seed                : int   = None):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.T_min      = T_min
        self.alpha      = alpha
        self.time_limit = time_limit
        self.target_acceptance = target_acceptance
        self.seed       = seed

        default_weights = {
            "remove"              : 1.0,
            "swap"                : 1.0,
            "relocate"            : 1.0,
            "stock_reset"         : 1.0,
            "pattern_replace_all" : 1.0,
            "insert"              : 1.0,
            "stock_open"          : 1.0,
            "close_open"          : 1.0,
        }
        active_weights = neighborhood_weights if neighborhood_weights \
                         else default_weights
        self.neighborhoods        = list(active_weights.keys())
        self.neighborhood_weights = list(active_weights.values())

    def _initial_temperature(self, solution: Solution) -> float:
        """Same auto-estimation as SimulatedAnnealing."""
        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }
        placements, _ = decode(solution, self.stocks)
        current_cost, _, _ = evaluate(solution, placements, self.stocks, self.products)
        deltas = []
        for _ in range(50):
            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)
            if move is None:
                continue
            candidate = self.apply_move(solution, move, inplace=False)
            if candidate is None:
                continue
            placements, fully_placed = decode(candidate, self.stocks)
            if not fully_placed:
                continue
            candidate_cost, _, _ = evaluate(candidate, placements,
                                            self.stocks, self.products)
            delta = candidate_cost - current_cost
            if delta > 0:
                deltas.append(delta)
        if not deltas:
            return 100.0
        avg_delta = sum(deltas) / len(deltas)
        return max(-avg_delta / math.log(self.target_acceptance), 1.0)

    def run(self, solution: Solution):
        if self.seed is not None:
            random.seed(self.seed)

        solution, repair_counts = self.repair(solution)

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )
        """
        if sum(current_unmet.values()) > 0:
            print(f"  SA_AdaptiveTinit: repair failed — unmet={current_unmet}")
            return solution, repair_counts, {}, {}
        """

        # ← CHANGE: estimate T_init from cost scale
        T = self._initial_temperature(solution)
        # ← END CHANGE

        print(f"  SA_AdaptiveTinit start: cost={current_cost:.4f}  "
              f"T_init={T:.4f}  alpha={self.alpha}  T_min={self.T_min}")

        best_solution = copy_solution(solution)
        best_cost     = current_cost

        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }

        move_stats = {
            n: {"sampled": 0, "improving": 0, "accepted_worse": 0,
                "infeasible": 0, "none": 0, "rejected": 0}
            for n in self.neighborhoods
        }

        accepted_worse = 0
        rejected       = 0
        start_time     = time.time()
        iteration      = 0
        eps            = 1e-6

        while T > self.T_min and iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                print(f"  SA_AdaptiveTinit: time limit reached at iteration {iteration}")
                break

            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)

            if move is None:
                move_stats[neighborhood]["none"] += 1
                continue

            move_stats[neighborhood]["sampled"] += 1
            move_type = move[0]

            if move_type in ("relocate", "relocate_with_slide"):
                if move_type == "relocate":
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos = move
                else:
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos, gap_i = move
                cost_delta             = self._delta_relocate(solution, stock_id_from, index,
                                                            stock_id_to, eidx_to, pos)
                candidate_cost         = current_cost + cost_delta
                candidate_unmet        = current_unmet
                candidate_overproduced = current_overproduced
                candidate              = None

            else:
                if move_type == "remove":
                    _, stock_id, index = move
                    cost_delta, _ = self._delta_remove(solution, stock_id, index)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= self.alpha
                        continue
                elif move_type in ("swap", "swap_with_slide"):
                    if move_type == "swap":
                        _, stock_id, index, new_pattern, new_eidx, new_pos = move
                    else:
                        _, stock_id, index, new_pattern, new_eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_swap(solution, stock_id, index,
                                                    new_pattern, new_eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= self.alpha
                        continue
                elif move_type == "pattern_replace_all":
                    _, stock_id, old_pattern, new_pattern = move
                    new_eidx = min(
                        range(len(new_pattern.stock_entries[stock_id])),
                        key=lambda i: new_pattern.stock_entries[stock_id][i].cost_per_rep
                    )
                    cost_delta, _ = self._delta_pattern_replace_all(
                        solution, stock_id, old_pattern, new_pattern, new_eidx
                    )
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= self.alpha
                        continue
                elif move_type in ("insert", "insert_with_slide"):
                    if move_type == "insert":
                        _, stock_id, pattern, eidx, new_pos = move
                    else:
                        _, stock_id, pattern, eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_insert(solution, stock_id, pattern, eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= self.alpha
                        continue

                candidate = self.apply_move(solution, move, inplace=False)
                if candidate is None:
                    move_stats[neighborhood]["infeasible"] += 1
                    iteration += 1
                    T *= self.alpha
                    continue
                placements, fully_placed = decode(candidate, self.stocks)
                if not fully_placed:
                    move_stats[neighborhood]["infeasible"] += 1
                    iteration += 1
                    T *= self.alpha
                    continue
                candidate_cost, candidate_unmet, candidate_overproduced = evaluate(
                    candidate, placements, self.stocks, self.products
                )

            current_total_unmet   = sum(current_unmet.values())
            candidate_total_unmet = sum(candidate_unmet.values())

            if candidate_total_unmet < current_total_unmet:
                accept    = True
                improving = True
            elif candidate_total_unmet == current_total_unmet \
                    and candidate_cost < current_cost - eps:
                accept    = True
                improving = True
            elif candidate_total_unmet == current_total_unmet \
                    and candidate_cost >= current_cost - eps:
                delta  = candidate_cost - current_cost
                prob   = math.exp(-delta / T) if delta > 0 else 1.0
                accept = random.random() < prob
                improving = False
            else:
                accept    = False
                improving = False

            if accept:
                if candidate is None:
                    result = self.apply_move(solution, move, inplace=True)
                    if result is None:
                        move_stats[neighborhood]["none"] += 1
                        T *= self.alpha
                        iteration += 1
                        continue
                    solution = result
                else:
                    solution = candidate
                placements, _ = decode(solution, self.stocks)
                current_cost, current_unmet, current_overproduced = evaluate(
                    solution, placements, self.stocks, self.products
                )
                if sum(current_unmet.values()) > 0:
                    print(f"  WARNING: unmet after {move[0]}: {current_unmet}")

                if improving:
                    move_stats[neighborhood]["improving"] += 1
                    if current_cost < best_cost - eps:
                        best_cost     = current_cost
                        best_solution = copy_solution(solution)
                        print(f"  SA_AdaptiveTinit [{iteration}]: ✓ new best  "
                              f"cost={best_cost:.4f}  T={T:.4f}  move={move[0]}")
                    else:
                        print(f"  SA_AdaptiveTinit [{iteration}]: improving  "
                              f"cost={current_cost:.4f}  T={T:.4f}  move={move[0]}")
                else:
                    move_stats[neighborhood]["accepted_worse"] += 1
                    accepted_worse += 1
                    print(f"  SA_AdaptiveTinit [{iteration}]: accepted worse  "
                          f"cost={current_cost:.4f}  delta={candidate_cost - best_cost:.4f}  "
                          f"prob={prob:.4f}  T={T:.4f}  move={move[0]}")
            else:
                move_stats[neighborhood]["rejected"] += 1
                rejected += 1

            T *= self.alpha
            iteration += 1

        print(f"  SA_AdaptiveTinit finished: best={best_cost:.4f}  "
              f"iterations={iteration}  T_final={T:.6f}  "
              f"accepted_worse={accepted_worse}  rejected={rejected}")

        print(f"\n=== MOVE STATISTICS (SA_AdaptiveTinit) ===")
        for n, stats in move_stats.items():
            hit_rate = (stats['improving'] / stats['sampled'] * 100
                        if stats['sampled'] > 0 else 0)
            print(f"  {n:25s}: sampled={stats['sampled']:5d}  "
                  f"improving={stats['improving']:4d}  "
                  f"accepted_worse={stats['accepted_worse']:4d}  "
                  f"rejected={stats['rejected']:4d}  "
                  f"infeasible={stats['infeasible']:4d}  "
                  f"none={stats['none']:5d}  "
                  f"hit_rate={hit_rate:.1f}%")

        improvement_counts = {
            n: s["improving"] for n, s in move_stats.items() if s["improving"] > 0
        }
        return best_solution, repair_counts, improvement_counts, move_stats


class SimulatedAnnealingFullyAdaptive(LocalSearch):
    """
    SA with both instance-specific T_init AND adaptive alpha.
    T_init estimated from cost scale targeting 80% initial acceptance.
    Alpha computed so T reaches T_min exactly at the time limit.
    No parameters to tune — fully automatic.
    Everything else identical to SimulatedAnnealing.
    """

    def __init__(self, stocks, products, patterns,
                 T_min               : float = 1e-3,
                 max_iterations      : int   = 10_000_000,
                 time_limit          : float = 60.0,
                 neighborhood_weights: Dict[str, float] = None,
                 verbose             : bool  = False,
                 seed                : int   = None):
        super().__init__(stocks, products, patterns, max_iterations, verbose)
        self.T_min      = T_min
        self.time_limit = time_limit
        self.seed       = seed

        default_weights = {
            "remove"              : 1.0,
            "swap"                : 1.0,
            "relocate"            : 1.0,
            "stock_reset"         : 1.0,
            "pattern_replace_all" : 1.0,
            "insert"              : 1.0,
            "stock_open"          : 1.0,
            "close_open"          : 1.0,
        }
        active_weights = neighborhood_weights if neighborhood_weights \
                         else default_weights
        self.neighborhoods        = list(active_weights.keys())
        self.neighborhood_weights = list(active_weights.values())

    def _initial_temperature(self, solution: Solution) -> float:
        """Same auto-estimation as SimulatedAnnealing."""
        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }
        placements, _ = decode(solution, self.stocks)
        current_cost, _, _ = evaluate(solution, placements, self.stocks, self.products)
        deltas = []
        for _ in range(50):
            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)
            if move is None:
                continue
            candidate = self.apply_move(solution, move, inplace=False)
            if candidate is None:
                continue
            placements, fully_placed = decode(candidate, self.stocks)
            if not fully_placed:
                continue
            candidate_cost, _, _ = evaluate(candidate, placements,
                                            self.stocks, self.products)
            delta = candidate_cost - current_cost
            if delta > 0:
                deltas.append(delta)
        if not deltas:
            return 100.0
        avg_delta = sum(deltas) / len(deltas)
        return max(-avg_delta / math.log(0.8), 1.0)

    def run(self, solution: Solution):
        if self.seed is not None:
            random.seed(self.seed)

        solution, repair_counts = self.repair(solution)

        placements, _ = decode(solution, self.stocks)
        current_cost, current_unmet, current_overproduced = evaluate(
            solution, placements, self.stocks, self.products
        )
        """
        if sum(current_unmet.values()) > 0:
            print(f"  SA_FullyAdaptive: repair failed — unmet={current_unmet}")
            return solution, repair_counts, {}, {}
        """

        # ← CHANGE: estimate T_init and adaptive alpha
        samplers = {
            "remove"              : self.sample_remove_move,
            "swap"                : self.sample_swap_move,
            "relocate"            : self.sample_relocate_move,
            "stock_reset"         : self.sample_stock_reset_move,
            "pattern_replace_all" : self.sample_pattern_replace_all_move,
            "insert"              : self.sample_insert_move,
            "stock_open"          : self.sample_stock_open_move,
            "close_open"          : self.sample_close_open_move,
        }

        T = self._initial_temperature(solution)

        t_probe = time.time()
        for _ in range(100):
            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            samplers[neighborhood](solution)
        t_per_iter = (time.time() - t_probe) / 100
        estimated_iterations = max(100, self.time_limit / t_per_iter)
        alpha = (self.T_min / T) ** (1.0 / estimated_iterations)

        print(f"  SA_FullyAdaptive start: cost={current_cost:.4f}  "
              f"T_init={T:.4f}  alpha={alpha:.8f}  "
              f"estimated_iterations={estimated_iterations:.0f}")

        best_solution = copy_solution(solution)
        best_cost     = current_cost

        move_stats = {
            n: {"sampled": 0, "improving": 0, "accepted_worse": 0,
                "infeasible": 0, "none": 0, "rejected": 0}
            for n in self.neighborhoods
        }

        accepted_worse = 0
        rejected       = 0
        start_time     = time.time()
        iteration      = 0
        eps            = 1e-6

        while T > self.T_min and iteration < self.max_iterations:
            if time.time() - start_time > self.time_limit:
                print(f"  SA_FullyAdaptive: time limit reached at iteration {iteration}")
                break

            neighborhood = random.choices(
                self.neighborhoods, weights=self.neighborhood_weights, k=1
            )[0]
            move = samplers[neighborhood](solution)

            if move is None:
                move_stats[neighborhood]["none"] += 1
                continue

            move_stats[neighborhood]["sampled"] += 1
            move_type = move[0]

            if move_type in ("relocate", "relocate_with_slide"):
                if move_type == "relocate":
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos = move
                else:
                    _, stock_id_from, index, stock_id_to, pat, eidx_to, pos, gap_i = move
                cost_delta             = self._delta_relocate(solution, stock_id_from, index,
                                                            stock_id_to, eidx_to, pos)
                candidate_cost         = current_cost + cost_delta
                candidate_unmet        = current_unmet
                candidate_overproduced = current_overproduced
                candidate              = None

            else:
                if move_type == "remove":
                    _, stock_id, index = move
                    cost_delta, _ = self._delta_remove(solution, stock_id, index)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= alpha  # ← use local alpha
                        continue
                elif move_type in ("swap", "swap_with_slide"):
                    if move_type == "swap":
                        _, stock_id, index, new_pattern, new_eidx, new_pos = move
                    else:
                        _, stock_id, index, new_pattern, new_eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_swap(solution, stock_id, index,
                                                    new_pattern, new_eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= alpha  # ← use local alpha
                        continue
                elif move_type == "pattern_replace_all":
                    _, stock_id, old_pattern, new_pattern = move
                    new_eidx = min(
                        range(len(new_pattern.stock_entries[stock_id])),
                        key=lambda i: new_pattern.stock_entries[stock_id][i].cost_per_rep
                    )
                    cost_delta, _ = self._delta_pattern_replace_all(
                        solution, stock_id, old_pattern, new_pattern, new_eidx
                    )
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= alpha  # ← use local alpha
                        continue
                elif move_type in ("insert", "insert_with_slide"):
                    if move_type == "insert":
                        _, stock_id, pattern, eidx, new_pos = move
                    else:
                        _, stock_id, pattern, eidx, new_pos, gap_i = move
                    cost_delta, _ = self._delta_insert(solution, stock_id, pattern, eidx, new_pos)
                    if current_cost + cost_delta >= current_cost - eps:
                        move_stats[neighborhood]["infeasible"] += 1
                        iteration += 1
                        T *= alpha  # ← use local alpha
                        continue

                candidate = self.apply_move(solution, move, inplace=False)
                if candidate is None:
                    move_stats[neighborhood]["infeasible"] += 1
                    iteration += 1
                    T *= alpha  # ← use local alpha
                    continue
                placements, fully_placed = decode(candidate, self.stocks)
                if not fully_placed:
                    move_stats[neighborhood]["infeasible"] += 1
                    iteration += 1
                    T *= alpha  # ← use local alpha
                    continue
                candidate_cost, candidate_unmet, candidate_overproduced = evaluate(
                    candidate, placements, self.stocks, self.products
                )

            current_total_unmet   = sum(current_unmet.values())
            candidate_total_unmet = sum(candidate_unmet.values())

            if candidate_total_unmet < current_total_unmet:
                accept    = True
                improving = True
            elif candidate_total_unmet == current_total_unmet \
                    and candidate_cost < current_cost - eps:
                accept    = True
                improving = True
            elif candidate_total_unmet == current_total_unmet \
                    and candidate_cost >= current_cost - eps:
                delta  = candidate_cost - current_cost
                prob   = math.exp(-delta / T) if delta > 0 else 1.0
                accept = random.random() < prob
                improving = False
            else:
                accept    = False
                improving = False

            if accept:
                if candidate is None:
                    result = self.apply_move(solution, move, inplace=True)
                    if result is None:
                        move_stats[neighborhood]["none"] += 1
                        T *= alpha  # ← use local alpha
                        iteration += 1
                        continue
                    solution = result
                else:
                    solution = candidate
                placements, _ = decode(solution, self.stocks)
                current_cost, current_unmet, current_overproduced = evaluate(
                    solution, placements, self.stocks, self.products
                )
                if sum(current_unmet.values()) > 0:
                    print(f"  WARNING: unmet after {move[0]}: {current_unmet}")

                if improving:
                    move_stats[neighborhood]["improving"] += 1
                    if current_cost < best_cost - eps:
                        best_cost     = current_cost
                        best_solution = copy_solution(solution)
                        print(f"  SA_FullyAdaptive [{iteration}]: ✓ new best  "
                              f"cost={best_cost:.4f}  T={T:.4f}  move={move[0]}")
                    else:
                        print(f"  SA_FullyAdaptive [{iteration}]: improving  "
                              f"cost={current_cost:.4f}  T={T:.4f}  move={move[0]}")
                else:
                    move_stats[neighborhood]["accepted_worse"] += 1
                    accepted_worse += 1
                    print(f"  SA_FullyAdaptive [{iteration}]: accepted worse  "
                          f"cost={current_cost:.4f}  delta={candidate_cost - best_cost:.4f}  "
                          f"prob={prob:.4f}  T={T:.4f}  move={move[0]}")
            else:
                move_stats[neighborhood]["rejected"] += 1
                rejected += 1

            T *= alpha  # ← use local alpha
            iteration += 1

        print(f"  SA_FullyAdaptive finished: best={best_cost:.4f}  "
              f"iterations={iteration}  T_final={T:.6f}  "
              f"accepted_worse={accepted_worse}  rejected={rejected}")

        print(f"\n=== MOVE STATISTICS (SA_FullyAdaptive) ===")
        for n, stats in move_stats.items():
            hit_rate = (stats['improving'] / stats['sampled'] * 100
                        if stats['sampled'] > 0 else 0)
            print(f"  {n:25s}: sampled={stats['sampled']:5d}  "
                  f"improving={stats['improving']:4d}  "
                  f"accepted_worse={stats['accepted_worse']:4d}  "
                  f"rejected={stats['rejected']:4d}  "
                  f"infeasible={stats['infeasible']:4d}  "
                  f"none={stats['none']:5d}  "
                  f"hit_rate={hit_rate:.1f}%")

        improvement_counts = {
            n: s["improving"] for n, s in move_stats.items() if s["improving"] > 0
        }
        return best_solution, repair_counts, improvement_counts, move_stats
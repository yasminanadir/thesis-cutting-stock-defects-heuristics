"""
solution.py
-----------
Solution representation, decoder and cost evaluation
 
Data structure:
  Solution.active : dict stock_id -> [(pattern, entry_index, start_pos), ...]
  One entry per repetition, sorted by start_pos within each stock.
  entry_index refers to pattern.stock_entries[stock_id][entry_index] and
  identifies which PatternStockEntry governs the repetition.
  A new setup cost is charged when either the pattern changes or the
  entry_index changes for the same pattern on the same stock.
 
Functions:
  decode    -- validates placements and decodes a Solution into (pattern, entry, start, end) tuples
  evaluate  -- computes total cost, unmet demand and overproduction from decoded placements
  copy_solution -- copy of a Solution object
"""
 
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
 
from src.instance import Pattern
 
 
@dataclass
class Solution:
    """
    active : dict mapping stock_id -> ordered list of (pattern, entry_index, start_pos)
             One entry per repetition, sorted by start_pos.
             Only stocks that are actually used appear as keys.
    """
    active: Dict[str, List[Tuple[Pattern, int, float]]] = field(default_factory=dict)
 
    def is_empty(self) -> bool:
        return len(self.active) == 0
 
    def stocks_used(self) -> List[str]:
        return list(self.active.keys())
 
    def add_stock(self, stock_id: str) -> None:
        if stock_id not in self.active:
            self.active[stock_id] = []
 
    def remove_stock(self, stock_id: str) -> None:
        self.active.pop(stock_id, None)
 
    def add_repetition(self, stock_id: str, pattern: Pattern,
                       entry_index: int, start_pos: float) -> None:

        self.add_stock(stock_id)
        self.active[stock_id].append((pattern, entry_index, start_pos))
        self.active[stock_id].sort(key=lambda x: x[2])
 
    def remove_repetition(self, stock_id: str, index: int) -> None:
        
        if stock_id in self.active:
            self.active[stock_id].pop(index)
            if not self.active[stock_id]:
                self.remove_stock(stock_id)
 
    def get_repetition(self, stock_id: str, index: int) -> Tuple[Pattern, int, float]:
        return self.active[stock_id][index]
 
    def replace_repetition(self, stock_id: str, index: int,
                           new_pattern: Pattern, new_entry_index: int,
                           new_start_pos: float) -> None:
        """
        Replace the repetition at index with a new (pattern, entry_index, start_pos).
        List is re-sorted after replacement.
        """
        self.active[stock_id][index] = (new_pattern, new_entry_index, new_start_pos)
        self.active[stock_id].sort(key=lambda x: x[2])
 
    def __repr__(self) -> str:
        lines = ["Solution:"]
        for sid, entries in self.active.items():
            lines.append(f"  {sid}:")
            for pat, entry_idx, start_pos in entries:
                lines.append(f"    {pat.pattern_id}  entry={entry_idx}"
                             f"  start={start_pos:.0f}"
                             f"  end={start_pos + pat.length_consumed:.0f}")
        return "\n".join(lines)
 
 
# DECODER
 
def decode(solution: Solution, stocks: Dict, verbose: bool = False) -> Tuple[Dict, bool]:
    """
    Decode a solution into actual placements.
 
    For each repetition (pattern, entry_index, start_pos):
      - Retrieves the PatternStockEntry using entry_index
      - Verifies start_pos falls within one of the entry's windows
      - Verifies no overlap with the previous repetition
      - Verifies the repetition does not exceed stock length
 
    Returns:
        placements   : dict stock_id -> list of (pattern, entry_index, start, end)
        fully_placed : True if all repetitions were successfully placed
    """
    placements   = {}
    fully_placed = True
 
    for stock_id, entries in solution.active.items():
        placements[stock_id] = []
        stock = stocks[stock_id]
 
        for pattern, entry_idx, start_pos in entries:
            end_pos = start_pos + pattern.length_consumed
 
            if end_pos > stock.length + 1e-9:
                fully_placed = False
                if verbose:
                    print(f"  WARNING: {pattern.pattern_id} entry={entry_idx} "
                          f"at {start_pos:.0f} on {stock_id} "
                          f"— exceeds stock length {stock.length:.0f}")
                continue
 
            stock_entries = pattern.stock_entries.get(stock_id, [])
            if entry_idx >= len(stock_entries):
                fully_placed = False
                if verbose:
                    print(f"  WARNING: {pattern.pattern_id} entry={entry_idx} "
                          f"on {stock_id} — entry_index out of range")
                continue
 
            entry = stock_entries[entry_idx]
 
            in_window = any(
                ws <= start_pos + 1e-9 and end_pos <= we + 1e-9
                for ws, we in entry.windows
            )
            if not in_window:
                fully_placed = False
                if verbose:
                    print(f"  WARNING: {pattern.pattern_id} entry={entry_idx} "
                          f"at {start_pos:.0f} on {stock_id} "
                          f"— position not within any window of this entry")
                continue
 
            if placements[stock_id]:
                _, _, prev_start, prev_end = placements[stock_id][-1]
                if start_pos < prev_end - 1e-9:
                    fully_placed = False
                    if verbose:
                        print(f"  WARNING: {pattern.pattern_id} entry={entry_idx} "
                              f"at {start_pos:.0f} on {stock_id} "
                              f"— overlaps previous repetition ending at {prev_end:.0f}")
                    continue
 
            placements[stock_id].append((pattern, entry_idx, start_pos, end_pos))
 
    return placements, fully_placed
 
 
# COST EVALUATION
 
def evaluate(solution: Solution, placements: Dict, stocks: Dict,
             products: Dict) -> Tuple[float, Dict[str, int], Dict[str, int]]:
    """
    Evaluate a decoded solution.
 
    Setup cost is charged per contiguous run. A new run starts when:
      1. The pattern changes from the previous placement    
      2. The entry_index changes for the same pattern, implying a different
         window configuration and an interruption between runs
 
    Returns:
        total_cost   : sum of stock activation, setup and repetition costs
        unmet        : dict product_id -> shortfall (only products with unmet demand)
        overproduced : dict product_id -> excess (only products with overproduction)
    """
    total_cost = 0.0
    produced   = {pid: 0 for pid in products}
 
    for stock_id, stock_placements in placements.items():
        if not stock_placements:
            continue
 
        total_cost += stocks[stock_id].cost
 
        prev_pattern_id = None
        prev_entry_idx  = None
 
        for pattern, entry_idx, start, end in stock_placements:
 
            is_new_run = (
                prev_pattern_id is None
                or pattern.pattern_id != prev_pattern_id
                or entry_idx != prev_entry_idx
            )
 
            if is_new_run:
                total_cost += pattern.setup_cost
 
            stock_entries = pattern.stock_entries.get(stock_id, [])
            if entry_idx < len(stock_entries):
                total_cost += stock_entries[entry_idx].cost_per_rep
            else:
                print(f"  WARNING: no entry {entry_idx} for "
                      f"({pattern.pattern_id}, {stock_id}) — repetition cost skipped")
 
            for pid, qty in pattern.products_produced_per_rep.items():
                produced[pid] += qty
 
            prev_pattern_id = pattern.pattern_id
            prev_entry_idx  = entry_idx
 
    unmet = {
        pid: products[pid].demand - produced[pid]
        for pid in products
        if produced[pid] < products[pid].demand
    }
    overproduced = {
        pid: produced[pid] - products[pid].demand
        for pid in products
        if produced[pid] > products[pid].demand
    }
 
    return total_cost, unmet, overproduced
 
 
# UTILITIES
 
def copy_solution(sol: Solution) -> Solution:
    """
    copy of a solution.
    Copies the active dict structure only — Pattern references are kept
    as-is since Pattern objects are read-only and never modified by any algorithm.
    Significantly faster than copy.deepcopy() for large solutions.
    """
    new_sol = Solution()
    for stock_id, reps in sol.active.items():
        new_sol.active[stock_id] = list(reps)
    return new_sol
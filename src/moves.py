"""
moves.py
--------
Atomic neighborhood moves for the 1.5D CSP metaheuristics.
Provides the building blocks used by all local search methods
(First Improvement, Steepest Descent, Simulated Annealing, Tabu Search,
VNS and ILS).
 
Gap utilities:
  _gaps                   -- compute free intervals in a stock given current placements
  _first_feasible_in_gap  -- find first feasible position within a gap
  feasible_insertions     -- list all feasible insertion positions for a (pattern, entry)
 
Basic moves (one repetition at a time):
  remove      -- remove one repetition from a stock
  insert      -- add one repetition to an already open stock
  stock_open  -- activate a new stock with one initial repetition
  stock_close -- deactivate a stock by removing all its repetitions
  swap        -- replace one repetition with a different (pattern, entry)
  shift       -- move a repetition to a different position on the same stock
  close_open  -- close one stock and open another
 
Compound moves:
  remove_insert       -- remove a repetition and insert a new one in the freed space
  pattern_replace_all -- replace all repetitions of one pattern with another
  merge_stocks        -- move all repetitions from one stock onto another
"""
 
import copy
from typing import Optional, List, Tuple, Dict
from src.instance import Pattern, PatternStockEntry, Stock
from src.solution import Solution
 
 
# GAP UTILITIES
 
def _gaps(entries: list, stock_length: float, length_consumed: float) -> List[Tuple[float, float]]:
    """
    Compute all free gaps in a stock given its current sorted entries.
    A gap is a contiguous interval of free space where a new repetition
    could potentially be placed.
 
    Returns a list of (gap_start, gap_end) tuples.
    gap_end - gap_start >= length_consumed is not guaranteed — caller must check.
    """
    gaps = []
 
    first_start = entries[0][2] if entries else stock_length
    if first_start > 0:
        gaps.append((0.0, first_start))
 
    for i in range(len(entries) - 1):
        _, _, cur_start  = entries[i]
        cur_pat          = entries[i][0]
        next_start       = entries[i + 1][2]
        gap_start        = cur_start + cur_pat.length_consumed
        if next_start > gap_start + 1e-9:
            gaps.append((gap_start, next_start))
 
    if entries:
        last_pat, _, last_start = entries[-1]
        gap_start = last_start + last_pat.length_consumed
        if gap_start < stock_length - 1e-9:
            gaps.append((gap_start, stock_length))
 
    return gaps
 
 
def _first_feasible_in_gap(entry: PatternStockEntry, length_consumed: float,
                            gap_start: float, gap_end: float) -> Optional[float]:
    """
    Find the first feasible start position for a repetition within a gap,
    respecting the entry's windows.
 
    A position pos is feasible if:
        ws <= pos  AND  pos + length_consumed <= we
    for some window (ws, we) in entry.windows.
 
    Returns the first feasible position, or None if no position exists.
    """
    for ws, we in entry.windows:
        pos_start = max(gap_start, ws)
        pos_end   = min(gap_end, we) - length_consumed
 
        if pos_start <= pos_end + 1e-9:
            return pos_start
 
    return None
 
 
def feasible_insertions(pattern: Pattern, entry_idx: int,
                        stock_id: str, stock: Stock,
                        entries: list) -> List[float]:
    """
    Return all feasible start positions for one repetition of
    (pattern, entry_idx) on stock_id, given current entries.
 
    Checks every gap in the stock for feasible positions.
    Returns a list of feasible start positions (one per gap at most).
    """
    if stock_id not in pattern.stock_entries:
        return []
    if entry_idx >= len(pattern.stock_entries[stock_id]):
        return []
 
    entry           = pattern.stock_entries[stock_id][entry_idx]
    length_consumed = pattern.length_consumed
 
    current_reps = sum(
        1 for p, e, _ in entries
        if p.pattern_id == pattern.pattern_id and e == entry_idx
    )
    if current_reps >= entry.max_repetitions:
        return []
 
    positions = []
 
    for gap_start, gap_end in _gaps(entries, stock.length, length_consumed):
        pos = _first_feasible_in_gap(entry, length_consumed, gap_start, gap_end)
        if pos is not None:
            positions.append(pos)
 
    return positions
 
 
# BASIC MOVES
 
def remove(solution: Solution, stock_id: str, index: int,
           inplace: bool = False) -> Solution:
    """
    Remove the repetition at list index from stock_id.
    If stock becomes empty it is removed from active.
    """
    if not inplace:
        solution = copy.deepcopy(solution)
 
    solution.remove_repetition(stock_id, index)
    return solution
 
 
def insert(solution: Solution, stock_id: str, pattern: Pattern,
           entry_idx: int, start_pos: float,
           inplace: bool = False) -> Solution:
    """
    Insert one repetition of (pattern, entry_idx) at start_pos on stock_id.
    Stock must already be open — use stock_open for new stocks.
    Caller is responsible for ensuring start_pos is feasible.
    """
    if not inplace:
        solution = copy.deepcopy(solution)
 
    assert stock_id in solution.active, \
        f"Stock {stock_id} is not open — use stock_open instead"
 
    solution.add_repetition(stock_id, pattern, entry_idx, start_pos)
    return solution
 
 
def stock_open(solution: Solution, stock_id: str, pattern: Pattern,
               entry_idx: int, start_pos: float,
               inplace: bool = False) -> Solution:
    """
    Activate a new stock with one initial repetition of (pattern, entry_idx)
    at start_pos.
    Caller is responsible for ensuring start_pos is feasible.
    """
    if not inplace:
        solution = copy.deepcopy(solution)
 
    assert stock_id not in solution.active, \
        f"Stock {stock_id} is already open — use insert instead"
 
    solution.add_repetition(stock_id, pattern, entry_idx, start_pos)
    return solution
 
 
def stock_close(solution: Solution, stock_id: str,
                inplace: bool = False) -> Solution:
    """
    Deactivate a stock by removing all its repetitions.
    """
    if not inplace:
        solution = copy.deepcopy(solution)
 
    assert stock_id in solution.active, f"Stock {stock_id} is not open"
    solution.remove_stock(stock_id)
    return solution
 
 
def swap(solution: Solution, stock_id: str, index: int,
         new_pattern: Pattern, new_entry_idx: int, new_start_pos: float,
         inplace: bool = False) -> Solution:
    """
    Replace the repetition at index on stock_id with a new
    (pattern, entry_idx, start_pos).
    Caller is responsible for ensuring new_start_pos is feasible
    and does not overlap with other repetitions.
    """
    if not inplace:
        solution = copy.deepcopy(solution)
 
    solution.replace_repetition(stock_id, index, new_pattern,
                                new_entry_idx, new_start_pos)
    return solution
 
 
def shift(solution: Solution, stock_id: str, index: int,
          new_start_pos: float, inplace: bool = False) -> Solution:
    """
    Move the repetition at index to a new start position on the same stock,
    keeping the same (pattern, entry_idx).
    Caller is responsible for ensuring new_start_pos is feasible
    and does not overlap with other repetitions.
    """
    if not inplace:
        solution = copy.deepcopy(solution)
 
    pattern, entry_idx, _ = solution.get_repetition(stock_id, index)
    solution.replace_repetition(stock_id, index, pattern,
                                entry_idx, new_start_pos)
    return solution
 
 
def close_open(solution: Solution, stock_id_to_close: str,
               stock_id_to_open: str, pattern: Pattern,
               entry_idx: int, start_pos: float,
               inplace: bool = False) -> Solution:
    """
    Close one stock and open another with one initial repetition.
    Caller is responsible for ensuring start_pos is feasible on stock_id_to_open.
    """
    if not inplace:
        solution = copy.deepcopy(solution)
 
    assert stock_id_to_close in solution.active, \
        f"Stock {stock_id_to_close} is not open"
    assert stock_id_to_open not in solution.active, \
        f"Stock {stock_id_to_open} is already open"
 
    solution.remove_stock(stock_id_to_close)
    solution.add_repetition(stock_id_to_open, pattern, entry_idx, start_pos)
    return solution
 
 
# COMPOUND MOVES
 
def remove_insert(solution: Solution, stock_id: str, remove_index: int,
                  new_pattern: Pattern, new_entry_idx: int,
                  stocks: Dict[str, Stock],
                  inplace: bool = False) -> Optional[Solution]:
    """
    Remove the repetition at remove_index and insert a new repetition of
    (new_pattern, new_entry_idx) in the freed space.
 
    Returns the modified solution if a feasible insertion exists,
    or None if no feasible position is found after removal.
    """
    if not inplace:
        solution = copy.deepcopy(solution)
 
    solution.remove_repetition(stock_id, remove_index)
 
    stock     = stocks[stock_id]
    entries   = solution.active.get(stock_id, [])
    positions = feasible_insertions(
        new_pattern, new_entry_idx, stock_id, stock, entries
    )
 
    if not positions:
        return None
 
    solution.add_repetition(stock_id, new_pattern, new_entry_idx, positions[0])
    return solution
 
 
def pattern_replace_all(solution: Solution, stock_id: str,
                        old_pattern: Pattern, new_pattern: Pattern,
                        stocks: Dict[str, Stock],
                        inplace: bool = False) -> Optional[Solution]:
    """
    Replace every repetition of old_pattern on stock_id with new_pattern.
    new_pattern must have the same length_consumed as old_pattern.
 
    For each repetition of old_pattern, finds a feasible position for
    new_pattern anywhere within the same gap (not necessarily the exact
    same start position). Returns None if any replacement is infeasible.
    """
    if old_pattern.length_consumed != new_pattern.length_consumed:
        return None
 
    if stock_id not in solution.active:
        return None
 
    if stock_id not in new_pattern.stock_entries:
        return None
 
    entries = solution.active[stock_id]
    stock   = stocks[stock_id]
 
    targets = [
        i for i, (pat, eidx, start_pos) in enumerate(entries)
        if pat.pattern_id == old_pattern.pattern_id
    ]
 
    if not targets:
        return None
 
    sol_copy = copy.deepcopy(solution)
 
    offset = 0
    for i in targets:
        idx     = i + offset
        entries = sol_copy.active[stock_id]
 
        left_end    = entries[idx - 1][2] + entries[idx - 1][0].length_consumed \
                      if idx > 0 else 0.0
        right_start = entries[idx + 1][2] \
                      if idx + 1 < len(entries) else stock.length
 
        feasible_entry = None
        feasible_pos   = None
        for eidx, entry in enumerate(new_pattern.stock_entries[stock_id]):
            for ws, we in entry.windows:
                pos_start = max(left_end, ws)
                pos_end   = min(right_start, we) - new_pattern.length_consumed
                if pos_start <= pos_end + 1e-9:
                    feasible_entry = eidx
                    feasible_pos   = pos_start
                    break
            if feasible_entry is not None:
                break
 
        if feasible_entry is None:
            return None
 
        sol_copy.replace_repetition(
            stock_id, idx, new_pattern, feasible_entry, feasible_pos
        )
 
    if inplace:
        solution.active = sol_copy.active
        return solution
 
    return sol_copy
 
 
def merge_stocks(solution: Solution, stock_id_donor: str,
                 stock_id_receiver: str, stocks: Dict[str, Stock],
                 inplace: bool = False) -> Optional[Solution]:
    """
    Move all repetitions from stock_id_donor onto stock_id_receiver,
    then close the donor.
 
    For each repetition on the donor, finds a feasible insertion position
    on the receiver using the existing gaps. Attempts to place repetitions
    in start_pos order to minimise conflicts.
    Returns None if any repetition cannot be placed on the receiver.
    """
    if stock_id_donor not in solution.active:
        return None
    if stock_id_receiver not in solution.active:
        return None
 
    donor_entries = list(solution.active[stock_id_donor])
    stock_recv    = stocks[stock_id_receiver]
 
    sol_copy = copy.deepcopy(solution)
 
    for pat, eidx, start_pos in donor_entries:
        if stock_id_receiver not in pat.stock_entries:
            return None
 
        recv_entries = sol_copy.active[stock_id_receiver]
        placed       = False
 
        for recv_eidx, recv_entry in enumerate(pat.stock_entries[stock_id_receiver]):
            positions = feasible_insertions(
                pat, recv_eidx, stock_id_receiver, stock_recv, recv_entries
            )
            if positions:
                sol_copy.add_repetition(
                    stock_id_receiver, pat, recv_eidx, positions[0]
                )
                placed = True
                break
 
        if not placed:
            return None
 
    sol_copy.remove_stock(stock_id_donor)
 
    if inplace:
        solution.active = sol_copy.active
        return solution
 
    return sol_copy
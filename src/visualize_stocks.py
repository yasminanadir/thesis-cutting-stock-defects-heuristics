"""
visualize_solution.py
---------------------
Matplotlib visualization of the pattern-based CSP solution.
Width on x-axis, length on y-axis.
 
Functions:
  visualize_solution        -- overview visualization, one panel per active stock
  visualize_solution_detail -- detailed view of one stock with pattern and product labels
 
Usage in notebook:
    from src.visualize_solution import visualize_solution, visualize_solution_detail
    visualize_solution(solution, stocks, products, title="Pattern-first greedy")
    visualize_solution_detail(solution, "S2", stocks, products)
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import permutations, islice
from typing import Dict
import os

import json
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.solution import Solution, decode, evaluate
# ---------------------------------------------------------------------------
# Color palette — one color per product, consistent across all stocks
# ---------------------------------------------------------------------------

PRODUCT_PALETTE = [
    "#4E79A7", "#F28E2B", "#240E86", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#499894",
    "#86BCB6", "#D37295", "#8CD17D", "#B6992D", "#A0CBE8",
]

MAX_PERMS = 1000  # max permutations to try before fallback


def _try_place_strips(strips, free_zones):
    result   = []
    for pid, strip_width in strips:
        placed = False
        for zi in range(len(free_zones)):        # fresh scan per strip
            z_start, z_end = free_zones[zi]
            if z_end - z_start >= strip_width - 1e-9:
                result.append((pid, z_start, z_start + strip_width))
                # shrink this zone
                free_zones[zi] = (z_start + strip_width, z_end)
                placed = True
                break
        if not placed:
            return None
    return result


def _assign_strip_positions(pattern, stock, products, entry_idx, start_y=None, end_y=None):
    """
    Assign lateral (x/width) positions to strips in a pattern.
    Uses defects that fall within the actual repetition's length range.
    Tries permutations of strips to avoid defects.
    Falls back to naive left-to-right if no permutation fits.
    """
    strips = []
    for pid, qty in pattern.products_produced_per_rep.items():
        for _ in range(qty):
            strips.append((pid, products[pid].width))

    entry = pattern.stock_entries[stock.stock_id][entry_idx]

    # use actual repetition bounds if provided, else full window
    if start_y is not None and end_y is not None:
        rep_start = start_y
        rep_end   = end_y
    else:
        rep_start = min(ws for ws, we in entry.windows)
        rep_end   = max(we for ws, we in entry.windows)

    # only defects overlapping THIS repetition's length range
    defects_in = [
        d for d in stock.defects
        if d.start_in_length < rep_end
        and d.start_in_length + d.length > rep_start
    ]

    defect_intervals = sorted(
        [(d.start_in_width, d.start_in_width + d.width)
         for d in defects_in]
    )
    free_zones = []
    cursor     = 0.0
    for d_start, d_end in defect_intervals:
        if cursor < d_start:
            free_zones.append((cursor, d_start))
        cursor = max(cursor, d_end)
    if cursor < stock.width:
        free_zones.append((cursor, stock.width))

    if not free_zones:
        free_zones = [(0.0, stock.width)]

    # try permutations of strips to find one that avoids defects
    seen = set()
    for perm in islice(permutations(strips), MAX_PERMS):
        key = tuple(w for _, w in perm)
        if key in seen:
            continue
        seen.add(key)

        result = _try_place_strips(perm, list(free_zones))
        if result is not None:
            return result

    # fallback — naive left to right ignoring defects
    result = []
    x      = 0.0
    for pid, strip_width in strips:
        result.append((pid, x, x + strip_width))
        x += strip_width
    return result


def visualize_solution(
    solution,
    stocks: Dict,
    products: Dict,
    title: str = "Solution Visualization",
    max_cols: int = 3,
    figsize_per_stock=(4, 6),
    show: bool = True,
    save_path: str = None,
):
    """
    Overview visualization — one panel per active stock.
    Width on x-axis, length on y-axis.
    """
    placements, fully_placed = decode(solution, stocks)

    product_ids   = sorted(products.keys())
    product_color = {
        pid: PRODUCT_PALETTE[i % len(PRODUCT_PALETTE)]
        for i, pid in enumerate(product_ids)
    }

    active_stocks = list(solution.active.keys())
    if not active_stocks:
        print("No active stocks in solution.")
        return

    n      = len(active_stocks)
    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_stock[0] * n_cols, figsize_per_stock[1] * n_rows)
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    legend_handles = {}

    for idx, stock_id in enumerate(active_stocks):
        row   = idx // n_cols
        col   = idx % n_cols
        ax    = axes[row][col]
        stock = stocks[stock_id]

        ax.add_patch(mpatches.Rectangle(
            (0, 0), stock.width, stock.length,
            linewidth=1.5, edgecolor='#2c3e50', facecolor='#ecf0f1'
        ))

        for defect in stock.defects:
            ax.add_patch(mpatches.Rectangle(
                (defect.start_in_width, defect.start_in_length),
                defect.width, defect.length,
                linewidth=0.5, edgecolor='#c0392b',
                facecolor='#e74c3c', alpha=0.85, zorder=5
            ))

        stock_placements    = placements.get(stock_id, [])
        pattern_strip_cache = {}

        for pattern, entry_idx, start_y, end_y in stock_placements:
            # include start_y in cache key — different repetitions may have
            # different defects and need different lateral arrangements
            cache_key = (pattern.pattern_id, entry_idx, start_y)

            if cache_key not in pattern_strip_cache:
                strip_positions = _assign_strip_positions(
                    pattern, stock, products, entry_idx,
                    start_y=start_y, end_y=end_y
                )
                pattern_strip_cache[cache_key] = strip_positions
            else:
                strip_positions = pattern_strip_cache[cache_key]

            for prod_id, x0, x1 in strip_positions:
                color = product_color[prod_id]
                ax.add_patch(mpatches.Rectangle(
                    (x0, start_y), x1 - x0, end_y - start_y,
                    linewidth=0.5, edgecolor='white',
                    facecolor=color, alpha=0.85, zorder=3
                ))

                strip_w = x1 - x0
                strip_h = end_y - start_y
                if strip_w > stock.width * 0.08 and strip_h > stock.length * 0.04:
                    ax.text(
                        (x0 + x1) / 2, (start_y + end_y) / 2,
                        prod_id,
                        ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold', zorder=6
                    )

                if prod_id not in legend_handles:
                    legend_handles[prod_id] = mpatches.Patch(
                        facecolor=color, alpha=0.85, label=prod_id
                    )

        ax.set_xlim(0, stock.width)
        ax.set_ylim(0, stock.length)
        ax.set_title(
            f"{stock_id}  W={stock.width:.2f}m  L={stock.length:.0f}mm  "
            f"defects={len(stock.defects)}  cost={stock.cost:.0f}",
            fontsize=8, pad=3
        )
        ax.set_xlabel("Width (m)", fontsize=7)
        ax.set_ylabel("Length (mm)", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    legend_elements = (
        list(legend_handles.values()) +
        [mpatches.Patch(facecolor='#e74c3c', alpha=0.85, label='Defect'),
         mpatches.Patch(facecolor='#ecf0f1', edgecolor='#2c3e50', label='Stock')]
    )
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=min(len(legend_elements), 6),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02)
    )

    feasibility_str = "✓ Fully placed" if fully_placed else "⚠ Not fully placed"
    plt.suptitle(
        f"{title}  —  {feasibility_str}  |  {len(active_stocks)} stocks used",
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    return fig
 
 
def visualize_solution_detail(
    solution,
    stock_id: str,
    stocks: Dict,
    products: Dict,
    title: str = None,
):
    """
    Detailed view of one stock in the solution.
    Width on x-axis, length on y-axis.
    Shows all pattern repetitions with product and pattern labels.
    """
    placements, fully_placed = decode(solution, stocks)
 
    product_ids   = sorted(products.keys())
    product_color = {
        pid: PRODUCT_PALETTE[i % len(PRODUCT_PALETTE)]
        for i, pid in enumerate(product_ids)
    }
 
    stock   = stocks[stock_id]
    fig, ax = plt.subplots(figsize=(6, 10))
 
    ax.add_patch(mpatches.Rectangle(
        (0, 0), stock.width, stock.length,
        linewidth=2, edgecolor='#2c3e50', facecolor='#f8f9fa'
    ))
 
    for defect in stock.defects:
        ax.add_patch(mpatches.Rectangle(
            (defect.start_in_width, defect.start_in_length),
            defect.width, defect.length,
            linewidth=1, edgecolor='#c0392b',
            facecolor='#e74c3c', alpha=0.9, zorder=5
        ))
        ax.text(
            defect.start_in_width + defect.width / 2,
            defect.start_in_length + defect.length / 2,
            "D", ha='center', va='center',
            fontsize=7, color='white', fontweight='bold', zorder=6
        )
 
    stock_placements    = placements.get(stock_id, [])
    pattern_strip_cache = {}
    legend_handles      = {}
 
    for pattern, entry_idx, start_y, end_y in stock_placements:
        cache_key = (pattern.pattern_id, entry_idx)
 
        if cache_key not in pattern_strip_cache:
            strip_positions = _assign_strip_positions(
                pattern, stock, products, entry_idx
            )
            pattern_strip_cache[cache_key] = strip_positions
        else:
            strip_positions = pattern_strip_cache[cache_key]
 
        for prod_id, x0, x1 in strip_positions:
            color = product_color[prod_id]
            ax.add_patch(mpatches.Rectangle(
                (x0, start_y), x1 - x0, end_y - start_y,
                linewidth=0.5, edgecolor='white',
                facecolor=color, alpha=0.85, zorder=3
            ))
            ax.text(
                (x0 + x1) / 2, (start_y + end_y) / 2,
                prod_id,
                ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=6
            )
 
            if prod_id not in legend_handles:
                legend_handles[prod_id] = mpatches.Patch(
                    facecolor=color, alpha=0.85, label=prod_id
                )
 
        ax.text(
            stock.width * 1.01, (start_y + end_y) / 2,
            f"{pattern.pattern_id} e{entry_idx}",
            ha='left', va='center',
            fontsize=6, color='#555555', zorder=6
        )
 
        ax.axhline(
            y=end_y, color='white', linewidth=0.8,
            linestyle='--', alpha=0.5, zorder=4
        )
 
    ax.set_xlim(0, stock.width * 1.12)
    ax.set_ylim(0, stock.length)
    ax.set_xlabel("Width (m)", fontsize=10)
    ax.set_ylabel("Length (mm)", fontsize=10)
    ax.set_title(
        title or (
            f"{stock_id}  —  W={stock.width:.3f}m  L={stock.length:.0f}mm  "
            f"Defects={len(stock.defects)}  Cost={stock.cost:.2f}"
        ),
        fontsize=11, fontweight='bold'
    )
 
    legend_elements = (
        list(legend_handles.values()) +
        [mpatches.Patch(facecolor='#e74c3c', alpha=0.9, label='Defect')]
    )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()

def visualize_solution_to_pdf(json_path, instance, output_pdf, stocks_per_row=3, stocks_per_page=6):
    """Load a solution JSON and save a styled PDF with summary + stock visualizations."""
    name     = instance['name']
    stocks   = instance['stocks']
    products = instance['products']
    patterns = instance['patterns']

    with open(json_path, 'r') as f:
        payload = json.load(f)

    pattern_lookup = {pat.pattern_id: pat for pat in patterns}
    sol = Solution()
    for stock_id, reps in payload['active'].items():
        for pattern_id, entry_idx, start_pos in reps:
            pat = pattern_lookup.get(pattern_id)
            if pat is None:
                continue
            sol.add_repetition(stock_id, pat, entry_idx, start_pos)

    placements, _ = decode(sol, stocks)
    cost, unmet, overprod = evaluate(sol, placements, stocks, products)
    is_feasible = (sum(unmet.values()) == 0)

    with PdfPages(output_pdf) as pdf:
        # page 1 — summary
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        summary_text = (
            f"Instance   : {name}\n"
            f"Method     : {payload.get('method', 'N/A')}\n"
            f"Cost       : {cost:.4f}\n"
            f"Feasible   : {is_feasible}\n"
            f"Unmet      : {sum(unmet.values())}  {dict(unmet)}\n"
            f"Overprod   : {sum(overprod.values())}  {dict(overprod)}\n"
            f"Elapsed    : {payload.get('elapsed_sec', 'N/A')}s\n"
            f"Open stocks: {len(sol.active)}\n"
        )
        ax.text(0.05, 0.95, summary_text,
                transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace')
        ax.set_title(f"Solution Summary — {name}", fontsize=14, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # pages 2+ — stocks in chunks
        active_stocks = list(sol.active.keys())
        for i in range(0, len(active_stocks), stocks_per_page):
            chunk = active_stocks[i:i + stocks_per_page]
            sol_chunk = copy.copy(sol)
            sol_chunk.active = {sid: sol.active[sid] for sid in chunk}
            fig = visualize_solution(
                sol_chunk, stocks, products,
                title    = f"{name} — cost={cost:.2f}  feasible={is_feasible}",
                max_cols = stocks_per_row,
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  Saved: {output_pdf}")
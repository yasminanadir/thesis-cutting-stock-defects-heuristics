import time
import os
import json
import csv
import importlib
import numpy as np
import pandas as pd
import src.solution
importlib.reload(src.solution)
import src.constructive
importlib.reload(src.constructive)
import src.metaheuristic
importlib.reload(src.metaheuristic)
from src.solution import Solution, decode, evaluate
from src.constructive import (
    greedy_pattern_first_scarcity2, multistart_greedy, grasp
)
from src.metaheuristic import (
    FirstImprovement, SteepestImprovement,
    SimulatedAnnealing, VNS, IteratedLocalSearch, TabuSearch, SimulatedAnnealingAdaptiveAlpha
)

SOLUTIONS_DIR   = r"C:\path\to\your\thesis\outputs\solutions_convergence"
CONVERGENCE_DIR = r"C:\path\to\your\thesis\outputs\convergence_logs"

os.makedirs(SOLUTIONS_DIR,   exist_ok=True)
os.makedirs(CONVERGENCE_DIR, exist_ok=True)

TARGET_CATEGORIES = [
    "sweep1_High", "sweep1_Low",
    "sweep2_Loose", "sweep2_Tight",
    "sweep3_Short", "sweep3_Long", "sweep4_High",
    "sweep5_XS", "sweep5_S",
]

def _conv_path(instance_name, method, start_method, time_limit):
    """Return convergence CSV path and delete any existing file to start fresh."""
    fname = (f"{instance_name}_{method}_{start_method}"
             f"_TL{int(time_limit)}_convergence.csv")
    fpath = os.path.join(CONVERGENCE_DIR, fname)
    if os.path.exists(fpath):
        os.remove(fpath)
    return fpath


# ---------------------------------------------------------------------------
# Convergence log helpers  ← NEW
# ---------------------------------------------------------------------------

def save_convergence_csv(convergence_log, greedy_ref, instance_name, method,
                         start_method, time_limit, output_dir=CONVERGENCE_DIR):
    if not convergence_log:
        return
    fname = (f"{instance_name}_{method}_{start_method}"
             f"_TL{int(time_limit)}_convergence.csv")
    fpath = os.path.join(output_dir, fname)
    with open(fpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['elapsed_sec', 'best_cost', 'improvement_pct'])
        for elapsed, cost in convergence_log:
            impr = ((greedy_ref - cost) / greedy_ref * 100
                    if greedy_ref and greedy_ref > 0 else None)
            writer.writerow([
                elapsed,
                round(cost, 4),
                round(impr, 4) if impr is not None else ''
            ])


def convergence_exists(instance_name, method, start_method, time_limit,
                       output_dir=CONVERGENCE_DIR):
    fname = (f"{instance_name}_{method}_{start_method}"
             f"_TL{int(time_limit)}_convergence.csv")
    return os.path.exists(os.path.join(output_dir, fname))


# ---------------------------------------------------------------------------
# Solution helpers — identical to runner.py
# ---------------------------------------------------------------------------

def save_solution_json(
    sol, cost, is_feasible, unmet, overprod, elapsed_sec,
    instance_name, method, start_method, time_limit,
    output_dir=SOLUTIONS_DIR,
):
    os.makedirs(output_dir, exist_ok=True)
    fname = (f"{instance_name}_{method}_{start_method}"
             f"_TL{int(time_limit)}.json")
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


def solution_exists(instance_name, method, start_method, time_limit,
                    output_dir=SOLUTIONS_DIR):
    fname = (f"{instance_name}_{method}_{start_method}"
             f"_TL{int(time_limit)}.json")
    return os.path.exists(os.path.join(output_dir, fname))


def load_solution_json(fpath):
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_solution(payload, patterns):
    pattern_lookup = {pat.pattern_id: pat for pat in patterns}
    sol = Solution()
    for stock_id, reps in payload["active"].items():
        for pattern_id, entry_idx, start_pos in reps:
            pat = pattern_lookup.get(pattern_id)
            if pat is None:
                raise ValueError(
                    f"Pattern {pattern_id} not found — "
                    f"make sure you are using the correct instance."
                )
            sol.add_repetition(stock_id, pat, entry_idx, start_pos)
    return sol


# ---------------------------------------------------------------------------
# Construction — identical to runner.py
# ---------------------------------------------------------------------------

def build_start_solution(start_method, patterns, stocks, products,
                         config, seed):
    t0 = time.time()

    if start_method == "greedy":
        sol = greedy_pattern_first_scarcity2(patterns, stocks, products)
    elif start_method == "multistart":
        sol = multistart_greedy(
            patterns, stocks, products,
            n_starts=config["multistart"]["n_starts"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown start_method: {start_method}")

    construction_time = time.time() - t0
    return sol, construction_time


# ---------------------------------------------------------------------------
# Generic method runner — identical to runner.py + convergence log saving
# ---------------------------------------------------------------------------

def run_method(
    method_name,
    start_method,
    instance,
    config,
    time_limit,
    seed,
    greedy_ref    = None,   # ← NEW: needed to compute improvement_pct in CSV
    skip_existing = True,
    verbose       = False,
):
    name     = instance["name"]
    stocks   = instance["stocks"]
    products = instance["products"]
    patterns = instance["patterns"]

    # --- skip if already done ---
    if skip_existing and solution_exists(name, method_name, start_method,
                                         time_limit)\
        and convergence_exists(name, method_name, start_method, time_limit):
        if verbose:
            print(f"  [skip] {name} {method_name} {start_method} "
                  f"TL{int(time_limit)} — already exists")
        fname = (f"{name}_{method_name}_{start_method}"
                 f"_TL{int(time_limit)}.json")
        fpath = os.path.join(SOLUTIONS_DIR, fname)
        with open(fpath, "r") as f:
            payload = json.load(f)
        return {
            "instance"    : name,
            "method"      : method_name,
            "start_method": start_method,
            "time_limit"  : time_limit,
            "start_cost"  : None,
            "final_cost"  : payload["cost"],
            "unmet"       : sum(payload["unmet"].values())
                            if isinstance(payload["unmet"], dict)
                            else payload["unmet"],
            "overprod"    : sum(payload["overprod"].values())
                            if isinstance(payload["overprod"], dict)
                            else payload["overprod"],
            "open_stocks" : len(payload["active"]),
            "is_feasible" : payload["is_feasible"],
            "elapsed_sec" : payload["elapsed_sec"],
            "seed"        : seed,
            "error"       : None,
        }

    try:
        ls = None  # ← track ls object for convergence log

        # --- construction ---
        if method_name in ("greedy", "multistart"):
            t0 = time.time()

            if method_name == "greedy":
                sol = greedy_pattern_first_scarcity2(patterns, stocks, products)
            else:
                sol = multistart_greedy(
                    patterns, stocks, products,
                    n_starts=config["multistart"]["n_starts"],
                    seed=seed,
                )

            elapsed = round(time.time() - t0, 2)

            placements, _ = decode(sol, stocks)
            cost, unmet, overprod = evaluate(sol, placements, stocks, products)
            total_unmet    = sum(unmet.values())
            total_overprod = sum(overprod.values())
            is_feasible    = (total_unmet == 0)
            start_cost     = cost
            final_cost     = cost

        elif method_name in ("GRASP", "GRASP_SA"):
            t0  = time.time()
            cfg = config[method_name]

            sol, _ = grasp(
                patterns, stocks, products,
                n_restarts              = cfg["n_restarts"],
                alpha                   = cfg["alpha"],
                run_local_search        = True,
                sd_time_limit           = cfg.get("sd_time_limit", 0.0),
                fi_time_limit           = cfg.get("fi_time_limit", 30.0),
                repair_time_limit       = cfg.get("repair_time_limit", 5.0),
                sd_active_moves         = cfg["sd_active_moves"],
                fi_neighborhood_weights = {},
                outer_method            = cfg.get("outer_method", "FI"),
                T_init                  = cfg.get("T_init", 100.0),
                sa_neighborhood_weights = cfg.get("sa_neighborhood_weights", None),
                seed                    = seed,
                time_limit              = time_limit,
                verbose                 = verbose,
                convergence_csv_path = _conv_path(name, method_name, start_method, time_limit),
                log_interval         = 0.1,
            )

            elapsed = round(time.time() - t0, 2)

            placements, _ = decode(sol, stocks)
            cost, unmet, overprod = evaluate(sol, placements, stocks, products)
            total_unmet    = sum(unmet.values())
            total_overprod = sum(overprod.values())
            is_feasible    = (total_unmet == 0)
            start_cost     = None
            final_cost     = cost

        else:
            start_sol, construction_time = build_start_solution(
                start_method, patterns, stocks, products, config, seed
            )

            placements, _ = decode(start_sol, stocks)
            start_cost_val, start_unmet, _ = evaluate(
                start_sol, placements, stocks, products
            )
            start_cost = start_cost_val

            if sum(start_unmet.values()) > 0:
                t_repair = time.time()
                fi_repair = FirstImprovement(stocks, products, patterns,
                                            max_iterations = 999999,
                                            time_limit     = 30.0,
                                            verbose        = False)
                start_sol, _ = fi_repair.repair(start_sol,
                                                max_repair_iterations = 999999,
                                                time_limit            = 30.0)
                construction_time += time.time() - t_repair
                placements, _ = decode(start_sol, stocks)
                start_cost_val, start_unmet, _ = evaluate(start_sol, placements, stocks, products)
                start_cost = start_cost_val

            CONSTRUCTION_THRESHOLD = 50.0
            if construction_time > CONSTRUCTION_THRESHOLD:
                ls_time_limit = time_limit
            else:
                ls_time_limit = max(1.0, time_limit - construction_time)

            t_ls = time.time()
            ts_config_info = {}

            if method_name == "FI":
                cfg = config["FI"]
                ls  = FirstImprovement(
                    stocks, products, patterns,
                    max_iterations       = 999999,
                    time_limit           = ls_time_limit,
                    neighborhood_weights = cfg["neighborhood_weights"],
                    verbose              = verbose,
                    seed                 = seed,
                )
                ls.convergence_csv_path = _conv_path(name, method_name, start_method, time_limit)

            elif method_name == "SD":
                cfg = config["SD"]
                ls  = SteepestImprovement(
                    stocks, products, patterns,
                    time_limit   = ls_time_limit,
                    active_moves = cfg["active_moves"],
                    verbose      = verbose,
                    seed         = seed,
                )
                ls.convergence_csv_path = _conv_path(name, method_name, start_method, time_limit)
                sol, _, _, _, _ = ls.run(start_sol)

            elif method_name == "SA":
                cfg = config["SA"]
                ls  = SimulatedAnnealingAdaptiveAlpha(
                    stocks, products, patterns,
                    time_limit           = ls_time_limit,
                    T_init               = cfg["T_init"],
                    T_min                = cfg.get("T_min", 1e-3),
                    neighborhood_weights = cfg["neighborhood_weights"],
                    verbose              = verbose,
                    seed                 = seed,
                )
                ls.convergence_csv_path = _conv_path(name, method_name, start_method, time_limit)
                sol, _, _, _ = ls.run(start_sol)

            elif method_name in ("VNS", "VNS_SA"):
                cfg = config[method_name]
                ls  = VNS(
                    stocks, products, patterns,
                    time_limit              = ls_time_limit,
                    fi_max_iterations       = cfg["fi_max_iterations"],
                    sd_time_limit           = min(15.0, ls_time_limit * 0.25),
                    fi_time_limit           = min(45.0, ls_time_limit * 0.75),
                    fi_neighborhood_weights = cfg.get("fi_neighborhood_weights", None),
                    sd_active_moves         = cfg.get("sd_active_moves", {"relocate"}),
                    n2_method               = cfg.get("n2_method", "FI"),
                    T_init                  = cfg.get("T_init", 100.0),
                    alpha                   = cfg.get("alpha", 0.9999),
                    sa_neighborhood_weights = cfg.get("sa_neighborhood_weights", None),
                    verbose                 = verbose,
                    seed                    = seed,
                )
                ls.convergence_csv_path = _conv_path(name, method_name, start_method, time_limit)
                sol, _, _ = ls.run(start_sol)

            elif method_name in ("ILS", "ILS_SA"):
                cfg = config[method_name]
                ls  = IteratedLocalSearch(
                    stocks, products, patterns,
                    time_limit              = ls_time_limit,
                    local_search_method     = cfg.get("local_search_method", "SD"),
                    init_ls_method          = cfg.get("init_ls_method", "SD"),
                    local_search_time       = cfg.get("local_search_time", min(20.0, ls_time_limit * 0.3)),
                    local_search_iterations = cfg.get("local_search_iterations", 9999),
                    init_ls_time            = cfg.get("init_ls_time", min(20.0, ls_time_limit * 0.3)),
                    init_ls_iterations      = cfg.get("init_ls_iterations", 9999),
                    active_moves            = cfg.get("active_moves", {"relocate"}),
                    T_init                  = cfg.get("T_init", 100.0),
                    alpha                   = cfg.get("alpha", 0.9999),
                    sa_neighborhood_weights = cfg.get("sa_neighborhood_weights", None),
                    perturb_k               = cfg["perturb_k"],
                    perturb_k_max           = cfg.get("perturb_k_max", 3),
                    patience                = cfg["patience"],
                    seed                    = seed,
                    verbose                 = verbose,
                )
                ls.convergence_csv_path = _conv_path(name, method_name, start_method, time_limit)
                sol, _, _ = ls.run(start_sol)

            elif method_name == "TS":
                cfg = config["TS"]
                ls  = TabuSearch(
                    stocks, products, patterns,
                    time_limit     = ls_time_limit,
                    tabu_tenure    = cfg.get("tabu_tenure", 15),
                    max_iterations = cfg.get("max_iterations", 9999),
                    active_moves   = cfg.get("active_moves", {"relocate", "stock_reset"}),
                    seed           = seed,
                    verbose        = verbose,
                )
                ls.convergence_csv_path = _conv_path(name, method_name, start_method, time_limit)
                sol, _, _, _, _ = ls.run(start_sol)
                ts_config_info = {
                    "tabu_tenure"   : ls.tabu_tenure,
                    "active_moves"  : str(ls.active_moves),
                    "max_iterations": ls.max_iterations,
                    "time_limit"    : ls.time_limit,
                }

            else:
                raise ValueError(f"Unknown method: {method_name}")

            elapsed_ls = time.time() - t_ls
            elapsed    = round(construction_time + elapsed_ls, 2)

            placements, _ = decode(sol, stocks)
            cost, unmet, overprod = evaluate(sol, placements, stocks, products)
            total_unmet    = sum(unmet.values())
            total_overprod = sum(overprod.values())
            is_feasible    = (total_unmet == 0)
            final_cost     = cost

        # --- add final convergence point ← NEW ---
        if ls is not None and hasattr(ls, 'convergence_log'):
            ls._log_convergence(elapsed, final_cost)

        # --- save solution to disk ---
        save_solution_json(
            sol           = sol,
            cost          = final_cost,
            is_feasible   = is_feasible,
            unmet         = {k: int(v) for k, v in unmet.items()},
            overprod      = {k: int(v) for k, v in overprod.items()},
            elapsed_sec   = elapsed,
            instance_name = name,
            method        = method_name,
            start_method  = start_method,
            time_limit    = time_limit,
        )

        row = {
            "instance"    : name,
            "method"      : method_name,
            "start_method": start_method,
            "time_limit"  : time_limit,
            "start_cost"  : round(start_cost, 2) if start_cost else None,
            "final_cost"  : round(final_cost, 2),
            "unmet"       : total_unmet,
            "overprod"    : total_overprod,
            "open_stocks" : len(sol.active),
            "is_feasible" : is_feasible,
            "elapsed_sec" : elapsed,
            "seed"        : seed,
            "error"       : None,
        }

        if method_name == "SA":
            row.update({
                "T_init"              : cfg["T_init"],
                "T_min"               : cfg.get("T_min", 1e-3),
                "alpha_estimated"     : True,
                "neighborhood_weights": str(cfg["neighborhood_weights"]),
            })
        if method_name in ("GRASP", "GRASP_SA"):
            row.update({
                "alpha_grasp"      : cfg["alpha"],
                "n_restarts"       : cfg["n_restarts"],
                "outer_method"     : cfg.get("outer_method", "FI"),
                "fi_time_limit"    : cfg.get("fi_time_limit", 30.0),
                "repair_time_limit": cfg.get("repair_time_limit", 5.0),
                "sd_active_moves"  : str(cfg["sd_active_moves"]),
                "sa_T_init"        : cfg.get("T_init", None),
                "sa_weights"       : str(cfg.get("sa_neighborhood_weights", None)),
            })
        if method_name == "FI":
            row.update({"neighborhood_weights": str(cfg["neighborhood_weights"])})
        if method_name == "SD":
            row.update({"active_moves": str(cfg["active_moves"])})
        if method_name in ("ILS", "ILS_SA"):
            row.update({
                "local_search_method"    : cfg.get("local_search_method", "SD"),
                "T_init"                 : cfg.get("T_init", None),
                "alpha"                  : cfg.get("alpha", None),
                "perturb_k"              : cfg["perturb_k"],
                "perturb_k_max"          : cfg.get("perturb_k_max", 3),
                "patience"               : cfg["patience"],
                "local_search_time"      : cfg.get("local_search_time", None),
                "sa_neighborhood_weights": str(cfg.get("sa_neighborhood_weights", None)),
            })
        if method_name == "TS":
            row.update({
                "tabu_tenure"   : cfg.get("tabu_tenure", 15),
                "active_moves"  : str(cfg.get("active_moves", None)),
                "max_iterations": cfg.get("max_iterations", 9999),
            })
        if method_name in ("VNS", "VNS_SA"):
            row.update({
                "n1_active_moves": str(ls.sd_active_moves),
                "n1_time"        : ls.sd_time_limit,
                "n2_method"      : ls.n2_method,
                "n2_time"        : ls.fi_time_limit,
                "n2_alpha"       : ls.alpha if ls.n2_method == "SA" else None,
                "n2_T_init"      : ls.T_init if ls.n2_method == "SA" else None,
                "n2_weights"     : str(ls.sa_neighborhood_weights)
                                   if ls.n2_method == "SA"
                                   else str(ls.fi_neighborhood_weights),
            })
        if method_name == "TS":
            row.update(ts_config_info)

    except Exception as e:
        print(f"  ERROR: {name} {method_name} {start_method} — {e}")
        row = {
            "instance"    : name,
            "method"      : method_name,
            "start_method": start_method,
            "time_limit"  : time_limit,
            "start_cost"  : None,
            "final_cost"  : None,
            "unmet"       : None,
            "overprod"    : None,
            "open_stocks" : None,
            "is_feasible" : False,
            "elapsed_sec" : None,
            "seed"        : seed,
            "error"       : str(e),
        }

    return row


METHODS_CONFIG = {
    "greedy"     : {"has_ls": False},
    "multistart" : {"has_ls": False, "n_starts": 10},
    "FI"         : {
        "has_ls": True,
        "neighborhood_weights": {
            "remove": 1.0, "swap": 2.0, "relocate": 4.0,
            "stock_reset": 2.0, "pattern_replace_all": 0.5
        },
    },
    "SD"  : {"has_ls": True, "active_moves": {"relocate", "stock_reset", "swap", "pattern_replace_all", "remove"}},
    "SA"  : {
        "has_ls": True, "T_init": 100, "T_min": 1e-3,
        "neighborhood_weights": {
            "remove": 1.0, "swap": 2.0,
            "relocate": 4.0, "stock_reset": 3.0, "pattern_replace_all": 1.0,
            "insert": 1.0, "stock_open": 1.0,
        },
    },

    "GRASP": {
        "has_ls"         : True,
        "alpha"          : 0.1,
        "n_restarts"     : 30,
        "sd_active_moves": {"relocate"},
    },
    "VNS_SA": {
        "has_ls"                 : True,
        "fi_max_iterations"      : 9999,
        "n2_method"              : "SA",
        "T_init"                 : 100.0,
        "alpha"                  : 0.9999,
        "sd_active_moves"        : {"relocate", "stock_reset"},
        "sa_neighborhood_weights": {
            "remove": 1.0, "swap": 2.0,
            "relocate": 4.0, "stock_reset": 3.0, "pattern_replace_all": 1.0,
            "insert": 1.0, "stock_open": 1.0,
        },
    },
    "ILS_SA": {
        "has_ls"                 : True,
        "local_search_method"    : "SA",
        "init_ls_method"         : "SA",
        "local_search_time"      : 30.0,
        "local_search_iterations": 9999,
        "init_ls_time"           : 20.0,
        "init_ls_iterations"     : 9999,
        "T_init"                 : 100.0,
        "alpha"                  : 0.9999,
        "sa_neighborhood_weights": {
            "remove": 1.0, "swap": 2.0,
            "relocate": 4.0, "stock_reset": 3.0, "pattern_replace_all": 1.0,
            "insert": 1.0, "stock_open": 1.0,
        },
        "perturb_k"              : 1,
        "perturb_k_max"          : 3,
        "patience"               : 3,
    },
    "TS": {
        "has_ls"        : True,
        "tabu_tenure"   : 20,
        "max_iterations": 9999,
        "active_moves"  : {"relocate", "stock_reset"},
    },
    "GRASP_SA": {
        "has_ls"                 : True,
        "alpha"                  : 0.1,
        "n_restarts"             : 9999,
        "sd_active_moves"        : {"relocate"},
        "outer_method"           : "SA",
        "fi_time_limit"          : 30.0,
        "repair_time_limit"      : 5.0,
        "sd_time_limit"          : 15.0,
        "T_init"                 : 100.0,
        "alpha_sa"               : 0.9999,
        "sa_neighborhood_weights": {
            "remove": 1.0, "swap": 2.0,
            "relocate": 4.0, "stock_reset": 3.0, "pattern_replace_all": 1.0,
            "insert": 1.0, "stock_open": 1.0,
        },
    },
}
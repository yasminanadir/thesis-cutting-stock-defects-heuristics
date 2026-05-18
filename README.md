## Overview
This repository contains the full implementation for the master's thesis on the Cutting Stock Problem with defects, including instance generation, constructive heuristics, metaheuristic improvement methods, and experimental results.

## Requirements
Python 3.14.3. Install dependencies with:

    pip install -r requirements.txt

## Project Structure

    ├── main.ipynb   # Main experiment notebook
    ├── requirements.txt
    ├── src/
    │   ├── instance.py              # Instance loading and pattern building
    │   ├── solution.py              # Solution representation, decode, evaluate
    │   ├── constructive.py          # Greedy, multistart, GRASP
    │   ├── metaheuristic.py         # FI, SD, SA, TS, VNS, ILS and base class
    │   ├── moves.py                 # Neighborhood move operators
    │   ├── runner.py                # Main experiment runner + METHODS_CONFIG
    │   ├── runner_convergence.py    # Convergence logging runner
    │   ├── utils.py                 # Solution save/load utilities
    │   ├── visualize_stocks.py      # Solution visualization
    │   └── data_generator.py        # Synthetic instance generator
    ├── Data/
    │   ├── tuning/                  # 50 baseline instances for parameter tuning
    │   ├── testing/
    │   │   ├── baseline/            # 100 baseline test instances
    │   │   ├── sweep1_defect/       # Sensitivity: defect density (Low/High)
    │   │   ├── sweep2_demand/       # Sensitivity: demand tightness (Loose/Tight)
    │   │   ├── sweep3_length_ratio/ # Sensitivity: length ratio (Short/Long)
    │   │   ├── sweep4_width_ratio/  # Sensitivity: width ratio (Low/High)
    │   │   └── sweep5_size/         # Sensitivity: instance size (XS/S/L/XL)
    │   └── company_instances_updated/   # Real company instances

## How to Run the main.ipynb

### Step 1 — Imports and setup
Run cell [Code 0] — imports all required modules.

### Step 2 — Load instances
Run cell [Code 7] — loads test instances from Data/testing/ into check_instances.
For tuning, run cell [Code 39] — loads instances from Data/tuning/.

### Step 3 — Run experiments
Run cell [Code 20] — imports runner.py and METHODS_CONFIG.
Then run the main runner loop — iterates over all methods and instances.
Results are saved as JSON in SOLUTIONS_DIR defined in runner.py.

To run with convergence logging, import runner_convergence.py instead.
This saves both JSON solutions and convergence CSV files.

### Step 4 — Visualize solutions
Run cell [Code 25] to visualize one solution interactively.
Run cell [Code 27] to save all solutions as PDFs to output/pdfs/.

### Step 5 — Convergence plots
Run the cells under the Convergence section to generate plots per method and category.

### Step 6 — Tuning
Run cells under the Tuning section — one subsection per method, each saves results to an Excel file in outputs/.

## Output Files
- outputs/solutions_final/     solution JSON files (one per method x instance)
- outputs/results/final/       Excel result files
- outputs/convergence_logs/    convergence CSV files (one per method x instance)
- output/pdfs/                 solution visualization PDFs

## Notes
- Random seed fixed at 42 for reproducibility.

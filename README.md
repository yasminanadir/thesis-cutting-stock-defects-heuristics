## Overview
This repository contains the full implementation for the master's thesis on the Cutting Stock Problem with defects, including instance generation, constructive heuristics, metaheuristic improvement methods, and experimental results.

## Requirements
Python 3.14.3. Install dependencies with:

    pip install -r requirements.txt

## Project Structure
    ├── main.ipynb               # Main experiment notebook
    ├── tuning.ipynb             # Parameter tuning notebook (part 1)
    ├── tuning-2.ipynb           # Parameter tuning notebook (part 2)
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
    ├── Data/                        # Not included — see Google Drive link below

## Data
Instance files are available on Google Drive:
https://drive.google.com/drive/folders/1FAiTSRaxQHdXBkbnvYePEMy2qSXeygrU?usp=sharing

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
Run tuning.ipynb and tuning-2.ipynb — one subsection per method, each saves results to an Excel file in outputs/.

## Output Files
- outputs/solutions_final/     solution JSON files (one per method x instance)
- outputs/results/final/       Excel result files
- outputs/convergence_logs/    convergence CSV files (one per method x instance)
- output/pdfs/                 solution visualization PDFs

## Notes
- Random seed fixed at 42 for reproducibility.
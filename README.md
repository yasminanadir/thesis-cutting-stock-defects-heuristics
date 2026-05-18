## Overview
This repository contains the full implementation for the master's 
thesis on the Cutting Stock Problem with defects, including instance 
generation, constructive heuristics, metaheuristic improvement 
methods, and experimental results.

## Requirements
Python 3.10. Install dependencies with:

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

## Data
Instance files are not included in this repository due to their size.
They are available on Google Drive:
https://drive.google.com/drive/folders/1FAiTSRaxQHdXBkbnvYePEMy2qSXeygrU?usp=sharing

## How to Run

### main.ipynb — Main experiments
**Step 1 — Imports and setup**
Run the first cell to import all required modules.

**Step 2 — Load instances**
Run the loading cell to load test instances from `Data/testing/`.

**Step 3 — Run experiments**
Run the runner loop to iterate over all methods and instances.
Results are saved as Excel files in `RESULTS_DIR` defined in `runner.py`.
For convergence logging, use `runner_convergence.py` instead.

**Step 4 — Visualize solutions**
Run the visualization cells to display solutions interactively or 
save them as PDFs.

**Step 5 — Convergence plots**
Run the convergence section to generate plots per method and category.

### tuning.ipynb and tuning-2.ipynb — Parameter tuning
Each notebook contains one subsection per method. Run the loading 
cell first, then run each tuning section independently. Results are 
saved as Excel files in `outputs/tuning/`.

## Output Files

    outputs/solutions_final/    solution JSON files (one per method x instance)
    outputs/results/final/      Excel result files
    outputs/convergence_logs/   convergence CSV files (one per method x instance)
    outputs/pdfs/               solution visualization PDFs

## Notes
- Random seed fixed at 42 for reproducibility.
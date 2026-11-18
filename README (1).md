# Battery Health Analysis — NASA Li-Ion Ageing Dataset

Analysing how lithium-ion batteries lose capacity as they age, and predicting their **State of Health (SOH)** with a machine learning model. A self-directed data-science project built end to end in Python, from raw sensor data to a trained model.

---

## Motivation

Data science has become essential across almost every engineering domain — the ability to turn raw sensor data into case-by-case analysis and useful predictions. As a first-year engineering student, I wanted to build that skill hands-on by taking one real project all the way from raw data to a working model.

## Dataset

The data comes from the **NASA Ames Prognostics Center of Excellence** battery ageing set (Saha & Goebel, 2007). Four 18650 lithium-ion cells (B0005, B0006, B0007, B0018), each rated at 2.0 Ah, were repeatedly charged, discharged, and measured for impedance at room temperature until they aged. The measured discharge capacity of each cycle is the indicator of cell health.

- Source: NASA Prognostics Data Repository, NASA Ames Research Center.
- Format: MATLAB `.mat` files, one per cell.

## What I did

1. **Loaded and processed** the raw MATLAB files in Python (`scipy.io.loadmat`), extracting the discharge capacity of every cycle. Because each file interleaves charge, discharge, and impedance operations, discharges were counted separately to give a true cycle number.
2. **Analysed capacity fade** across all four cells, computed **State of Health** (capacity ÷ first measured capacity), and found each cell's **end of life** (defined as 1.4 Ah, 70% of the 2.0 Ah rating).
3. **Visualised** the results with Matplotlib.
4. **Built a linear regression model** (scikit-learn) to predict a cell's State of Health from its discharge behaviour.

## Results

Capacity degradation across the four cells:

| Cell  | Cycles | First (Ah) | Last (Ah) | Total fade | End of life (cycle ≤1.4 Ah) |
|-------|--------|------------|-----------|------------|------------------------------|
| B0005 | 168    | 1.856      | 1.325     | 28.6%      | 125                          |
| B0006 | 168    | 2.035      | 1.186     | 41.7%      | 109                          |
| B0007 | 168    | 1.891      | 1.432     | 24.3%      | never reached                |
| B0018 | 132    | 1.855      | 1.341     | 27.7%      | 97                           |

![Capacity fade](figures/capacity_fade.png)

**Key findings**

- **Identical cells degrade differently.** Same model, same test protocol, yet end of life ranged from cycle 97 to 125 — and one cell never reached it. Cell health can't be read from cycle count alone.
- **Capacity doesn't fall smoothly.** The curves show repeated upward jumps: *capacity-regeneration events*, where a resting cell temporarily recovers capacity as lithium redistributes within the electrodes.
- **B0007 never reaches 1.4 Ah** (minimum 1.40 Ah), so it has no defined end of life — reported as such rather than forcing a result by lowering the threshold.

**Model** — a linear regression predicting State of Health, evaluated with **leave-one-battery-out cross-validation** (train on three cells, test on the fourth, each cell held out once):

| Held-out cell | R²   |
|---------------|------|
| B0005         | 0.94 |
| B0006         | 0.93 |
| B0007         | 0.07 |
| B0018         | 0.91 |
| **Average**   | **0.71** |

The model predicts SOH well for three of the four cells (average absolute error ≈ 3.8 SOH points). B0007 is the exception — the same cell that never reaches end of life, so it behaves differently from the cells the model trained on, and is much harder to predict.

## Tech stack

Python · pandas · NumPy · SciPy · Matplotlib · scikit-learn · Jupyter

## Repository structure

```
.
├── README.md
├── NASA_battery.ipynb          # the main notebook
├── figures/
│   └── capacity_fade.png       # capacity-vs-cycle plot
└── data/                       # .mat files (see note below)
```

## Running it

1. Install the libraries:
   ```
   pip install numpy pandas scipy matplotlib scikit-learn jupyter
   ```
2. Download the four `.mat` files (B0005, B0006, B0007, B0018) from the NASA PCoE repository and place them in a `data/` folder.
3. Open the notebook:
   ```
   jupyter notebook NASA_battery.ipynb
   ```

> **Note on the data:** the `.mat` files are not included in this repository, as they belong to NASA. Download them from the NASA Prognostics Data Repository and add them to `data/`.

## Acknowledgements

B. Saha and K. Goebel (2007), *Battery Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center.

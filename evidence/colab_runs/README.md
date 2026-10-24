# Colab run evidence

Each folder holds the `report.md` and `manifest.json` of one run made on Google Colab, named by its run id.
Notebook 06 (`notebooks/06_Evaluation.ipynb`) collects them by default and compares them with any further runs
it is given; the final experiment report cites them.

Only the report and manifest are kept here: no source images, masters or figures (PRD 21.5). The reports link
figures that stay in the original run folders.

| Run | Notebook | Source | GPU |
|---|---|---|---|
| `20261024T090241Z_40e30440_baselines` | 00 (v1.1, commit c3c1abb) | 6336 x 2688 AI-rendered festival portrait | NVIDIA L4 |
| `20261024T093611Z_5fe78cd3_baselines` | 00 | 6336 x 2688 AI-rendered night forest path | NVIDIA L4 |

To add a run, copy its `report.md` and `manifest.json` from the report zip into a new folder named by its run id.

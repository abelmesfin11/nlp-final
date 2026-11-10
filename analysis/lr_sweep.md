# Learning-rate sanity check (seed 1)

Default LRs used in the main grid: Full FT 1e-5, LoRA/DoRA 2e-4 (marked *).

| Method | LR | Clean SST-2 | Noisy SST-2 | Noisy IMDb | Memorization (noisy) | Best noisy SST-2 (oracle) |
|---|---|---|---|---|---|---|
| Full FT | 5e-06 | 94.4 | 80.0 | 76.8 | 76.1 | 89.0 |
| Full FT | 1e-05* | 94.4 | 80.7 | 81.0 | 84.0 | 85.7 |
| Full FT | 2e-05 | 92.9 | 78.0 | 81.2 | 75.6 | 85.9 |
| LoRA | 0.0001 | 93.7 | 88.3 | 79.5 | 32.0 | 91.4 |
| LoRA | 0.0002* | 94.8 | 84.5 | 84.0 | 52.6 | 87.2 |
| LoRA | 0.0005 | 81.0 | 50.9 | 50.0 | 44.1 | 59.9 |
| DoRA | 0.0001 | 93.3 | 88.6 | 78.7 | 37.1 | 90.8 |
| DoRA | 0.0002* | 93.2 | 83.1 | 84.5 | 60.9 | 88.8 |
| DoRA | 0.0005 | 90.1 | 72.4 | 67.7 | 27.6 | 72.4 |

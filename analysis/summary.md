| Method | Noise | Seeds | Trainable params | SST-2 acc | IMDb acc | Memorization | Corrected | Best SST-2 (oracle) |
|---|---|---|---|---|---|---|---|---|
| DoRA | 0% | 3 | 10,438,656 | 93.7 ± 0.4 | 88.8 ± 0.6 | – | – | 93.9 ± 0.1 |
| DoRA | 30% | 3 | 10,438,656 | 83.7 ± 1.5 | 81.7 ± 2.7 | 54.2 ± 7.0 | 45.8 ± 7.0 | 88.1 ± 0.7 |
| Full FT | 0% | 3 | 596,051,968 | 93.7 ± 0.6 | 88.1 ± 0.2 | – | – | 93.8 ± 0.7 |
| Full FT | 30% | 3 | 596,051,968 | 81.4 ± 2.1 | 78.5 ± 3.2 | 80.8 ± 3.3 | 19.2 ± 3.3 | 86.4 ± 1.5 |
| LoRA | 0% | 3 | 10,094,592 | 94.4 ± 0.5 | 89.3 ± 0.7 | – | – | 94.4 ± 0.4 |
| LoRA | 30% | 3 | 10,094,592 | 85.0 ± 1.2 | 81.8 ± 2.4 | 46.4 ± 5.6 | 53.6 ± 5.6 | 87.3 ± 0.4 |

**Accuracy drop from clean → noisy (percentage points, mean over seeds)**

| Method | SST-2 drop | IMDb drop |
|---|---|---|
| Full FT | 12.3 | 9.5 |
| LoRA | 9.4 | 7.6 |
| DoRA | 10.0 | 7.1 |

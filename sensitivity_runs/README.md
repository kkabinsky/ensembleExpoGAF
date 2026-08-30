# Sensitivity of the reported results to threshold, seed and window overlap

Five programs. Two need no training and are exact; three retrain and are
reproducible only up to training noise.

| Program | What it answers | Trains? |
|---|---|---|
| `threshold_sweep.py` | how much of the reported performance depends on where the alarm threshold was put | no |
| `overlap_check.py` | how much independent information the overlapping windows carry, and what that costs in precision | no |
| `overlap_corrected_tests.py` | confidence intervals that resample contiguous blocks and the cells themselves | no |
| `lstm_ablation.py` | whether the recurrent layer in the TadGAN front end is doing anything | yes |
| `seed_variability.py` | how far the reported pipeline moves when only the seed changes | yes |

```
pip install numpy pandas scipy torch openpyxl
python threshold_sweep.py
python overlap_check.py
python overlap_corrected_tests.py
python lstm_ablation.py --check-only
python seed_variability.py --seeds 3
```

`lstm_ablation.py --check-only` runs the free structural check: it places a
hook on the recurrent cell and reads what actually reaches it. The window
arrives as the feature dimension of a single timestep, not as a sequence, so
the layer performs no recurrence.

The two programs that train write every finished cell immediately. Interrupt
either and rerun the same command: it continues from the next unfinished cell.

They read from `../Iran_new_run`, so run them from inside this folder.

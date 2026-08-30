# Does the ordering of the four angular mappings depend on who reads the image?

The window sweep compares the four angular mappings with a supervised
convolutional classifier and puts the exponential map first. The detector the
framework proposes is one-class and never sees a crash label. This folder reads
the same images six ways on the same split and reports where the ordering holds.

```
pip install numpy pandas torch openpyxl
python run_ablation.py --epochs 20 --seeds 3 --tag _full
python verify.py --results output/ablation_results_full.csv
```

`verify.py` separates two kinds of number. The mapping profile, the linear
probe and the distance to the mean normal image are closed-form and must
reproduce to machine precision. The three trained arms are seeded but not
bit-exact across builds of PyTorch, so they are checked against a stated
tolerance; what should be compared is the ordering, not the third decimal. The
check also asserts that the reproduced f-AnoGAN still has the published
parameter counts, so a change to that network cannot pass unnoticed.

`budget_sweep.py` repeats the comparison at four training budgets, and
`lambda_sweep.py` at two gradient-penalty weights. The second was run to test a
stated prediction, that lowering the penalty would let the mappings separate
again. It did not, and the result is reported as a failed prediction rather
than dropped.

It reads the saved window positions, labels and price series from
`../Iran_new_run`, so run it from inside this folder.

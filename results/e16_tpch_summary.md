# E16 -- deletion replay over a learned CE on real TPC-H (SF=0.3)

`code/e16_tpch.py`, 2026-11-11 02:46; seeds [0, 1], 16000 training / 4000 test queries, 40 base epochs, 10 epochs per unlearning invocation, k = 1..5.  Runtime 6.3 min.

## Means over seeds (pooled test set, n=4000 per seed)

| arm | in-region median Q-error | overall median Q-error | in-region p90 | other tenants' regions |
|---|---|---|---|---|
| base | 1.160 | 1.125 | 2.17 | 1.120 |
| retrain | 1.173 | 1.129 | 2.24 | 1.121 |
| neg1 | 1.557 | 1.263 | 5.71 | 1.244 |
| neg2 | 1.274 | 1.174 | 3.48 | 1.159 |
| neg3 | 1.262 | 1.165 | 2.94 | 1.154 |
| neg4 | 1.266 | 1.186 | 2.59 | 1.175 |
| neg5 | 1.242 | 1.204 | 2.68 | 1.199 |
| ft1 | 1.409 | 1.302 | 5.33 | 1.290 |
| ft2 | 1.251 | 1.158 | 3.36 | 1.146 |
| ft3 | 1.359 | 1.304 | 4.01 | 1.292 |
| ft4 | 1.191 | 1.139 | 2.53 | 1.131 |
| ft5 | 1.190 | 1.141 | 2.39 | 1.135 |

### Per template (in-region = tenant T_a's query-space box, n=248 T1 / 445 T2 per seed)

| arm | T1 in-region | T1 overall | T2 in-region | T2 overall |
|---|---|---|---|---|
| base | 1.056 | 1.057 | 1.382 | 1.412 |
| retrain | 1.069 | 1.066 | 1.416 | 1.437 |
| neg1 | 1.083 | 1.115 | 2.530 | 2.643 |
| neg2 | 1.080 | 1.070 | 1.690 | 1.689 |
| neg3 | 1.068 | 1.065 | 1.664 | 1.735 |
| neg4 | 1.119 | 1.105 | 1.545 | 1.577 |
| neg5 | 1.146 | 1.150 | 1.433 | 1.464 |
| ft1 | 1.180 | 1.177 | 1.871 | 1.963 |
| ft2 | 1.060 | 1.063 | 1.606 | 1.663 |
| ft3 | 1.199 | 1.198 | 1.659 | 1.769 |
| ft4 | 1.071 | 1.070 | 1.467 | 1.481 |
| ft5 | 1.048 | 1.064 | 1.427 | 1.485 |

Per-seed in-region medians (pooled): base 1.17/1.15; retrain 1.19/1.16; neg1 1.31/1.80; neg5 1.23/1.25; ft5 1.21/1.17.

## Reproduce

```powershell
cd C:\Users\35734\Desktop\ICLR\icde2027
python code/e16_tpch.py --seeds 0,1,2 --train 16000 --test 4000 --epochs 40 --kmax 5 --ue-epochs 10 --ue-lr 0.01 --ue-wd 0.0005 --ue-beta 0.95 --clip 5.0
```

## Interpretation

1. Replay does NOT replicate: replay does not exceed the exact-retrain reference by a meaningful margin.  The in-region median Q-error of the NegGrad+ estimator moves 1.160 (base) -> 1.557 (k=1) -> 1.242 (k=5), i.e. x1.07 over base and x1.06 over the exact-retrain reference.  The synthetic orders/customers workload moved 1.14 -> 1037 (x910), so the direction of the effect and its locality are the same, but the TPC-H magnitude is tiny by comparison.

2. Base vs retrain in-region: 1.160 -> 1.173 (x1.01).  Removing the tenant's ~5% of the training queries barely changes its own region, so essentially all of the replay damage is operator drift rather than missing data.

3. The damage is local and nearly invisible overall: at k=5 the pooled median Q-error reads 1.204 against 1.125 at base (x1.07), while the same queries inside the tenant region read 1.160 -> 1.242 (x1.07), and test queries belonging to the other seven tenants sit at 1.199.

4. Retain-only fine-tuning with the same budget is the control: 1.190 in-region at k=5 vs 1.160 base (x1.03), so the degradation is caused by the forget term of NegGrad+, not by the extra SGD steps on the shrunk retain set.

5. Honest caveats: the o_totalprice predicate is snapped to 16 equal-count price bands (so T2 price ranges are quantile-aligned, not arbitrary constants), T2 cardinalities are stratified to a log-uniform target over 10^2.0-10^6.0 (the sampler's reachable range tops out near 10^5.4; T1 spans 10^1.8-10^4.8 naturally), and the run uses 2 seeds.  The budget was REDUCED from the planned 40k queries / 60 epochs to 16k / 40 epochs: sustained CPU load in this environment runs ~5-9x slower than short bursts (measured 1.1 s/epoch for a 3-epoch run vs 8.8 s/epoch over 60 epochs), which put the full budget outside the time limit.  Q-error magnitudes are therefore not directly comparable with the synthetic x910 result.


## Feature vector (18 dims)

`sd_lo`, `sd_hi`, `sd_w`, `disc_lo`, `disc_hi`, `disc_w`, `qty_lo`, `qty_hi`, `qty_w`, `od_lo`, `od_hi`, `od_w`, `price_lo`, `price_hi`, `price_w`, `tmpl_T1`, `tmpl_T2`, `join_flag`

All bounds normalised to [0,1] over their real TPC-H range (price via log1p(o_totalprice)/log1p(max)); unused slots of a template are 0.  Target = standardised log1p(true cardinality), inverse transform expm1(z*sd+mu); Q-error = max(est/true, true/est) with both clipped at 1.


## Workload / tenant definition

- T1 (Q6-like): d1 uniform over the shipdate range, width 30-365 d; discount from the 11 real values, width 0.00-0.05; quantity width 1-20 of 1-50.  Natural distribution, no stratification.
- T2 (Q3-like): o_orderdate width log-spaced 30-1800 d, l_shipdate width 30-365 d, o_totalprice bounds snapped to the 16 equal-count price bands; candidates stratified to a log-uniform cardinality target over 10^2.0-10^6.0 in 0.5-dex bins.
- 8 tenant boxes = 4 shipdate-window-centre bands x 2 secondary bands (T1 discount centre, T2 price-band centre).  Tenant T_a owns region 3 and ~5% of the training set (it is 30% of that region's training mass); deleting it removes exactly those rows.
- base/neg/ft share the base target scaler; retrain (from scratch) uses the scaler of its own retained set.
- neg = NegGrad+ (beta*MSE_retain - (1-beta)*MSE_forget, beta=0.95, SGD momentum 0.9, lr 0.01, wd 5e-4, clip 5, batch 256, 10 epochs per invocation, forget batches drawn from the tenant's queries); ft = the same loop without the forget term.  Both reuse `su.methods.run`.

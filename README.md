# RABS: Risk-Adaptive Bandwidth Scaling for Smart-Greenhouse NCS

Reproducible, stdlib-only simulation code for the RABS policy family
(Risk-Adaptive Bandwidth Scaling) for networked control of smart-greenhouse
sensor systems under burst packet loss.

The study treats the per-step transmission budget `B_t` as a time-varying
control variable selected *before* the current packet is transmitted, rather
than a fixed budget. Three variants are evaluated:

- **RABS-H** — threshold rule on a pre-transmission risk index.
- **RABS-L** — predicted loss/cost balancing rule.
- **RABS-PD** — primal-dual selector with online penalty updates for
  bandwidth, information age, and missed-violation risk.

## Reproducing the results

Run everything from this directory. Python 3.9+ only; **no third-party
dependencies** for the core simulation.

```bash
# 1. Main experiment (two Gilbert-Elliott burst-loss scenarios, 16 windows x 20 seeds, n=320)
python3 code/run_rabs_adaptive_bandwidth.py

# 2. Risk-weight sensitivity sweep (5 weight configurations)
python3 code/run_rabs_weight_sensitivity.py
python3 code/make_sensitivity_table.py

# 3. Paired Wilcoxon significance analysis (requires scipy)
python3 code/analyze_rabs_wilcoxon.py

# 4. Sanity-check the regenerated outputs
python3 code/validate_rabs_outputs.py
```

### Inputs

```text
data/source/safety_probability_calibration_raw.csv
```

Processed greenhouse microclimate data derived from a public Mendeley
greenhouse dataset, combined with simulated Gilbert-Elliott burst loss.
This is **not** a real wireless trace; the channel scenarios are simulated
stress conditions.

### Outputs

```text
outputs/rabs/rabs_summary.csv          # full summary, both scenarios
outputs/rabs/rabs_holdout_summary.csv  # held-out windows 9-15
outputs/rabs/rabs_wilcoxon.csv         # paired significance results
outputs/tables/*.tex                   # LaTeX tables used in the paper
```

## Channel scenarios (Gilbert-Elliott)

| Scenario      | p(loss\|good) | p(good→bad) | p(loss\|bad) | p(bad→good) |
|---------------|--------------|-------------|--------------|-------------|
| burst         | 0.06         | 0.035       | 0.65         | 0.22        |
| severe_burst  | 0.08         | 0.055       | 0.82         | 0.15        |

## Headline (n=320 paired)

- RABS-PD saves **48.7–50.5%** bandwidth vs Fixed-B3.
- In the `burst` scenario RABS-PD reaches statistical parity with Fixed-B3 on
  the objective (Δ≈+0.0006, p≈0.28) at roughly half the bandwidth.
- In the `severe_burst` scenario RABS-PD significantly improves the objective
  vs Fixed-B3 (p<0.001) while still saving bandwidth.
- Gap to Oracle-B remains 44.1% (burst) / 29.0% (severe_burst): Oracle-B is a
  favorable reference, not a deployable policy.

Results describe a controlled simulation, not a closed-loop greenhouse
deployment.

## Citation

See `CITATION.cff`.

## License

MIT — see `LICENSE`.

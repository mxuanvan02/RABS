# RABS: Risk-Adaptive Bandwidth Scaling for Safety-Critical Smart-Agriculture IoT

Reference implementation for the paper

> **Self-Tuning Risk-Adaptive Bandwidth Scaling for Safety-Critical Smart-Agriculture IoT Networks**

RABS decides, every time slot, *how many* sensors to poll and *which ones*, under a
shared, loss-prone wireless channel. It is built for safety-critical environmental
monitoring — here, heat-stress detection over rice-growing zones in the Mekong
delta — where a missed hot spell is costly but the bandwidth budget is small.

## The problem

A gateway polls `N` field sensors over a bursty-loss channel with a per-slot budget
of `B` polls. Fixed-budget schedulers waste bandwidth when conditions are calm and
starve the network during a heat episode, when fresh readings matter most. RABS
treats the budget itself as a control variable that expands under risk and
contracts when the field is safe.

## Contributions

- **Risk-adaptive budget.** The per-slot polling budget is an online decision, not a
  preset constant: a network-level risk index scales the effective cost of polling so
  the controller opens up during excursions and tightens otherwise.
- **Pre-transmission urgency score.** Each zone is ranked before transmission by a
  blend of threshold-violation probability, predicted deviation, and age of
  information, so scarce polls go to the zones most likely to matter.
- **Self-tuning via primal–dual feedback (RABS-PD).** The bandwidth, staleness, and
  missed-risk penalties are Lagrangian multipliers updated from observed constraint
  slack, removing per-scenario penalty tuning while keeping fixed design targets.
- **A time-average guarantee.** Under bounded multipliers, the regulated surrogate
  slacks converge to their targets at rate `O(1/T)`, stated without per-scenario
  retuning.

## How it works

Let `x_i` be a zone's true temperature, `x̂_i` the gateway estimate from its last
successful poll, and `a_i` the age of that estimate. With safety band
`[τ_min, τ_max] = [22, 34] °C` and predictor spread `σ = 1.2`, the violation
probability is a two-sided Gaussian tail:

```
p_i = Φ((τ_min - x̂_i)/σ) + 1 - Φ((τ_max - x̂_i)/σ)
```

**Urgency score (which zones to poll).** Zones are ranked by

```
S_i = 0.55 · p_i + 0.25 · δ_i + 0.20 · min(a_i / 8, 1)
```

where `δ_i = min(|x_i - x̂_i| / r, 1)` is the normalized deviation and `r` is the
half-width of the band. The top `B` zones are polled.

**Risk index (how hard to push the budget).** A single network-level scalar blends
the worst zone, the mean, the average staleness, the recent channel quality `p_bad`,
and the largest deviation:

```
R = 0.45·max_i p_i + 0.20·mean_i p_i + 0.18·min(ā/8, 1) + 0.10·p_bad + 0.07·max_i δ_i
```

**Budget selection (how many to poll).** For each candidate budget `b ∈ {1,2,3}` the
controller scores predicted tracking loss `L̂`, age `Â`, and the missed-risk surrogate
`M̂`, with a risk discount that makes polling cheaper when the field is hot:

```
B_t = argmin_b  L̂(b) + (c_B + λ_B)·b + (c_A + λ_A)·Â(b) + (c_M + λ_M)·M̂(b) − c_R·R·b
```

with `c_R = 0.030`. A missed excursion is weighted 5× a false alarm in `L̂`.

**Self-tuning multipliers (RABS-PD).** The three penalties are dual variables driven
by their own constraint slack against fixed targets (`B* = 1.55`, `A* = 1.60`,
`M* = 0.008`):

```
λ_B ← [ λ_B + 0.010·(b      − B*) ]₊
λ_A ← [ λ_A + 0.006·(ā      − A*) ]₊
λ_M ← [ λ_M + 0.020·(m̂_t    − M*) ]₊
```

Overspending raises the matching price, which makes the next slot more frugal;
underspending lets it fall back to zero. This feedback is what allows one parameter
set to work across calm periods, heat episodes, and different loss regimes.

The headline result is a composite safety objective of `0.100` under severe burst
loss versus `0.146` for the strongest fixed-budget baseline, at an average budget of
`1.37` slots instead of `3.00` — roughly half the bandwidth for a strictly safer
schedule.

## Application

Evaluation replays real ERA5 hourly 2-m temperature (2024) for Mekong-delta rice
zones — three primary stations (Cần Thơ, Sóc Trăng, Cà Mau) for the main study and up
to 20 stations for scalability — fetched reproducibly from the public Open-Meteo
archive. The traces are genuine heat-stress regimes (2–4% of hours above 34 °C, peaks
near 38 °C). Packet propagation and sensor placement are synthetic Gilbert–Elliott
burst-loss channels, not field radio measurements; the approach transfers to any
safety-critical low-bandwidth sensing task with an excursion-style risk model.

## Quick start

```bash
git clone https://github.com/mxuanvan02/RABS.git
cd RABS
bash reproduce.sh        # one command: env + data + experiments + tables + figure
```

`reproduce.sh` builds a virtual environment, installs `numpy`/`scipy`/`matplotlib`,
fetches (or reuses the cached) ERA5 data, runs the full experiment suite, and
regenerates every table and figure under `outputs/`. No API key is needed and all
runs are seeded, so results are deterministic for a given environment and dataset.
Python 3.9+; see `code/README.md` for individual entry points.

## Repository layout

```text
code/    experiment scripts, ablations, and table/figure generators
data/    ERA5 fetcher + cached station traces (data/era5_vn/*.csv)
reproduce.sh   end-to-end pipeline
```

Generated outputs are not committed; regenerate them with `bash reproduce.sh`. If
Open-Meteo availability changes, the cached `data/era5_vn/*.csv` files reproduce the
reported results exactly.

## License

See `LICENSE`.

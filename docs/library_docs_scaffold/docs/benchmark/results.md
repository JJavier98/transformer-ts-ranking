# Benchmark & Results

The library is validated by an independent benchmark — [`transformer-ts-ranking`][bench] — that runs
every eligible model across long-term and M4 forecasting under a fixed protocol (temporal split,
3 seeds, fp32) and ranks them with the standard statistical machinery (Friedman, Nemenyi, Critical
Difference diagram).

!!! note "Placeholder — populated on release (docs phase D6)"
    This page **embeds** the benchmark's published artifacts; it does not recompute them. On each
    release, a sync step copies the current outputs into `docs/benchmark/`:

    - `leaderboard_long_term.csv`, `leaderboard_short_term.csv` — accuracy rankings.
    - `cd_diagram_long_term.png`, `cd_diagram_m4.png` — Critical Difference diagrams.
    - `leaderboard_efficiency.csv` — accuracy-vs-cost.

    Until the current benchmark run completes, this section is intentionally a stub.

## Protocol (summary)

- **Long-term:** 9 datasets × 4 horizons × 3 seeds.
- **Short-term (M4):** 6 frequency slices, ranked by OWA.
- **Precision:** fp32 on every node, for hardware-independent comparability.
- **Ranking:** per-configuration ranks aggregated across datasets; incomplete models are reported
  separately, never silently dropped.

[bench]: https://github.com/ari-dasci/S-TransformerTS

# QPL Implementation Notes

This project uses a QAHO/QFSE-inspired QPL calculation as the main implementation.

The pipeline is:

1. Estimate a rolling return probability density `rho(r)` from past returns only.
2. Build a normalized wavefunction `psi(r)` from `rho(r)=|psi(r)|^2`.
3. Construct a QAHO-style potential `V(r) = c_gamma_d^2 r^2 - (c_gamma_v / 4) r^4`.
4. Build a finite-difference Hamiltonian for the time-independent QFSE.
5. Solve low-order energy levels.
6. Map energy spacing to normalized quantum price relatives, `NQPR`.
7. Compute positive and negative QPL bands:
   `QPL+ = Open_t * NQPR` and `QPL- = Open_t / NQPR`.

The rolling volatility proxy is retained only as a fallback for short windows or numerical failures. It should not be described as the main QPL experiment.

## Timing

`QPL_t` uses only returns strictly before date `t`. When `anchor: open` is used, the model assumes the QPL boundary is formed after the date `t` open is known.

High/low touch events are not valid action-pre observation features for the same date. They are used as execution and risk simulation signals inside Gate V2, or can be used as lagged features on the next date.

## Touch Features

For level `n`, the package exports:

- `touch_plus_by_high_n = high_t >= QPL+_n`
- `touch_minus_by_low_n = low_t <= QPL-_n`
- `intraday_breakout_n = touch_plus_by_high_n and close_t > QPL+_n`
- `intraday_breakdown_n = touch_minus_by_low_n and close_t < QPL-_n`

If a dataset does not provide high/low, the loader falls back to close-based touch detection and records that fallback in `MarketOHLCV.fallback_fields`.


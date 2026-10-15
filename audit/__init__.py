"""
Auditor / self-improvement layer.

  journal      append-only bet journal (place -> close -> settle)
  learner      Bayesian edge tracker per strategy / segment -> stake multipliers, ACTIVE/WATCH/RETIRED
  calibration  reliability / Brier / ECE + online (past-only) recalibrator
  confidence   pre-bet grade (A+/A/B/C/PASS) + recommended stake
  replay       walk-forward proof (static vs adaptive staking), regime-flip test, historical CLV study
"""

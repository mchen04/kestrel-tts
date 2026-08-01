# 62 — duration head, done with the right recipe

## Two defects found in cycle 60's recipe (read from `train_prosody.py`, before any code here)
1. **Wrong target.** The original training regresses on `durraw` — the *unrounded* float
   `sigmoid(duration_proj).sum(-1)` (`capture_prosody.py:83`, `train_prosody.py:61`). Cycle 60
   trained on rounded integers, discarding the sub-frame signal the head is supposed to learn.
2. **Wrong parameter set.** The original loss is joint — `4·ten + 2·dur + f0 + n`
   (`train_prosody.py:104`) — over a *shared* encoder. Cycle 60 optimized `dur` alone through that
   shared encoder, which is almost certainly why MCD, F0 and vuv all regressed there: it damaged the
   `ten` features the decode student consumes. **The control in cycle 60 regressing against shipped
   is explained by this, not by noise.**

question:      with the raw-float target and the encoder **frozen** (training only `dur_head`, which
               cannot damage `ten`/F0/N), does duration accuracy improve over the shipped head?
axis:          exactness, and — per cycle 61 — fidelity: correct durations are worth
               mel L1 1.618 → 0.591 on `student-fast`, so this is the cheap route to most of that.
prediction:    drift improves on both mean and worst versus shipped 4.97 / 50.30 %, with **no**
               regression in MCD / F0 / vuv (frozen encoder makes that structural, not hopeful).
falsifier:     drift fails to beat shipped on *both* mean and worst-case, or any non-duration metric
               moves at all (which would mean the freeze is not actually holding and the run is
               invalid).
budget:        3 h (stop at 6 h regardless)
controls:      - **freeze check**: assert `ten` output is bit-identical before and after training.
                 If it is not, the experiment is void regardless of the drift numbers.
               - matched data, natural style only (cycle 60 killed uniform-random augmentation).
               - full frozen battery vs the same references; nothing ships unless every gate passes.

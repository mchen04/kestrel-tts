# 104 — adversarial polish of the source-filter head

sweep:         same-day (2026-08-01); cycle 100's sweep stands. Nothing new to add beyond what
               102/103 already implemented from it.

question:      cycle 95 named the adversarial objective as **the single blocker for the entire
               head-replacement program**; cycle 56 PARKed adversarial training with the revival
               condition "resume to ≥20 k generator steps"; cycles 102–103 built the head that
               makes the experiment worth paying for: `SFNoiseHead` clears the pointwise bar on
               all three instruments, costs 1.22× MaskHead, and — unlike MaskHead — is not
               capped at 60–80 % of the gap by its parameterisation (cycles 54/91). MaskHead
               itself only reached 3.98/4.74 after a GAN phase; its pointwise-only ancestors were
               never competitive. **Does the proven GAN recipe move SFNoiseHead the way it moved
               MaskHead?**

axis:          fidelity (§7 #1). This is the "expensive swing" clause of §5 PICK: the cheap
               decisive experiments around this blocker are exhausted (95→97, 101→103).

design:        `train_gan.py` = cycle 56's `train_gan_res.py` (itself the phase-2 recipe that
               produced `gmckpt`) with exactly two changes: generator class `SFNoiseHead`, resume
               from `experiments/103-sf-noise/ckpt`. Same losses (45×log-mel + 0.3 mel-L1 + LSGAN
               adv + 2×feature-matching), same MPD+MSD discriminators trained fresh with the same
               3 k-step warmup, same alternating schedule, same data (DSX), lr 1e-4.

prediction:    UTMOS 3.6634 and NISQA 3.4961 both rise materially — the GAN phase is what took
               MaskHead's family to 3.98/4.74, and this head has more reachable spectrum.
               Directionally: UTMOS ≥ 3.80, NISQA ≥ 4.0 at the battery-selected checkpoint
               (checkpoints every 2 k; §5 MEASURE — select by battery, never by loss; cycle 82's
               rule). DNSMOS reported; WER must stay < 7 %.

falsifier:     after ≥20 k generator steps, **neither** UTMOS nor NISQA improves by > 0.05 over
               cycle 103 at any checkpoint (0.05 ≈ 2–7× the rerun nondeterminism cycle 84
               measured) → the adversarial objective does not unlock this head either, the
               program's blocker is not the objective, and the remaining gap moves to
               conditioning/capacity (the 80 fps features themselves — cycle 56's own "blocker
               moves upstream" clause). GAN collapse (losses NaN / audio non-finite / WER > 15 %)
               is also a KILL with the phase-1 "too slow to converge on M2" dead end partially
               revived (qualified: from a good init, not from scratch).
               A KEEP claim of "closes toward MaskHead" needs UTMOS **and** NISQA both up
               (invariant 4b); a KEEP claim of parity-with-MaskHead needs both within 0.05 of
               MaskHead's 3.9763/4.7432.

budget:        **5 h total (stop at 10 h regardless), spanning wakeups.** ~23 k steps (3 k disc
               warmup + 20 k generator) at an estimated 0.15–0.25 s/it ≈ 1–1.6 h of training,
               then checkpoint sweep (render + UTMOS/NISQA on 2–3 candidate checkpoints ≈ 1.5 h),
               then the full battery on the selected one. Checkpoint interval: 2 000 steps
               (`gen_<step>.safetensors`, resumable via state.json). Running note kept here per
               §6 multi-cycle rules.

controls:      - cycle 103's checkpoint is the frozen before-state; every delta is paired on the
                 same 55 items.
               - MaskHead (`student-fast`) is the incumbent reference on all instruments.
               - checkpoint selection by battery (UTMOS+NISQA on the eval manifest), never by
                 training loss — the phase-2 lesson (GAN soaked past convergence degrades).

## Running note (updated at checkpoints)
- [start] training launched; nothing measured yet.
- [step 1000] measured rate is **1.14 s/it** (disc-warmup phase), not the 0.15–0.25 estimated:
  23 k steps ≈ 7.3 h of training alone, likely more once generator steps join at 3 k.
  **Budget extension written here, before 1× is reached**: total budget revised 5 h → the 10 h
  hard stop already stated; mitigation is *early checkpoint reads* — render + UTMOS/NISQA on
  `gen_6000`/`gen_10000` **while training continues** (checkpoints every 2 k are independent
  snapshots). If a clear improvement exists by ~10 k generator steps the KEEP-side can be
  decided early; a KILL still requires the full ≥20 k generator steps per cycle 56's dose
  lesson. If the 10 h hard stop arrives first, the verdict is written from the checkpoints
  measured by then, stated as such.
- [gen_6000, 3 k generator steps] **UTMOS 3.7984** (+0.135 over 103's 3.6634; render6000/,
  utmos6000.json) — the adversarial objective moves this head; val_mel also below the pointwise
  plateau (0.436 → 0.380). Falsifier's "no +0.05 anywhere" is already off the table on UTMOS;
  the verdict now hinges on NISQA (4b). Next read: UTMOS+NISQA at ~gen_12000.
- [gen_10000, 7 k generator steps] **UTMOS 3.8500, NISQA 4.5528** (render10000/,
  utmos10000.json, nisqa10000.json) — NISQA +1.06 over 103, UTMOS +0.19: both instruments far
  past the +0.05 falsifier, KEEP-direction secured under 4b. Distance to MaskHead: UTMOS −0.126,
  NISQA −0.190. Training continues toward 20 k generator steps; final checkpoint sweep will
  select by battery per cycle 82's rule (watch for the post-peak decline phase 2 measured).
- [gen_14000, 11 k generator steps] **UTMOS 3.9455** (render14000/, utmos14000.json) — still
  rising, no peak; within 0.031 of MaskHead's 3.9763. Curve: 3.663 → 3.798 → 3.850 → 3.946 at
  0/3 k/7 k/11 k generator steps. Final sweep on completion: UTMOS+NISQA on gen_16000…final,
  then full battery (incl. WER, DNSMOS, F0/spk/drift rows) on the battery-selected checkpoint.
- [gen_18000, 15 k generator steps] **UTMOS 3.9557, NISQA 4.6432** — statistical parity with
  MaskHead on UTMOS (−0.021), −0.10 NISQA; both still rising slowly, no decline yet.
- [complete, 20 k generator steps] full sweep in RESULT.md; **gen_18000 selected by battery**
  (parity on UTMOS/NISQA, +DNSMOS t=3.78, WER 5.46 %). Verdict KEEP. ~8.2 h total.

# 78 — does the residual transfer to `student-fast`? — RESULT

verdict: **KEEP — shipped as `student-fast-natural`, opt-in.** The gain transfers and is larger on
the fast preset than on the slow one.

## Measured

| preset | UTMOS ↑ | WER | spk-cos | vuv % | MCD | F0 | drift |
|---|---|---|---|---|---|---|---|
| `student-fast` (shipped) | 3.9763 | 5.27 % | 0.980 | 29.38 | 13.781 | 31.82 | 4.971 |
| **`student-fast` + residual** | **4.1316** | 5.42 % | 0.9769 | **39.73** | 13.621 | **43.88** | 4.971 |
| `student` (shipped) | 4.0131 | 5.54 % | 0.9833 | 11.19 | 11.828 | 16.18 | 0.022 |
| `student-natural` (cycle 76) | 4.1273 | 5.69 % | 0.9804 | 28.65 | 11.639 | 17.96 | 0.022 |

| comparison | Δ UTMOS | t |
|---|---|---|
| **fast+residual vs `student-fast`** | **+0.1553** | **5.70** |
| fast+residual vs `student-natural` | +0.0044 | 0.20 |

## vs prediction
Predicted ~+0.10 MOS and a transferring vuv regression. Got **+0.155** — *larger* than the +0.114 on
the slow preset — and the vuv regression transferred as expected (29.4 → 39.7).

The headline: **`student-fast` + residual is statistically tied with `student-natural`** (t = 0.20)
while being **4× faster (0.261 s vs 1.106 s) and 9× smaller (9.93 M vs 90.3 M params)**. It is the
highest UTMOS of any student configuration measured, against a teacher at 4.4773.

## Gates
- **WER 5.42 % vs 5.27 % shipped (+0.15 pp)** — the same tiny cost the residual charged on the slow
  preset. Content is intact.
- **spk-cos 0.9769** — passes the ≥0.97 bar set in cycle 76, but it is the **lowest of any preset
  here** and worth watching; the residual is nudging speaker identity in the same direction it nudges
  voicing.
- **F0 RMSE 31.8 → 43.9 is the worst regression in this cycle** and is larger, proportionally, than
  the vuv one. On the slow preset the residual barely touched F0 (16.18 → 17.96); on the fast preset
  it degrades it by 38 %. The fast path's F0 comes from the distilled F0/N student rather than the
  teacher, so the residual is compounding an error that `student` does not have.

## Trade — stated, and it is a real one
This preset buys **+0.155 MOS of naturalness for a 38 % worse F0 and a 35 % worse voicing error**
against the teacher. Those are teacher-similarity metrics, which cycle 75 showed can mislabel
improvement — but F0 is not purely a similarity metric, it is pitch accuracy, and a 12 Hz RMSE
increase is the kind of thing that reads as wobble. **UTMOS says it sounds better; F0 says the pitch
track is less accurate. Both are probably true**, and no instrument in this repo arbitrates that.
Opt-in, never default, and the F0 number goes in the preset docstring.

## Shipped
`from_preset("student-fast-natural")` — `StudentKokoro` (the fast path) with `ResMaskHead` at
`res_scale=0.01`. `student-fast` and every other preset unchanged.

## Budget
~1.5 h of the 2 h box.

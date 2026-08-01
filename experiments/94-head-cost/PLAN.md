# 94 — cost-screen candidate head shapes before training any

question:      cycle 93 set the rule: screen replacement heads on **cost first** (~0.26 s at chapter
               scale), quality second. Nothing has been priced yet — not even the current head's
               internal split. Where does `student-fast`'s 0.261 s go, and what would a
               Vocos-class head (predict magnitude+phase directly, no harmonic template) cost by
               comparison?
axis:          speed (§1), as the screening gate for the whole replacement program.
prediction:    the harmonic **template construction** (a scatter-add over 5 taps × 96 harmonics) is a
               meaningful share of the head — ≥20 % — so a Vocos-class head that drops it is
               **cheaper** than MaskHead. If so, the 45× requirement is already satisfied by a
               same-shaped head and the replacement question is purely "can it be trained to teacher
               quality", not "can it be made fast enough".
falsifier:     the template is negligible (<5 %) and the trunk/iSTFT dominate. Then dropping it saves
               nothing, any replacement of similar shape costs what MaskHead costs, and extra quality
               must come from extra capacity — which costs time linearly and puts the 45×
               requirement genuinely at risk.
budget:        2 h (stop at 4 h regardless)
controls:      - same chapter text as cycle 50, warm, median of 5.
               - components timed in isolation with `mx.eval` barriers so lazy evaluation cannot
                 shift work between them.

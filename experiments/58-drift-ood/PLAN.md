# 58 — is the drift tail an out-of-distribution problem in the capture corpus?

question:      cycle 57 concluded the `student-fast` drift failures are out-of-distribution rather
               than plumbing or accuracy. Is that actually true of the duration student's training
               data?
axis:          exactness / robustness (§1). Diagnostic; measurement decides whether a retrain is
               even worth attempting.

first look (run before this plan, reported honestly):
               at the *text* level the capture corpus is **not** starved of adversarial patterns —
               6581 texts, of which 56.0 % contain stacked punctuation, 77.2 % an ellipsis, 17.6 %
               an em-dash, and 8.2 % a repeated identical sentence. That weakens the naive OOD story
               and is why this cycle exists rather than a retrain cycle.

refined question: the duration student never sees a *text* — it sees a **chunk** (phoneme sequence +
               style). The failing eval items are short (65–70 chars) *and* punctuation-dense. So the
               hypothesis under test is the joint one: **chunks that are simultaneously short and
               punctuation-dense are rare or absent in training**, and that is the region where the
               head diverges.
prediction:    the failing eval chunks (`patho03`, `patho02`, `stress08`, `stress03`) sit in a
               sparsely-populated corner of the training distribution — under ~1 % of training
               chunks share their (length, punctuation-density) region — while the 9 bit-exact items
               sit in the dense core.
falsifier:     the failing chunks land in a well-populated region (>5 % of training chunks nearby).
               Then the data covers these inputs, the head simply predicts them badly, and the
               problem is model capacity/objective — a much more expensive cycle, and one that
               should not be started on a false premise.
budget:        2 h (stop at 4 h regardless)
controls:      - measure at the **chunk** level, the unit the model is trained and evaluated on.
               - locate the 9 bit-exact items in the same space as the contrast group.
               - no eval text is ever added to training in this cycle; nothing is retrained here.

# 59 — is the drift tail caused by the `len(ps)-1` style-pack lookup?

question:      both the fast head and the teacher index a style pack by `len(ps)-1`. Does that
               lookup — a mechanism unrelated to text distribution — drive the `student-fast`
               duration failures cycles 57–58 characterized?
axis:          exactness / robustness (§1). Diagnostic.
constraint already on the record: cycle 57's token-level comparison fed the **same** style vector
               (`fast.pack[len(ps)-1]`) to both the fast head and the teacher, and still measured
               fast=251 vs teacher=167 frames on `patho03`. So the lookup cannot explain that
               divergence *unless* the two engines' packs differ. Step 1 is therefore to check
               whether they are the same array; if they are, this hypothesis is already dead and the
               cycle becomes a sensitivity measurement instead.
prediction:    predicted total duration for `patho03` is **strongly** style-sensitive — varying the
               pack index across its range changes the total by >20 % — which would make an
               index-dependent lookup a plausible amplifier at unusual chunk lengths.
falsifier:     total duration varies by <5 % across the whole pack. Then style is not a meaningful
               lever on this failure at all, candidate (b) from cycle 58 is dead, and the remaining
               explanation is content//regression-at-the-edges (candidates (a) and (c)).
budget:        2 h (stop at 4 h regardless)
controls:      - run the identical sweep on a **bit-exact** item (`short07`) as the contrast: if the
                 exact item is equally style-sensitive, sensitivity is not what separates them.
               - compare fast-head sensitivity against teacher sensitivity on the same chunk, since
                 only a *difference* in sensitivity can produce a *divergence*.

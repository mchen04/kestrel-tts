# GOAL

Take **Kokoro-82M** and make it **100× faster and 100× smaller — without losing any quality.**

Platform: **Apple MLX**, M2, 16 GB. It's already running here at `~/Epub_Listener` (PyTorch `kokoro` and
`mlx-community/Kokoro-82M-bf16`), rendering audiobooks with the `af_heart` voice. That's the real workload.

---

## That's the whole goal. The direction is yours to find.

This is deliberately not a plan. Research it, decide the approach, change your mind when the evidence says to.

Three things are worth knowing before you start:

**Start from the current literature, not from priors.** It's July 2026 and this field moves in weeks. Anything
you already "know" about compressing speech models is probably stale. Sweep the research, check whether someone
has already shipped what you're about to build, and check whether a newer small model has simply beaten
Kokoro-82M outright — if switching models wins, that's a real answer and a good one.

**"Without losing quality" needs a definition before it can be a constraint.** Figuring out how to measure
quality — well enough that you can *prove* a 100× smaller model didn't lose any — is part of the work, and
probably the part that determines whether the rest is trustworthy. Casual listening won't catch what aggressive
compression does. Decide what "no loss" means, build something that measures it, and be skeptical of any single
number that says you succeeded.

**100× on both axes at zero loss is beyond anything published.** It may not be reachable by compression alone.
Getting there likely means questioning what's being preserved, not just how it's stored — and possibly finding
something that isn't in the literature yet. Chase it seriously. Ship the intermediate wins along the way rather
than holding out for the full 100×.

---

## Working notes

- `baseline/` `eval/` `bench/` `experiments/` `notes/` — use them however you want.
- `notes/prior-brainstorm.md` is an earlier detailed dump of ideas. **Non-binding.** Read it after forming your
  own view, or ignore it. It is not a plan and following it is not the goal.
- Measure on a quiet machine, warm runs, medians. An M2 will fabricate speedups that don't exist.
- Report both axes always — a 100× smaller model that runs at the same speed is a common, useless outcome.
- Write down what fails. On a goal this aggressive most things will, and the dead ends are the real map.

**Done means:** a configuration that is provably faster and smaller by a stated factor, with quality loss you
have genuinely tried and failed to detect — plus a drop-in MLX provider for `Epub_Listener` and a full chapter
rendered with it that a human has listened to end to end.

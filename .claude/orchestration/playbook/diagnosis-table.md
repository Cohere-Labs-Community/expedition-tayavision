# Diagnosis table — pointer

**The diagnosis table is canonical in `.claude/agents/tpu-diagnoser.md`.** This file is a
pointer, deliberately. It is not a summary, and it must never become a second copy.

`tinyaya-stage2-scale` carried the table in five places — the playbook, the agent, the
SPEC, the orchestrate skill, and the watchdog — and they drifted. By the time anyone
noticed, the playbook copy said "13 entries" while the file had 15, and the skill's row 1
described a signature that existed nowhere else. A diagnosis table that disagrees with
itself is worse than no table: it makes a confident wrong classification.

## Routing

| You want | Go to |
|---|---|
| Detection — signature → classification | `.claude/agents/tpu-diagnoser.md` |
| Policy — classification → what to do | `tier-definitions.md` |
| Recovery mechanics — how a tier is executed | `../SPEC.md` §4-5 |
| Whether it should work at all yet | `../SPEC.md` §9 |

## Adding a signature

1. Add the row to `.claude/agents/tpu-diagnoser.md` — signature, classification, action,
   tier. Nowhere else.
2. If it needs a tier that does not exist, update `tier-definitions.md` **first**. A row
   pointing at an undefined tier is unactionable.
3. If the fix is structural rather than operational, record it in `.claude/memories.md`
   too — the table says what to do now, memories says why the shape is what it is.
4. Prefer a signature that matches the **cause** over one that matches a downstream
   symptom. The worked example: a multi-host `DEADLINE_EXCEEDED` rendezvous failure was
   really an apt/dpkg lock race on two workers. Matching the rendezvous error alone cost
   an hour of looking at the wrong layer.
5. Mark provenance. Rows inherited from a sibling project are predictions here until
   observed; rows derived from this repo's code but never seen are predictions too. Say
   which.

# Agent harness (library replication)

**Intent:** Port a source library to a target stack with documented parity (semantics over literal translation).

**Assumption:** Many short sessions and sometimes parallel agents. Continuity lives in **git + `docs/`**, not chat.

## Doc layout (keep files small)

- **Index + leaves:** Each topic has a **short hub** under `docs/` that only links and summarizes. Detail lives in **subfolders**, one focused file per unit (e.g. one ADR, one parity area, one plan phase).
- **Split when long:** If a hub passes ~**120 lines** or stops being scannable, **add** `docs/<topic>/README.md` (or `00-index.md`) and move sections into `docs/<topic>/*.md`. Update the hub to link them; do not let a single `.md` become a dump.
- **Logs by time:** Prefer `docs/session-log/YYYY-MM.md` (append, newest at bottom) instead of one unbounded `SESSION_LOG.md`.
- **Naming:** Stable slugs (`docs/parity/pk-models.md`), not `notes2.md`.

Suggested hubs (create as needed):

| Hub | Role |
|-----|------|
| `docs/HARNESS.md` | Commands, env, paths, how to run tests — update when tooling changes. |
| `docs/STATE.md` | ≤~25 lines: phase, branch, blockers, next action. |
| `docs/PLAN.md` | Queue + links to `docs/plan/` tasks/phases if the queue grows. |
| `docs/PARITY.md` | Contract summary + links to `docs/parity/*.md`. |
| `docs/DECISIONS.md` | Index of ADRs; each ADR in `docs/decisions/NNN-slug.md`. |

**Rule:** Facts the next run needs → `STATE.md`, `DECISIONS`, or the relevant plan/parity leaf — not chat only.

**R reference runs:** Use **Gautschi** for real `nlmixr2` (goldens, parity, upstream checks). **Env, modules, Slurm, paths:** use the **Gautschi / nlmixr2 cluster skill** — not ad hoc local R and not duplicated setup docs in this repo. Always submit to standby qos.

## Planning & handoffs

- **Plan = queue:** Checkpoint-sized tasks, dependencies noted, done / doing / next. Large plans → one file per phase under `docs/plan/`. Don't stop if a task is not done.
- **Before stop:** Sync `STATE.md` + current plan entry; note commands + pass/fail.
- **Handoff (brief):** Done | in progress (branch, files) | next steps | risks | copy-paste commands → session log for the month.
- **Branches:** Prefer one stream per branch/task id; avoid two agents on one branch without coordination.
- **Subagents:** One task, written report into plan leaf or session log (paths + findings).

## Testing

- **TDD:** Add or extend **failing unit tests before** implementation for each new/changed function; only then write code until green.
- **Coverage per function:** **Several** tests each (typical: happy path, edge/boundary, error or invalid input, and a regression tied to `docs/parity/` when relevant). No function ships with a single test.
- **Cadence:** Run the **full test suite often** while working (after each logical chunk, before handoff). Record failing commands + output in the session log if not fixed.

## Defaults

License compliance; parity/golden checks stay aligned with `docs/parity/`; tight scope, match repo style.

## New agent boot

1. `docs/HARNESS.md` → `docs/STATE.md` → `docs/PLAN.md` (and linked leaves).  
2. Skim `docs/PARITY.md` + latest `docs/decisions/*` (invoke the when work needs R reference jobs).  
3. Next unblocked task; update STATE, plan, session log before exit.

If `docs/` is empty, add **hubs + one stub leaf** per topic so growth stays structured from day one.

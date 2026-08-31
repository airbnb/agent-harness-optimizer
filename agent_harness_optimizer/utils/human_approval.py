"""Human-in-the-loop approval helpers shared by all optimizers."""

from __future__ import annotations


def ask_bh_decision(
    auto_accepted: bool,
    auto_reason: str,
    proposal: str,
    cand_train_passed: int,
    cand_train_total: int,
    cand_holdout_passed: int,
    cand_holdout_total: int,
    cand_reliability: float,
    iteration: int,
) -> tuple[bool, str]:
    """Print BH automated decision and prompt for human override. Returns (accepted, reason)."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"[human] Iteration {iteration} — automated: {'ACCEPT' if auto_accepted else 'REJECT'}")
    print(f"[human] Reason:    {auto_reason}")
    print(
        f"[human] Candidate: train={cand_train_passed}/{cand_train_total}  "
        f"holdout={cand_holdout_passed}/{cand_holdout_total}  "
        f"reliability={cand_reliability:.3f}"
    )
    print(f"[human] Proposal:\n{proposal[:500]}")
    print(sep)
    while True:
        raw = input("[human] Override? [a=auto / y=accept / n=reject]: ").strip().lower()
        if raw in ("a", ""):
            return auto_accepted, auto_reason
        if raw == "y":
            note = input("[human] Acceptance note (optional, enter to skip): ").strip()
            return True, f"human_accept: {note}" if note else f"human_accept ({auto_reason})"
        if raw == "n":
            note = input("[human] Rejection note (optional, enter to skip): ").strip()
            return False, f"human_reject: {note}" if note else f"human_reject ({auto_reason})"
        print("[human] Enter a, y, or n.")


def ask_prism_frontier(frontier: list, generation: int) -> list:
    """Print PRISM Pareto frontier and prompt human to remove candidates. Returns kept frontier."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"[human] Generation {generation} — Pareto frontier ({len(frontier)} candidates):")
    for i, c in enumerate(frontier):
        print(
            f"  [{i}] {c.uid:<38} pass_rate={c.pass_rate:.3f}  "
            f"reliability={c.reliability:.3f}  "
            f"train={c.train_passed}/{c.train_total}  "
            f"holdout={c.holdout_passed}/{c.holdout_total}"
        )
    print(sep)
    while True:
        raw = input("[human] Remove candidates? [enter=keep all / comma-list of indices]: ").strip()
        if not raw:
            return frontier
        try:
            indices = {int(x.strip()) for x in raw.split(",") if x.strip()}
            n = len(frontier)
            if not all(0 <= idx < n for idx in indices):
                print(f"[human] Invalid indices — enter numbers 0–{n - 1}.")
                continue
            kept = [c for pos, c in enumerate(frontier) if pos not in indices]
            if not kept:
                print("[human] Cannot remove all — keeping at least one.")
                continue
            print(f"[human] Removing: {[frontier[idx].uid for idx in sorted(indices)]}")
            return kept
        except ValueError:
            print("[human] Invalid input — enter comma-separated indices or press enter.")


def ask_prompt_review(best_prompt: str, optimizer_name: str, baseline_prompt: str) -> str:
    """Show optimized prompt and ask human to accept, reject, or edit. Returns final prompt."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"[human] {optimizer_name} — optimized prompt ({len(best_prompt)} chars):")
    print(best_prompt[:600] + ("…" if len(best_prompt) > 600 else ""))
    print(sep)
    while True:
        raw = input("[human] Accept? [y=yes / n=use baseline / e=edit]: ").strip().lower()
        if raw == "y":
            return best_prompt
        if raw == "n":
            print("[human] Rejecting — baseline prompt will be used for final eval.")
            return baseline_prompt
        if raw == "e":
            print("[human] Paste replacement prompt. End with a line containing only '###END###':")
            lines: list[str] = []
            while True:
                line = input()
                if line == "###END###":
                    break
                lines.append(line)
            edited = "\n".join(lines).strip()
            if edited:
                return edited
            print("[human] Empty input — try again.")
        else:
            print("[human] Enter y, n, or e.")

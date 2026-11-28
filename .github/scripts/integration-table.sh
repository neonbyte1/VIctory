#!/bin/sh
# Integration table for the fork (see issue #6). Usage: integration-table.sh <branch> [--upstream-ref origin/main]
# The asset policy allows no .py files, so the Python lives in this shell script.
exec python3 - "$@" <<'PY'
#!/usr/bin/env python3
"""Integration table for the fork: every upstream PR merged into a branch.

Usage: integration-table.py <branch> [--upstream-ref origin/main] > integration.md

Source of truth is the git history only: every merge commit between upstream
and the branch whose subject names an upstream PR (`#N`, `pr-N`, `pull/N`)
gives the PR and the exact head we merged (the merge's second parent; the
newest merge wins when a PR was merged again at a newer head). Our own
non-merge commits that name `#N` are listed as fixes/adaptations for that PR.
Our own non-merge commits (by author) are listed after the table.
GitHub is asked only for title, author, state and current head (batched).
"""
import json
import re
import subprocess
import sys

REPO = "Marvisak/VIctory"


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main():
    branch = sys.argv[1]
    upstream = sys.argv[sys.argv.index("--upstream-ref") + 1] if "--upstream-ref" in sys.argv else "origin/main"
    base = git("merge-base", branch, upstream).strip()
    merged, fixes = {}, {}
    for line in git("log", "--topo-order", "--format=%H%x09%P%x09%s", f"{base}..{branch}").splitlines():
        sha, parents, subject = line.split("\t", 2)
        parents = parents.split()
        nums = [int(n) for n in re.findall(r"(?:#|\bpr-|pull/)(\d{2,5})\b", subject)]
        if len(parents) == 2:
            if nums:
                merged.setdefault(nums[0], (sha, parents[1]))
        else:
            for n in nums:
                fixes.setdefault(n, []).append((sha, subject))
    info = {}
    nums = sorted(merged)
    for i in range(0, len(nums), 50):
        chunk = nums[i:i + 50]
        q = " ".join(
            f'p{n}: pullRequest(number: {n}) {{ title state author {{ login }} headRefOid mergeCommit {{ oid }} }}'
            for n in chunk
        )
        out = subprocess.run(
            ["gh", "api", "graphql", "-f", f'query={{ repository(owner: "Marvisak", name: "VIctory") {{ {q} }} }}'],
            check=True, capture_output=True, text=True,
        ).stdout
        for key, pr in json.loads(out)["data"]["repository"].items():
            if pr:
                info[int(key[1:])] = pr

    print(f"Upstream PRs merged into `{branch}` (`{git('rev-parse', '--short=9', branch).strip()}`), "
          f"on upstream `{git('rev-parse', '--short=9', base).strip()}`: **{len(merged)}**.\n")
    print("| PR | Author | Title | Merged head | Upstream now | Our commits for it |")
    print("|---|---|---|---|---|---|")
    stale = gone = 0
    for n in nums:
        m_sha, head = merged[n]
        pr = info.get(n, {})
        state = pr.get("state", "?")
        now = pr.get("headRefOid") or ""
        if state == "MERGED":
            up = "**merged upstream**"
            gone += 1
        elif state == "CLOSED":
            up = "**closed**"
            gone += 1
        elif now and now != head:
            up = f"**newer `{now[:8]}`**"
            stale += 1
        else:
            up = "same"
        ours = ", ".join(f"`{s[:8]}`" for s, _ in fixes.get(n, [])) or ""
        title = (pr.get("title") or "").replace("|", "\\|")
        author = (pr.get("author") or {}).get("login", "?")
        print(f"| {REPO}#{n} | {author} | {title} | `{head[:8]}` | {up} | {ours} |")
    print(f"\n{stale} with a newer upstream head, {gone} merged or closed upstream since.")
    mine = git("log", "--no-merges", "--author=186292855+neonbyte1@users.noreply.github.com",
               "--format=%h%x09%s", f"{base}..{branch}").splitlines()
    if mine:
        print(f"\n### Our own commits ({len(mine)}; candidates for our upstream PRs or offered on the PR they fix)\n")
        for line in mine:
            h, s = line.split("\t", 1)
            print(f"- `{h}` {s}")


if __name__ == "__main__":
    main()
PY

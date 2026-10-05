"""Export a warmstart-ready training CSV from a subreddit you moderate.

Connects with a moderator account via PRAW and builds `body,label` rows
(label 1 = removed by moderators/AutoModerator, 0 = kept), the exact schema
`modmodmod-warmstart train --csv` consumes. Extra provenance columns are
included; the trainer ignores them.

Label sources, strongest first:
  1. mod log `removecomment` / `spamcomment` actions  -> label 1
  2. mod log `approvecomment` actions                 -> label 0 (explicit keep)
  3. recent comment listing, not removed              -> label 0 (implicit keep)
The latest action per comment wins (a removal later approved counts as kept),
then rows are de-duplicated by body with the last label winning, matching the
Kumar et al. corpus convention warmstart was evaluated on.

Auth (script-type app at reddit.com/prefs/apps), via environment:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD,
  REDDIT_USER_AGENT (a contact string, e.g. "modlog-export by u/yourname")
or a standard praw.ini site named in REDDIT_SITE.

Usage:
  pip install praw
  python export_modlog_csv.py --subreddit yoursub --out yoursub.csv
  python export_modlog_csv.py --subreddit yoursub --out yoursub.csv \
      --state yoursub.state.json          # incremental: only new actions
  python export_modlog_csv.py --subreddit yoursub --out yoursub.csv --balance

Notes on coverage and ethics. The mod log paginates far back, but the
implicit-keep listing only reaches the ~1000 newest comments, so run this on
a schedule (see --state) to accumulate older keeps over time. Usernames are
not exported. Bodies of deleted comments are skipped. The export is for
moderating your own community; do not redistribute comment text.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import praw

REMOVE_ACTIONS = ("removecomment", "spamcomment")
APPROVE_ACTION = "approvecomment"
BOTS = {"AutoModerator"}


def reddit_client():
    site = os.environ.get("REDDIT_SITE")
    if site:
        return praw.Reddit(site)
    need = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USERNAME",
            "REDDIT_PASSWORD", "REDDIT_USER_AGENT"]
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        sys.exit(f"missing env vars: {missing} (or set REDDIT_SITE for praw.ini)")
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        username=os.environ["REDDIT_USERNAME"],
        password=os.environ["REDDIT_PASSWORD"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def usable(body):
    return body and body not in ("[deleted]", "[removed]") and len(body.strip()) >= 3


def harvest_modlog(sub, since_utc):
    """(comment_fullname -> (utc, label, mod_kind, body_or_None)), newest action wins."""
    actions = {}
    newest = since_utc
    for kind, label in [(a, 1) for a in REMOVE_ACTIONS] + [(APPROVE_ACTION, 0)]:
        for entry in sub.mod.log(action=kind, limit=None):
            if entry.created_utc <= since_utc:
                break
            newest = max(newest, entry.created_utc)
            fid = entry.target_fullname
            if not fid or not fid.startswith("t1_"):
                continue
            prev = actions.get(fid)
            if prev is None or entry.created_utc > prev[0]:
                mod_kind = "automod" if str(entry.mod) in BOTS else "human"
                actions[fid] = (entry.created_utc, label, mod_kind, entry.target_body)
    return actions, newest


def harvest_keeps(sub, exclude_bots):
    """Implicit keeps from the newest comment listing (mod view: banned_by None)."""
    keeps = {}
    for c in sub.comments(limit=None):
        if getattr(c, "banned_by", None):
            continue
        if exclude_bots and c.author and str(c.author) in BOTS:
            continue
        if usable(c.body):
            keeps[c.fullname] = (c.created_utc, 0, "listing", c.body)
    return keeps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subreddit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", help="JSON state file enabling incremental runs")
    ap.add_argument("--balance", action="store_true",
                    help="downsample the majority class to a 50/50 CSV")
    ap.add_argument("--exclude-bots", action="store_true",
                    help="drop AutoModerator-authored comments from the keep side")
    args = ap.parse_args()

    reddit = reddit_client()
    sub = reddit.subreddit(args.subreddit)
    me = str(reddit.user.me())
    if me not in [str(m) for m in sub.moderator()]:
        sys.exit(f"u/{me} does not moderate r/{args.subreddit}")

    since = 0.0
    seen = {}
    out = Path(args.out)
    if args.state and Path(args.state).exists():
        st = json.loads(Path(args.state).read_text())
        since = st.get("since_utc", 0.0)
    if out.exists():
        with out.open() as f:
            for row in csv.DictReader(f):
                seen[row["comment_id"]] = row

    print(f"r/{args.subreddit}: pulling mod log since {since or 'beginning'}", flush=True)
    actions, newest = harvest_modlog(sub, since)
    print(f"  mod-log actions: {len(actions)}", flush=True)
    keeps = harvest_keeps(sub, args.exclude_bots)
    print(f"  listed keeps: {len(keeps)}", flush=True)

    for fid, (utc, label, kind, body) in {**keeps, **actions}.items():
        if body is None or not usable(body):
            continue
        seen[fid] = {"comment_id": fid, "created_utc": int(utc), "label": label,
                     "source": kind, "subreddit": args.subreddit, "body": body}

    rows = sorted(seen.values(), key=lambda r: int(r["created_utc"]))
    bybody = {}
    for r in rows:
        bybody[r["body"]] = r
    rows = list(bybody.values())

    if args.balance:
        pos = [r for r in rows if int(r["label"]) == 1]
        neg = [r for r in rows if int(r["label"]) == 0]
        n = min(len(pos), len(neg))
        rows = sorted(pos, key=lambda r: -int(r["created_utc"]))[:n] + \
               sorted(neg, key=lambda r: -int(r["created_utc"]))[:n]

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["body", "label", "comment_id",
                                          "created_utc", "source", "subreddit"])
        w.writeheader()
        w.writerows(rows)
    npos = sum(1 for r in rows if int(r["label"]) == 1)
    print(f"wrote {out}: {len(rows)} rows ({npos} removed / {len(rows) - npos} kept)")
    if args.state:
        Path(args.state).write_text(json.dumps(
            {"since_utc": newest, "updated": int(time.time())}))
        print(f"state -> {args.state} (re-run on a schedule to keep learning)")
    if npos < 50 or len(rows) - npos < 50:
        print("note: fewer than 50 rows in one class; expect a weak head. "
              "Consider the coldstart global head until more labels accumulate.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scrub-repo.py — remove identifying details from the whole repo before publishing.

Walks every text file under --root (results, logs, scripts, docs; skips .git/, __pycache__,
binaries and this script) and rewrites:
    /home/<user>            -> ~
    <user>@<host>           -> user@dgx-spark
    <host>                  -> dgx-spark
    <user> (whole word)     -> user
    private IPs             -> <lan-ip>        (10.x, 172.16-31.x, 192.168.x, plus any --ip)
    --handle / --email      -> <hf-user> / <email>
    --extra "string"        -> <redacted>      (repeatable)
Mentions of Claude / Anthropic are reported; with --strip-ai the containing lines are removed.
Run with --dry-run first. Run it BEFORE the first git commit (history keeps whatever was committed).

  scripts/scrub-repo.py --user bytex --host spark-bee55 --handle ntlhzx --dry-run
  scripts/scrub-repo.py --user bytex --host spark-bee55 --handle ntlhzx --strip-ai
"""
import argparse
import os
import re
import sys

SKIP_DIRS = {".git", "__pycache__", "node_modules"}
SKIP_EXT = {".pyc", ".tgz", ".gz", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".safetensors", ".bin"}
AI_RE = re.compile(r"claude|anthropic", re.IGNORECASE)
PRIVATE_IP_RE = re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def is_text(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"\0" not in chunk
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--user", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--ip", action="append", default=[])
    ap.add_argument("--handle", action="append", default=[], help="public handle(s) to redact, e.g. your HF user")
    ap.add_argument("--email", action="append", default=[], help="specific e-mail(s); any e-mail address is redacted anyway")
    ap.add_argument("--extra", action="append", default=[], help="any other literal string to redact")
    ap.add_argument("--strip-ai", action="store_true", help="delete lines mentioning Claude/Anthropic (default: report only)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    me = os.path.abspath(__file__)

    subs = [(re.compile(re.escape(f"/home/{a.user}")), "~"),
            (re.compile(re.escape(f"{a.user}@{a.host}")), "user@dgx-spark"),
            (re.compile(re.escape(a.host)), "dgx-spark"),
            (re.compile(r"\b" + re.escape(a.user) + r"\b"), "user")]
    subs += [(re.compile(re.escape(ip)), "<lan-ip>") for ip in a.ip]
    subs += [(PRIVATE_IP_RE, "<lan-ip>")]
    subs += [(re.compile(r"\b" + re.escape(h) + r"\b"), "<hf-user>") for h in a.handle]
    subs += [(re.compile(re.escape(e)), "<email>") for e in a.email]
    subs += [(EMAIL_RE, "<email>")]
    subs += [(re.compile(re.escape(x)), "<redacted>") for x in a.extra]

    total, ai_hits = 0, []
    for dp, dns, fns in os.walk(a.root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.abspath(p) == me or os.path.splitext(fn)[1].lower() in SKIP_EXT or not is_text(p):
                continue
            try:
                s = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            t, n = s, 0
            for rx, rep in subs:
                t, k = rx.subn(rep, t)
                n += k
            lines = t.splitlines(keepends=True)
            ai_lines = [ln for ln in lines if AI_RE.search(ln)]
            if ai_lines:
                ai_hits.append((p, len(ai_lines)))
                if a.strip_ai:
                    t = "".join(ln for ln in lines if not AI_RE.search(ln))
                    n += len(ai_lines)
            if n:
                total += n
                print(f"{n:5d}  {p}")
                if not a.dry_run:
                    open(p, "w", encoding="utf-8").write(t)
    print(f"{'would change' if a.dry_run else 'changed'} {total} occurrences under {os.path.abspath(a.root)}")
    if ai_hits:
        print(("removed" if a.strip_ai and not a.dry_run else "FOUND (use --strip-ai to remove)") + " Claude/Anthropic mentions:",
              ", ".join(f"{p} ({k} lines)" for p, k in ai_hits))

    # final scan: anything left?
    needles = [a.user, a.host] + a.ip + a.handle + a.email + a.extra
    left = []
    for dp, dns, fns in os.walk(a.root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = os.path.join(dp, fn)
            if os.path.abspath(p) == me or os.path.splitext(fn)[1].lower() in SKIP_EXT or not is_text(p):
                continue
            try:
                s = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            hits = [x for x in needles if x and re.search(r"\b" + re.escape(x) + r"\b", s)]
            if PRIVATE_IP_RE.search(s):
                hits.append("private-ip")
            if AI_RE.search(s) and not (a.strip_ai and not a.dry_run):
                hits.append("claude/anthropic")
            if hits:
                left.append(f"{p}: {', '.join(hits)}")
    if left and not a.dry_run:
        print("STILL PRESENT — check by hand:")
        for x in left:
            print("   " + x)
        return 1
    print("final scan: clean" if not a.dry_run else "dry run only; nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

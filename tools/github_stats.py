#!/usr/bin/env python3
"""Query GitHub and generate release-note statistics.

Adapted from the dipy / IPython projects (BSD-3 licence).

Usage
-----
Print stats since a tag::

    python tools/github_stats.py 0.1.0

Print stats for the last N days::

    python tools/github_stats.py 30

Set GITHUB_TOKEN in your environment for an authenticated request
(higher rate limit).

Returns RST-formatted output suitable for inserting into CHANGES.rst.
"""

import argparse
from datetime import datetime, timedelta
import json
import os
import re
from subprocess import check_output
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100
REPO = "fury-gl/polyxios"

_element_pat = re.compile(r"<(.+?)>")
_rel_pat = re.compile(r'rel=[\'"](\w+)[\'"]')


def _api_request(url):
    token = os.environ.get("GITHUB_TOKEN")
    req = Request(url)
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    return urlopen(req)


def _parse_link_header(headers):
    link_s = headers.get("link", "")
    urls = _element_pat.findall(link_s)
    rels = _rel_pat.findall(link_s)
    return dict(zip(rels, urls))


def _get_paged(url):
    results = []
    while url:
        print(f"  fetching {url}", file=sys.stderr)
        resp = _api_request(url)
        results.extend(json.load(resp))
        url = _parse_link_header(resp.headers).get("next")
    return results


def _get_closed(*, pulls, since, branch=None):
    which = "pulls" if pulls else "issues"
    url = (
        f"https://api.github.com/repos/{REPO}/{which}"
        f"?state=closed&sort=updated&since={since.strftime(ISO8601)}"
        f"&per_page={PER_PAGE}"
    )
    if pulls and branch:
        url += f"&base={quote(branch, safe='')}"
    items = _get_paged(url)
    items = [i for i in items if datetime.strptime(i["closed_at"], ISO8601) > since]
    if pulls:
        items = [i for i in items if i.get("merged_at")]
    else:
        items = [i for i in items if "pull_request" not in i]
    return sorted(items, key=lambda i: i["closed_at"], reverse=True)


def generate_stats(*, since_tag=None, since_days=None, branch=None):
    """Return RST-formatted release statistics.

    Parameters
    ----------
    since_tag : str, optional
        Git tag marking the start of the period (e.g. ``"0.1.0"``).
        Mutually exclusive with *since_days*.
    since_days : int, optional
        Number of days to look back.  Used when no tag is given.
    branch : str, optional
        Restrict merged PRs to those targeting this branch.

    Returns
    -------
    str
        RST block ready to insert into CHANGES.rst.
    """
    if since_tag:
        raw = check_output(
            ["git", "log", "-1", "--format=%ai", since_tag], text=True
        ).strip()
        tagday = raw.rsplit(" ", 1)[0]
        since = datetime.strptime(tagday, "%Y-%m-%d %H:%M:%S")
    else:
        since = datetime.now() - timedelta(days=since_days or 30)

    issues = _get_closed(pulls=False, since=since, branch=branch)
    pulls = _get_closed(pulls=True, since=since, branch=branch)

    since_tag_arg = f"{since_tag}.." if since_tag else ""
    log_cmd = ["git", "log", "--oneline", since_tag_arg]
    author_cmd = ["git", "log", "--format=* %aN", since_tag_arg]
    if branch:
        log_cmd.append(branch)
        author_cmd.append(branch)

    ncommits = len(check_output(log_cmd, text=True).splitlines())
    unique_authors = sorted(set(check_output(author_cmd, text=True).splitlines()))

    today = datetime.today().strftime("%Y/%m/%d")
    since_str = since.strftime("%Y/%m/%d")

    lines = [
        f"GitHub stats for {since_str} - {today} (tag: {since_tag})",
        "",
        "These lists are automatically generated and may be incomplete or "
        "contain duplicates.",
        "",
    ]

    if unique_authors:
        lines += [
            f"The following {len(unique_authors)} authors contributed "
            f"{ncommits} commits.",
            "",
            *unique_authors,
            "",
            "",
        ]

        if branch:
            lines.append(
                f"We closed a total of {len(pulls)} pull requests "
                f"(merged into ``{branch}``)."
            )
        else:
            lines.append(
                f"We closed a total of {len(issues) + len(pulls)} issues, "
                f"{len(pulls)} pull requests and {len(issues)} regular issues."
            )
        lines += ["", f"Pull Requests ({len(pulls)}):", ""]
        lines += [f"* :ghpull:`{pr['number']}`: {pr['title']}" for pr in pulls]

        if not branch:
            lines += ["", f"Issues ({len(issues)}):", ""]
            lines += [f"* :ghissue:`{i['number']}`: {i['title']}" for i in issues]

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tag_or_days",
        nargs="?",
        metavar="TAG_OR_DAYS",
        help="Git tag (e.g. '0.1.0') or number of days to look back.",
    )
    parser.add_argument("--branch", "-b", default=None, metavar="BRANCH")
    args = parser.parse_args()

    since_tag = None
    since_days = None

    if args.tag_or_days is None:
        since_tag = check_output(["git", "describe", "--abbrev=0"], text=True).strip()
    else:
        try:
            since_days = int(args.tag_or_days)
        except ValueError:
            since_tag = args.tag_or_days

    print(
        generate_stats(since_tag=since_tag, since_days=since_days, branch=args.branch)
    )

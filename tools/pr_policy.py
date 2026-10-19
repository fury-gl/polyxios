#!/usr/bin/env python3
"""Issue-first policy for pull requests.

Run by ``.github/workflows/pr_policy.yml`` from the base branch, under
``pull_request_target`` and ``issues`` events. It reads the event payload and
calls the GitHub API; it never checks out or executes code from a pull
request. Standard library only, so the workflow needs no Python setup.

A pull request from outside the maintainer team must reference an open issue
in this repository that is assigned to its author. Otherwise the policy posts
one sticky comment, labels the pull request ``needs-issue`` and closes it.
A closed issue no longer counts, from the next event on the pull request; the
policy does not watch issues close. If GitHub refuses to reopen a pull request
whose head branch or fork is gone, the policy says so and lets it go. Editing
the description or assigning the author reopens it; unassigning the author
closes it again. A maintainer who reopens a held pull request waives the
check for it. Logins listed in ``.github/community-members.txt`` skip the
check.

The verdict is published as the ``pr-policy`` commit status on the head
commit, not as the outcome of the job: an ``issues`` run is not attached to
the pull requests it re-checks, so only a status can turn a held pull request
green again. The job itself fails only when the GitHub API does.

Every run re-reads the pull request from the API rather than trusting the
event payload. Runs that overlap (an edit and an assignment at the same
moment) sit in different concurrency groups, so a run that closes or reopens
a pull request reads its issues again afterwards and starts over when an
assignment changed meanwhile: whichever run changes the state last has seen
the assignment the other one reacted to. The comment and the status of a
pull request whose state did not change can lag behind until its next event.

Only pull requests into the default branch are checked, whatever it is
named; the rest are skipped without a label, comment or status.

Usage
-----
Inside GitHub Actions, with ``GITHUB_TOKEN`` set::

    python3 tools/pr_policy.py
"""

import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

MARKER = "<!-- polyxios-pr-policy -->"
LABEL = "needs-issue"
DOCS = "https://polyxios.org/dev/contributing.html"
STATUS_CONTEXT = "pr-policy"
POLICY_ACTOR = "github-actions[bot]"
TRUSTED = frozenset({"OWNER", "MEMBER"})
WRITE_PERMISSIONS = frozenset({"admin", "write"})
RETURNING = TRUSTED | {"COLLABORATOR", "CONTRIBUTOR"}
MEMBERS_FILE = Path(__file__).resolve().parents[1] / ".github/community-members.txt"
MEMBERSHIP_CATEGORY = "community-membership"
PER_PAGE = 100
MAX_REFS = 10

_URL_REF = re.compile(
    r"https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)/(issues|pull|discussions)/(\d+)",
    re.IGNORECASE,
)
_GH_REF = re.compile(r"(?<![\w-])(?i:GH)-(\d+)\b")
# An owner starts only at a token boundary: one that could start anywhere
# would rescan a long run of `.` or `-` from every position, quadratic in the
# description.
_SHORT_REF = re.compile(r"(?:(?<![\w./-])([\w.-]+)/([\w.-]+)|(?<![\w/]))#(\d+)\b")
# One alternation so the construct that opens first wins: a `<!--` inside
# code is literal text, and a fence inside a comment is not a fence. A fence
# opens and closes only at the start of a line (after any list or quote
# marker), and only a line of the same character, at least as long and with
# nothing after it, closes it.
_NOISE = re.compile(
    r"<!--.*?(?:-->|\Z)"
    r"|^[ \t>]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)?"
    r"(?:(?P<bt>`{3,})[^`\n]*(?:\n.*?)??(?:\n[ \t>]*(?P=bt)`*[ \t]*$|\Z)"
    r"|(?P<tl>~{3,})[^\n]*(?:\n.*?)??(?:\n[ \t>]*(?P=tl)~*[ \t]*$|\Z))"
    r"|(?<!`)(?P<tick>`+)(?!`)[^\n]*?(?<!`)(?P=tick)(?!`)",
    re.DOTALL | re.MULTILINE,
)
_KEYWORD = re.compile(
    r"(?<![\w-])(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?):?[ \t]+(\S+)",
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r" {0,3}(?:[-*+]|\d{1,9}[.)])(?:[ \t]|$)")
_STAMP = re.compile(r"<!-- polyxios-pr-policy closed_at=(\S+) -->")
_WAIVER = re.compile(r"<!-- polyxios-pr-policy waived_by=(\S+) -->")
REOPEN_NOTE = (
    "Reopened automatically. GitHub starts no CI for a reopen made by a "
    "workflow: if you pushed while this pull request was closed, push again "
    "or ask a maintainer to re-run the checks."
)
STUCK_NOTE = (
    "GitHub refused to reopen this pull request, usually because its head "
    "branch or fork was deleted. Open a new pull request from a branch that "
    "exists."
)
REOPENED = "reopened"
STUCK = "stuck"

_CLOSING_QUERY = """
query($owner: String!, $repo: String!) {
  repository(owner: $owner, name: $repo) {
%s
  }
}
"""
_CLOSING_FIELD = """    pr%d: pullRequest(number: %d) {
      closingIssuesReferences(first: 20) {
        nodes { number repository { nameWithOwner } }
      }
    }"""
CLOSING_BATCH = 50


class ApiError(Exception):
    """A GitHub API request that failed.

    Parameters
    ----------
    status : int
        HTTP status code, 200 for a GraphQL response carrying errors, or 0
        when the request never got a response.
    message : str
        Response body or GraphQL error text.
    """

    def __init__(self, *, status, message):
        super().__init__(f"{status}: {message}")
        self.status = status


def make_call(*, token, api_url):
    """Build an authenticated GitHub REST and GraphQL caller.

    Parameters
    ----------
    token : str
        Token sent as a bearer credential.
    api_url : str
        API root, such as ``https://api.github.com``.

    Returns
    -------
    callable
        ``call(method, path, *, params=None, payload=None)`` returning the
        decoded JSON response, or None for an empty body. The caller raises
        `ApiError` on an HTTP or network error; building it raises nothing.
    """

    def call(method, path, *, params=None, payload=None):
        url = api_url.rstrip("/") + path
        if params:
            url += "?" + urlencode(params)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, method=method, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read()
        except HTTPError as err:
            raise ApiError(
                status=err.code, message=err.read().decode(errors="replace")
            ) from err
        except (URLError, TimeoutError) as err:
            raise ApiError(status=0, message=str(err)) from err
        return json.loads(body) if body else None

    return call


def graphql(call, query, *, variables, missing_ok=False):
    """Run a GraphQL query.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    query : str
        GraphQL document.
    variables : dict
        Query variables.
    missing_ok : bool
        Whether ``NOT_FOUND`` errors are tolerated, leaving the missing
        fields null in the returned data.

    Returns
    -------
    dict
        The ``data`` member of the response.

    Raises
    ------
    ApiError
        When the request fails or the response carries GraphQL errors.
    """
    response = call(
        "POST", "/graphql", payload={"query": query, "variables": variables}
    )
    errors = [
        e
        for e in response.get("errors") or []
        if not (missing_ok and e.get("type") == "NOT_FOUND")
    ]
    if errors:
        raise ApiError(status=200, message="; ".join(e["message"] for e in errors))
    return response["data"]


def paginate(call, path, *, params):
    """Collect every page of a REST list endpoint.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    path : str
        Endpoint path.
    params : dict
        Query parameters, without paging.

    Returns
    -------
    list
        All items, in API order.
    """
    items = []
    page = 1
    while True:
        batch = call("GET", path, params={**params, "per_page": PER_PAGE, "page": page})
        items.extend(batch)
        if len(batch) < PER_PAGE:
            return items
        page += 1


def call_tolerant(call, method, path, *, ok, payload=None):
    """Send a write whose listed failures mean the goal is already reached.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    method, path : str
        Request.
    ok : tuple of int
        HTTP statuses to treat as success, such as 404 for deleting
        something an overlapping run or a person already removed.
    payload : dict or None
        JSON body.

    Raises
    ------
    ApiError
        When the request fails with a status outside ``ok``.
    """
    try:
        call(method, path, payload=payload)
    except ApiError as err:
        if err.status not in ok:
            raise


def load_members(path):
    """Read the community member logins.

    Parameters
    ----------
    path : pathlib.Path
        File with one login per line; ``#`` starts a comment and a leading
        ``@`` is dropped.

    Returns
    -------
    frozenset of str
        Lower-cased logins, empty when the file does not exist.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return frozenset()
    logins = (line.split("#", 1)[0].strip().lstrip("@") for line in text.splitlines())
    return frozenset(login.lower() for login in logins if login)


def strip_noise(body):
    """Remove HTML comments and code from a description.

    Parameters
    ----------
    body : str or None
        Pull request description.

    Returns
    -------
    str
        The text a reference is searched in, so a template example or a
        code sample never counts as a link. Whichever of a comment or code
        opens first hides the other, and one left open hides the rest of
        the description, as GitHub renders it.
    """
    text = (body or "").replace("\r\n", "\n")
    return strip_indented_code(_NOISE.sub("", text))


def strip_indented_code(text):
    """Remove indented code blocks, keeping indented list continuations.

    Parameters
    ----------
    text : str
        Markdown with ``\\n`` line endings.

    Returns
    -------
    str
        ``text`` without the lines GitHub renders as an indented code block:
        a run of lines indented by four spaces or a tab that follows a blank
        line, or starts the text, outside a list. Inside a list the same
        indentation continues the item, which GitHub renders and links.
    """
    kept = []
    in_list = False
    in_code = False
    after_blank = True
    for line in text.split("\n"):
        blank = not line.strip()
        indented = line.startswith(("    ", "\t"))
        if in_code and (blank or indented):
            continue
        in_code = not blank and indented and after_blank and not in_list
        if in_code:
            continue
        if _LIST_ITEM.match(line):
            in_list = True
        elif not blank and not indented and after_blank:
            in_list = False
        kept.append(line)
        after_blank = blank
    return "\n".join(kept)


def extract_refs(body, *, owner, repo):
    """Find references to this repository in a description.

    Parameters
    ----------
    body : str or None
        Pull request description.
    owner, repo : str
        Repository the references must point to.

    Returns
    -------
    dict of int to str
        Number to kind: ``"issue"`` for an issue or pull URL,
        ``"discussion"`` for a discussion URL, ``"any"`` for a bare ``#N``.
        References right after a closing keyword (``Closes #12``) come
        first, so passing mentions cannot push them past `MAX_REFS`.
    """
    text = strip_noise(body)
    slug = f"{owner}/{repo}".lower()
    refs = scan_refs(text, slug=slug)
    closed = {
        number for m in _KEYWORD.finditer(text) for number in scan_refs(m[1], slug=slug)
    }
    return {
        **{n: kind for n, kind in refs.items() if n in closed},
        **{n: kind for n, kind in refs.items() if n not in closed},
    }


def scan_refs(text, *, slug):
    """Find references to a repository in text already stripped of noise.

    Parameters
    ----------
    text : str
        Text from `strip_noise`.
    slug : str
        Lower-cased ``owner/repo`` the references must point to.

    Returns
    -------
    dict of int to str
        Number to kind, as `extract_refs` describes, in order of appearance
        within each reference form.
    """
    refs = {}

    def add(number, kind):
        if refs.get(number, "any") == "any":
            refs[number] = kind

    for m in _URL_REF.finditer(text):
        if f"{m[1]}/{m[2]}".lower() == slug:
            add(int(m[4]), "discussion" if m[3].lower() == "discussions" else "issue")
    text = _URL_REF.sub("", text)
    for m in _SHORT_REF.finditer(text):
        if m[1] is None or f"{m[1]}/{m[2]}".lower() == slug:
            add(int(m[3]), "any")
    for m in _GH_REF.finditer(text):
        add(int(m[1]), "any")
    return refs


def closing_refs(call, *, owner, repo, numbers):
    """List issues linked to pull requests through their sidebar.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    numbers : list of int
        Pull request numbers, looked up `CLOSING_BATCH` per GraphQL query.

    Returns
    -------
    dict of int to list of int
        Pull request number to the issue numbers in this repository; a pull
        request deleted since it was listed links none, so it does not fail
        the rest of its batch.
    """
    slug = f"{owner}/{repo}".lower()
    linked = {}
    for start in range(0, len(numbers), CLOSING_BATCH):
        batch = numbers[start : start + CLOSING_BATCH]
        fields = "\n".join(_CLOSING_FIELD % (i, int(n)) for i, n in enumerate(batch))
        data = graphql(
            call,
            _CLOSING_QUERY % fields,
            variables={"owner": owner, "repo": repo},
            missing_ok=True,
        )
        for i, number in enumerate(batch):
            found = data["repository"][f"pr{i}"]
            nodes = found["closingIssuesReferences"]["nodes"] if found else []
            linked[number] = [
                n["number"]
                for n in nodes
                if n["repository"]["nameWithOwner"].lower() == slug
            ]
    return linked


def check_ref(call, *, owner, repo, number, kind, author):
    """Decide whether a reference is an open issue assigned to the author.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    number : int
        Referenced number.
    kind : str
        Kind from `extract_refs`.
    author : str
        Pull request author login.

    Returns
    -------
    problem : str or None
        Why the reference does not qualify, or None when it does.
    is_issue : bool
        Whether the reference is an issue of this repository, the only kind
        whose verdict an assignment can change.
    """
    assign = (
        "Discussions cannot be assigned: once the change is agreed there, a "
        "maintainer opens an issue and assigns it to you."
    )
    if kind == "discussion":
        return f"#{number} is a discussion. {assign}", False
    try:
        item = call("GET", f"/repos/{owner}/{repo}/issues/{number}")
    except ApiError as err:
        if err.status not in (404, 410):
            raise
        if kind == "any":
            return (
                f"#{number} is not an issue: it is a discussion or does not "
                f"exist. {assign}"
            ), False
        return f"#{number} does not exist.", False
    if (
        not item.get("repository_url", "")
        .lower()
        .endswith(f"/repos/{owner}/{repo}".lower())
    ):
        return f"#{number} has been transferred to another repository.", False
    if item.get("pull_request") is not None:
        return f"#{number} is a pull request, not an issue.", False
    if item.get("state") == "closed":
        return (
            f"Issue #{number} is closed. Reference an open issue assigned to you."
        ), True
    assignees = {a["login"].lower() for a in item.get("assignees") or []}
    if author.lower() not in assignees:
        return (
            f"Issue #{number} is not assigned to you. Comment on it to ask a "
            "maintainer to assign you."
        ), True
    return None, True


def any_assigned(call, *, owner, repo, numbers, author):
    """Tell whether any of some issues is assigned to the author now.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    numbers : list of int
        Issue numbers, as found by `check_ref`.
    author : str
        Pull request author login.

    Returns
    -------
    bool
    """
    where = {"call": call, "owner": owner, "repo": repo, "author": author}
    return any(check_ref(**where, number=n, kind="issue")[0] is None for n in numbers)


def welcome_text(author):
    """Write the welcome section for a first-time contributor.

    Parameters
    ----------
    author : str
        Pull request author login.

    Returns
    -------
    str
        Markdown.
    """
    return "\n".join(
        [
            f"### Welcome to polyxios, @{author}!",
            "",
            "Thanks for your first pull request. Every change to polyxios starts "
            "as an issue that a maintainer has assigned to its author, and the "
            "pull request links it (`Closes #123`).",
            "",
            f"Please read the [contributing guide]({DOCS}) before your next push: "
            "it covers commit message prefixes, the CI gates, the code style, and "
            "your responsibility for every line of AI-assisted work.",
        ]
    )


def status_text(author, *, linked, problems, repo_url):
    """Write the policy verdict section.

    Parameters
    ----------
    author : str
        Pull request author login.
    linked : list of int
        Qualifying issue numbers.
    problems : list of str or None
        Reasons the pull request is held, or None when it passes.
    repo_url : str
        Repository web URL.

    Returns
    -------
    str
        Markdown.
    """
    if problems is None:
        issues = ", ".join(f"#{n}" for n in linked)
        return (
            f":white_check_mark: Linked to issue {issues}, assigned to @{author}. "
            "A maintainer will review soon."
        )
    join_url = f"{repo_url}/discussions/new?category={MEMBERSHIP_CATEGORY}"
    return "\n".join(
        [
            f":no_entry: @{author}, this pull request has been closed because it "
            "does not link an issue assigned to you.",
            "",
            *(f"- {p}" for p in problems),
            "",
            "polyxios asks that every change is agreed on before it is written. "
            "To go on:",
            "",
            f"1. Find or [open an issue]({repo_url}/issues/new) describing the "
            "problem and the change you propose. Ideas can start as a "
            f"[discussion]({repo_url}/discussions/new/choose).",
            "2. Comment on the issue to ask to work on it. A maintainer assigns it "
            "to you.",
            "3. Reference it in this pull request description, for example "
            "`Closes #123`.",
            "",
            "The pull request reopens automatically once it references an issue "
            "assigned to you, whether you edit the description or get assigned.",
            "",
            "<sub>Contributing regularly? Members of the polyxios community skip "
            f"this check. [Ask to join]({join_url}) in Discussions; no write "
            "access is involved.</sub>",
        ]
    )


def upsert_comment(
    call, *, owner, repo, number, body, create, mine, closed_at=None, waived_by=None
):
    """Create or update the policy's sticky comment.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    number : int
        Pull request number.
    body : str
        Comment text, without the marker.
    create : bool
        Whether to post a new comment when none exists yet.
    mine : list of dict
        The policy's current comments, from `policy_comments`.
    closed_at : str or None
        ``closed_at`` of a close made by the policy, stamped into the comment
        so `closed_by_policy` can tell it from a later close by anyone else.
    waived_by : str or None
        Login of the maintainer who waived the check, stamped into the
        comment so `read_waiver` finds it on every later run.

    Notes
    -----
    Two overlapping runs can both find no comment and both post one, so
    every run keeps the oldest marked comment, writes its own text into it
    and deletes the others.
    """
    stamp = f"<!-- polyxios-pr-policy closed_at={closed_at} -->\n" if closed_at else ""
    if waived_by:
        stamp += f"<!-- polyxios-pr-policy waived_by={waived_by} -->\n"
    full = f"{MARKER}\n{stamp}{body}"
    if not mine and create:
        call(
            "POST",
            f"/repos/{owner}/{repo}/issues/{number}/comments",
            payload={"body": full},
        )
        mine = policy_comments(call, owner=owner, repo=repo, number=number)
    if mine and mine[0]["body"] != full:
        call(
            "PATCH",
            f"/repos/{owner}/{repo}/issues/comments/{mine[0]['id']}",
            payload={"body": full},
        )
    for extra in mine[1:]:
        call_tolerant(
            call,
            "DELETE",
            f"/repos/{owner}/{repo}/issues/comments/{extra['id']}",
            ok=(404,),
        )


def policy_comments(call, *, owner, repo, number):
    """List the policy's own comments on a pull request, oldest first.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    number : int
        Pull request number.

    Returns
    -------
    list of dict
        Comments by `POLICY_ACTOR` carrying the marker. Other bots are left
        out: one that quotes a description could otherwise forge a stamp.
    """
    comments = paginate(
        call, f"/repos/{owner}/{repo}/issues/{number}/comments", params={}
    )
    mine = [
        c
        for c in comments
        if (c.get("user") or {}).get("login") == POLICY_ACTOR
        and MARKER in (c["body"] or "")
    ]
    return sorted(mine, key=lambda c: c["id"])


def read_stamp(mine):
    """Read the close stamp from the policy's sticky comment.

    Parameters
    ----------
    mine : list of dict
        The policy's comments, from `policy_comments`.

    Returns
    -------
    str or None
        ``closed_at`` of the policy's last close, or None when the comment
        carries no stamp.
    """
    match = _STAMP.search(mine[0]["body"]) if mine else None
    return match[1] if match else None


def read_waiver(mine):
    """Read the maintainer waiver from the policy's sticky comment.

    Parameters
    ----------
    mine : list of dict
        The policy's comments, from `policy_comments`.

    Returns
    -------
    str or None
        Login of the maintainer who waived the check, or None.
    """
    match = _WAIVER.search(mine[0]["body"]) if mine else None
    return match[1] if match else None


def has_write_access(call, *, owner, repo, login):
    """Tell whether a user can push to the repository.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    login : str
        User login.

    Returns
    -------
    bool
        True when the base ``permission`` is ``admin`` or ``write``, which
        covers the ``maintain`` role and custom roles built on ``write``,
        whose ``role_name`` is their own. Catches organization members whose
        membership is private, whom ``author_association`` reports as
        ``CONTRIBUTOR``.
    """
    try:
        found = call(
            "GET", f"/repos/{owner}/{repo}/collaborators/{quote(login)}/permission"
        )
    except ApiError as err:
        if err.status != 404:
            raise
        return False
    return (found or {}).get("permission") in WRITE_PERMISSIONS


def ensure_label(call, *, owner, repo):
    """Create the ``needs-issue`` label if the repository lacks it.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    """
    try:
        call("GET", f"/repos/{owner}/{repo}/labels/{quote(LABEL)}")
    except ApiError as err:
        if err.status != 404:
            raise
        call_tolerant(
            call,
            "POST",
            f"/repos/{owner}/{repo}/labels",
            ok=(422,),
            payload={
                "name": LABEL,
                "color": "b60205",
                "description": "Pull request does not link an issue assigned to its "
                "author",
            },
        )


def has_label(pr):
    """Tell whether the policy is holding a pull request.

    Parameters
    ----------
    pr : dict
        Pull request object.

    Returns
    -------
    bool
    """
    return any(label["name"] == LABEL for label in pr.get("labels") or [])


def closed_by_policy(call, *, owner, repo, pr, stamp):
    """Tell whether the last close of a pull request was the policy's.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    pr : dict
        Pull request object.
    stamp : str or None
        Close stamp from `read_stamp`.

    Returns
    -------
    bool
        False when a person or another workflow closed it last, so their
        decision is never undone by an assignment or an edit.

    Notes
    -----
    A stamp decides alone: it matches ``closed_at`` only while the policy's
    own close is the latest. Without one (the comment write after a close
    failed) the last ``closed`` event must come from `POLICY_ACTOR`, which
    every workflow using ``GITHUB_TOKEN`` shares.
    """
    if stamp is not None:
        return stamp == pr.get("closed_at")
    number = pr["number"]
    events = paginate(call, f"/repos/{owner}/{repo}/issues/{number}/events", params={})
    closes = [e for e in events if e.get("event") == "closed"]
    return bool(closes) and (closes[-1].get("actor") or {}).get("login") == POLICY_ACTOR


def restore(call, *, owner, repo, pr, stamp):
    """Lift a hold: reopen what the policy closed and drop the label.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    pr : dict
        Pull request object.
    stamp : str or None
        Close stamp from `read_stamp`.

    Returns
    -------
    str or None
        `REOPENED` when the pull request was reopened, `STUCK` when GitHub
        refused to reopen it (its head branch or fork is gone), None
        otherwise. One that a person or another workflow closed since stays
        closed. Every outcome drops the label, since the policy no longer
        holds the pull request; a stuck one is then left alone instead of
        failing again on every later run.
    """
    if not has_label(pr) or pr.get("merged"):
        return None
    number = pr["number"]
    outcome = None
    if pr["state"] == "closed" and closed_by_policy(
        call, owner=owner, repo=repo, pr=pr, stamp=stamp
    ):
        try:
            call(
                "PATCH",
                f"/repos/{owner}/{repo}/pulls/{number}",
                payload={"state": "open"},
            )
            outcome = REOPENED
        except ApiError as err:
            if err.status != 422:
                raise
            outcome = STUCK
    call_tolerant(
        call,
        "DELETE",
        f"/repos/{owner}/{repo}/issues/{number}/labels/{quote(LABEL)}",
        ok=(404,),
    )
    return outcome


def restore_note(outcome):
    """Write the line telling the author how a lifted hold ended.

    Parameters
    ----------
    outcome : str or None
        Result of `restore`.

    Returns
    -------
    str
        Markdown starting with a blank line, or an empty string when the
        pull request was neither reopened nor refused.
    """
    notes = {REOPENED: REOPEN_NOTE, STUCK: STUCK_NOTE}
    return f"\n\n{notes[outcome]}" if outcome in notes else ""


def kept_stamp(pr, *, stamp, outcome):
    """Choose the close stamp a rewritten comment keeps.

    Parameters
    ----------
    pr : dict
        Pull request object, as read before this run changed it.
    stamp : str or None
        Close stamp from `read_stamp`.
    outcome : str or None
        Result of `restore` in this run.

    Returns
    -------
    str or None
        The old stamp while the pull request stays closed, even when it no
        longer matches: dropping it would hand the decision to the event
        fallback of `closed_by_policy`, which cannot tell another workflow's
        close from the policy's.
    """
    return stamp if pr["state"] == "closed" and outcome != REOPENED else None


def find_waiver(call, *, owner, repo, pr, reopened_by):
    """Find a maintainer who reopened a pull request the policy closed.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    pr : dict
        Pull request object the policy holds.
    reopened_by : str or None
        Login that reopened the pull request, for a ``reopened`` event.

    Returns
    -------
    str or None
        Login of the first reopener, other than the author and with write
        access, among ``reopened_by`` and then every ``reopened`` issue event
        that follows a close by `POLICY_ACTOR`, newest first.

    Notes
    -----
    The issue events make the waiver survive a cancelled run: a ``reopened``
    run that a later edit cancels in the concurrency group never stamps its
    sender, and the next run would otherwise close the pull request again.
    """
    author = pr["user"]["login"].lower()
    seen = {author, POLICY_ACTOR.lower()}

    def qualifies(login):
        if not login or login.lower() in seen:
            return False
        seen.add(login.lower())
        return has_write_access(call, owner=owner, repo=repo, login=login)

    if qualifies(reopened_by):
        return reopened_by
    number = pr["number"]
    events = paginate(call, f"/repos/{owner}/{repo}/issues/{number}/events", params={})
    reopeners = []
    closer = None
    for event in events:
        actor = (event.get("actor") or {}).get("login")
        if event.get("event") == "closed":
            closer = actor
        elif event.get("event") == "reopened" and closer == POLICY_ACTOR:
            reopeners.append(actor)
    return next((login for login in reversed(reopeners) if qualifies(login)), None)


def set_status(call, *, owner, repo, pr, state, description):
    """Publish the verdict as a commit status on the head commit.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    pr : dict
        Pull request object.
    state : {"success", "failure"}
        Verdict.
    description : str
        One-line reason, cut to the 140 characters GitHub accepts.
    """
    payload = {
        "state": state,
        "context": STATUS_CONTEXT,
        "description": description[:140],
    }
    if pr.get("html_url"):
        payload["target_url"] = pr["html_url"]
    call(
        "POST",
        f"/repos/{owner}/{repo}/statuses/{pr['head']['sha']}",
        payload=payload,
    )


def enforce(
    call,
    *,
    owner,
    repo,
    repo_url,
    pr,
    members,
    base_branch,
    reopened_by=None,
    recheck=True,
):
    """Apply the policy to one pull request.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    repo_url : str
        Repository web URL.
    pr : dict
        Pull request object, as the REST API returns it now.
    members : frozenset of str
        Lower-cased community member logins.
    base_branch : str
        Default branch; a pull request into any other is left alone.
    reopened_by : str or None
        Login that reopened the pull request, for a ``reopened`` event.
    recheck : bool
        Whether to read the issues again after closing or reopening the pull
        request, and start over on a fresh copy when an assignment changed
        in between. The second pass runs with ``recheck=False``.

    Returns
    -------
    dict
        ``{"skipped": reason}``, ``{"linked": numbers}`` or
        ``{"problems": reasons}``.

    Notes
    -----
    Bots, ``OWNER``/``MEMBER`` authors, anyone else with write access and
    community members skip the check, and a hold placed before they
    qualified is lifted. A merged pull request, or a closed one the policy
    is not holding, is left alone. A maintainer other than the author who
    reopens a pull request the policy closed waives the check for good; see
    `find_waiver`. Only the first `MAX_REFS` references are looked up,
    sidebar links first, so a description cannot spend the token's API
    budget.

    The recheck closes the window between reading an issue and changing the
    state: an assignment run that read the pull request before this run
    closed it saw nothing to do, and this run read the issue before the
    assignment, so without a second look neither would reopen it.
    """
    author = pr["user"]["login"]
    number = pr["number"]
    where = {"call": call, "owner": owner, "repo": repo}
    status = {**where, "pr": pr}
    association = pr["author_association"]
    if pr["base"]["ref"] != base_branch:
        return {"skipped": f"targets {pr['base']['ref']}"}
    if pr.get("merged"):
        return {"skipped": "merged"}
    if pr["state"] == "closed" and not has_label(pr):
        return {"skipped": "closed, not held by the policy"}
    if pr["user"].get("type") == "Bot":
        reason = f"{author} ({association})"
        set_status(**status, state="success", description=f"Skipped: {reason}")
        return {"skipped": reason}

    def start_over():
        return enforce(
            call,
            owner=owner,
            repo=repo,
            repo_url=repo_url,
            pr=call("GET", f"/repos/{owner}/{repo}/pulls/{number}"),
            members=members,
            base_branch=base_branch,
            reopened_by=reopened_by,
            recheck=False,
        )

    member = author.lower() in members
    maintainer = association in TRUSTED or (
        not member and has_write_access(**where, login=author)
    )
    if maintainer or member:
        mine = (
            policy_comments(**where, number=number) if member or has_label(pr) else []
        )
        stamp = read_stamp(mine)
        outcome = restore(**status, stamp=stamp)
        role = "a maintainer" if maintainer else "a polyxios community member"
        upsert_comment(
            **where,
            number=number,
            body=f":white_check_mark: @{author} is {role}.{restore_note(outcome)}",
            create=False,
            mine=mine,
            closed_at=kept_stamp(pr, stamp=stamp, outcome=outcome),
        )
        if not maintainer:
            reason = f"{author} (community member)"
        elif association in TRUSTED:
            reason = f"{author} ({association})"
        else:
            reason = f"{author} (write access)"
        set_status(**status, state="success", description=f"Skipped: {reason}")
        return {"skipped": reason}

    mine = policy_comments(**where, number=number)
    stamp = read_stamp(mine)
    waiver = read_waiver(mine)
    if waiver is None and (has_label(pr) or stamp is not None):
        waiver = find_waiver(**status, reopened_by=reopened_by)
    if waiver is not None:
        outcome = restore(**status, stamp=stamp)
        upsert_comment(
            **where,
            number=number,
            body=f":white_check_mark: @{waiver} reopened this pull request, which "
            f"waives the issue-first check.{restore_note(outcome)}",
            create=True,
            mine=mine,
            waived_by=waiver,
        )
        reason = f"{author} (waived by {waiver})"
        set_status(**status, state="success", description=f"Skipped: {reason}")
        return {"skipped": reason}

    sidebar = closing_refs(call, owner=owner, repo=repo, numbers=[number])
    refs = dict.fromkeys(sidebar[number], "issue")
    for ref, kind in extract_refs(pr.get("body"), owner=owner, repo=repo).items():
        refs.setdefault(ref, kind)
    refs.pop(number, None)

    linked = []
    problems = []
    issues = []
    for ref, kind in list(refs.items())[:MAX_REFS]:
        problem, is_issue = check_ref(
            call, owner=owner, repo=repo, number=ref, kind=kind, author=author
        )
        if is_issue:
            issues.append(ref)
        if problem is None:
            linked.append(ref)
        else:
            problems.append(problem)
    if not refs:
        problems.append("The description references no issue.")
    if len(refs) > MAX_REFS:
        problems.append(
            f"Only the first {MAX_REFS} references were checked; "
            f"{len(refs) - MAX_REFS} more were ignored."
        )
    ok = bool(linked)

    newcomer = association not in RETURNING
    sections = [welcome_text(author)] if newcomer else []

    if ok:
        outcome = restore(**status, stamp=stamp)
        if (
            outcome == REOPENED
            and recheck
            and not any_assigned(**where, numbers=linked, author=author)
        ):
            return start_over()
        verdict = status_text(author, linked=linked, problems=None, repo_url=repo_url)
        sections.append(f"{verdict}{restore_note(outcome)}")
        upsert_comment(
            **where,
            number=number,
            body="\n\n---\n\n".join(sections),
            create=newcomer or outcome == STUCK,
            mine=mine,
            closed_at=kept_stamp(pr, stamp=stamp, outcome=outcome),
        )
        issues = ", ".join(f"#{n}" for n in linked)
        set_status(
            **status,
            state="success",
            description=f"Linked to {issues}, assigned to @{author}",
        )
        return {"linked": linked}

    if not has_label(pr):
        ensure_label(call, owner=owner, repo=repo)
        call(
            "POST",
            f"/repos/{owner}/{repo}/issues/{number}/labels",
            payload={"labels": [LABEL]},
        )
    closed_at = None
    if pr["state"] == "open":
        closed = call(
            "PATCH",
            f"/repos/{owner}/{repo}/pulls/{number}",
            payload={"state": "closed"},
        )
        closed_at = (closed or {}).get("closed_at")
        if recheck and any_assigned(**where, numbers=issues, author=author):
            return start_over()
    elif stamp is not None:
        closed_at = stamp
    elif closed_by_policy(**status, stamp=None):
        closed_at = pr.get("closed_at")
    sections.append(
        status_text(author, linked=linked, problems=problems, repo_url=repo_url)
    )
    upsert_comment(
        **where,
        number=number,
        body="\n\n---\n\n".join(sections),
        create=True,
        mine=mine,
        closed_at=closed_at,
    )
    set_status(
        **status,
        state="failure",
        description=f"No linked issue is assigned to @{author}; see the comment",
    )
    return {"problems": problems}


def pulls_affected_by(call, *, owner, repo, login, issue):
    """Find a user's pull requests whose verdict hangs on an issue.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    owner, repo : str
        Repository.
    login : str
        User whose assignment changed.
    issue : int
        Issue number.

    Returns
    -------
    list of dict
        Open pull requests, and ones the policy closed, that reference the
        issue. The description in the issue listing is searched first, and
        the sidebar links of the rest are read in one batched query, so a
        full pull request is fetched only for a match.
    """
    path = f"/repos/{owner}/{repo}/issues"
    candidates = paginate(call, path, params={"creator": login, "state": "open"})
    candidates += paginate(
        call, path, params={"creator": login, "state": "closed", "labels": LABEL}
    )
    candidates = [item for item in candidates if item.get("pull_request") is not None]
    in_body = {
        item["number"]
        for item in candidates
        if issue in extract_refs(item.get("body"), owner=owner, repo=repo)
    }
    sidebar = closing_refs(
        call,
        owner=owner,
        repo=repo,
        numbers=[
            item["number"] for item in candidates if item["number"] not in in_body
        ],
    )
    return [
        call("GET", f"/repos/{owner}/{repo}/pulls/{item['number']}")
        for item in candidates
        if item["number"] in in_body or issue in sidebar.get(item["number"], [])
    ]


def run(call, *, event_name, event, owner, repo, server_url, members):
    """Apply the policy for one workflow event.

    Parameters
    ----------
    call : callable
        Caller built by `make_call`.
    event_name : str
        ``pull_request_target`` or ``issues``.
    event : dict
        Event payload, with the ``repository`` it came from.
    owner, repo : str
        Repository.
    server_url : str
        GitHub web root, such as ``https://github.com``.
    members : frozenset of str
        Lower-cased community member logins.

    Returns
    -------
    dict of int to dict
        Pull request number to its `enforce` result, or to ``{"error":
        message}`` when the API failed on it. One failure does not stop the
        pull requests after it.
    """
    if event_name == "issues":
        issue = event["issue"]
        assignee = event.get("assignee")
        if issue.get("pull_request") is not None or not assignee:
            return {}
        pulls = pulls_affected_by(
            call, owner=owner, repo=repo, login=assignee["login"], issue=issue["number"]
        )
        print(f"Issue #{issue['number']}: re-checking {len(pulls)} pull request(s).")
        reopened_by = None
    else:
        number = event["pull_request"]["number"]
        pulls = [call("GET", f"/repos/{owner}/{repo}/pulls/{number}")]
        reopened_by = (
            (event.get("sender") or {}).get("login")
            if event.get("action") == "reopened"
            else None
        )

    repo_url = f"{server_url}/{owner}/{repo}"
    results = {}
    for pr in pulls:
        try:
            result = enforce(
                call,
                owner=owner,
                repo=repo,
                repo_url=repo_url,
                pr=pr,
                members=members,
                base_branch=event["repository"]["default_branch"],
                reopened_by=reopened_by,
            )
        except ApiError as err:
            result = {"error": str(err)}
        if "skipped" in result:
            print(f"#{pr['number']}: skipping {result['skipped']}.")
        elif "linked" in result:
            print(f"#{pr['number']}: linked {result['linked']}.")
        elif "problems" in result:
            print(f"#{pr['number']}: {' '.join(result['problems'])}")
        results[pr["number"]] = result
    return results


def main():
    """Run the policy from the GitHub Actions environment.

    Returns
    -------
    int
        Process exit status.
    """
    env = os.environ
    owner, repo = env["GITHUB_REPOSITORY"].split("/")
    event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    call = make_call(
        token=env["GITHUB_TOKEN"],
        api_url=env.get("GITHUB_API_URL", "https://api.github.com"),
    )
    results = run(
        call,
        event_name=env["GITHUB_EVENT_NAME"],
        event=event,
        owner=owner,
        repo=repo,
        server_url=env.get("GITHUB_SERVER_URL", "https://github.com"),
        members=load_members(MEMBERS_FILE),
    )
    errors = [
        f"#{number}: {result['error']}"
        for number, result in results.items()
        if "error" in result
    ]
    for error in errors:
        print(f"::error::{error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

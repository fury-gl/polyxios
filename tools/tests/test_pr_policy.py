import re
import time
from urllib.error import URLError
from urllib.parse import unquote

import pr_policy
from pr_policy import LABEL, MARKER, POLICY_ACTOR, STATUS_CONTEXT, ApiError
import pytest

OWNER, REPO = "fury-gl", "polyxios"
SERVER = "https://github.com"
API = "https://api.github.com"
NO_MEMBERS = frozenset()
CLOSED_AT = "2026-10-19T12:00:00Z"
BOT = {"login": POLICY_ACTOR, "type": "Bot"}
BASE_PERMISSION = {"maintain": "write", "triage": "read"}
REPOSITORY = {"default_branch": "master"}


def fake_github(
    *,
    issues=None,
    pulls=None,
    comments=None,
    closing=None,
    events=None,
    fail=None,
    label=True,
    roles=None,
):
    """Build an in-memory GitHub behind the ``call`` interface.

    Parameters
    ----------
    issues : dict, optional
        Issue number to ``{"assignees": [...], "pull_request": bool,
        "repo": "owner/name", "state": "open" or "closed"}``.
    pulls : dict, optional
        Pull request number to pull request object, served by GET.
    comments : dict, optional
        Pull request number to its comment list.
    closing : dict, optional
        Pull request number to sidebar-linked issue numbers.
    events : dict, optional
        Pull request number to its issue events. A number missing here
        reads as last closed by the policy.
    fail : dict, optional
        ``(method, path)`` to the HTTP status that write raises.
    label : bool, optional
        Whether the ``needs-issue`` label already exists.
    roles : dict, optional
        Login to repository role name, or to ``(custom role name, base
        permission)``; anyone missing reads as ``read``.

    Returns
    -------
    call : callable
        Fake caller; ``call.pulls`` is the pull request registry and
        ``call.graphql_queries`` every GraphQL document sent.
    writes : list of tuple
        Every non-GET request that succeeded, as ``(method, path, payload)``.
    """
    issues = issues or {}
    pulls = {} if pulls is None else pulls
    comments = comments or {}
    closing = closing or {}
    events = events or {}
    fail = fail or {}
    roles = roles or {}
    writes = []
    graphql_queries = []
    slug = f"{OWNER}/{REPO}"
    base = f"/repos/{OWNER}/{REPO}"

    def call(method, path, *, params=None, payload=None):
        params = params or {}
        if method == "POST" and path == "/graphql":
            graphql_queries.append(payload["query"])
            fields = re.findall(
                r"(pr\d+): pullRequest\(number: (\d+)\)", payload["query"]
            )
            repository = {
                alias: {
                    "closingIssuesReferences": {
                        "nodes": [
                            {"number": n, "repository": {"nameWithOwner": slug}}
                            for n in closing.get(int(number), [])
                        ]
                    }
                }
                for alias, number in fields
            }
            return {"data": {"repository": repository}}
        if method != "GET":
            if (method, unquote(path)) in fail:
                raise ApiError(status=fail[method, unquote(path)], message="refused")
            writes.append((method, unquote(path), payload))
            if "/pulls/" in path and payload.get("state") == "closed":
                return {"state": "closed", "closed_at": CLOSED_AT}
            return {}
        parts = path.removeprefix(base).strip("/").split("/")
        if parts == ["issues"]:
            if params.get("page", 1) > 1:
                return []
            return [
                {"number": pr["number"], "pull_request": {}, "body": pr["body"]}
                for pr in pulls.values()
                if pr["user"]["login"] == params["creator"]
                and pr["state"] == params["state"]
                and ("labels" not in params or pr_policy.has_label(pr))
            ]
        if parts[0] == "issues" and len(parts) == 2:
            number = int(parts[1])
            if number not in issues:
                raise ApiError(status=404, message="Not Found")
            item = issues[number]
            found = {
                "number": number,
                "repository_url": f"{API}/repos/{item.get('repo', f'{OWNER}/{REPO}')}",
                "assignees": [{"login": a} for a in item.get("assignees", [])],
                "state": item.get("state", "open"),
            }
            if item.get("pull_request"):
                found["pull_request"] = {}
            return found
        if parts[0] == "issues" and parts[2] == "comments":
            if params.get("page", 1) > 1:
                return []
            return comments.get(int(parts[1]), [])
        if parts[0] == "issues" and parts[2] == "events":
            if params.get("page", 1) > 1:
                return []
            return events.get(
                int(parts[1]), [{"event": "closed", "actor": {"login": POLICY_ACTOR}}]
            )
        if parts[0] == "collaborators":
            role = roles.get(unquote(parts[1]), "read")
            role, permission = role if isinstance(role, tuple) else (role, role)
            return {
                "role_name": role,
                "permission": BASE_PERMISSION.get(permission, permission),
            }
        if parts[0] == "pulls":
            return pulls[int(parts[1])]
        if parts[0] == "labels":
            if not label:
                raise ApiError(status=404, message="Not Found")
            return {"name": LABEL}
        raise AssertionError(f"unexpected {method} {path}")

    call.pulls = pulls
    call.graphql_queries = graphql_queries
    return call, writes


def make_pr(number=100, *, login="alice", association="FIRST_TIME_CONTRIBUTOR", **kw):
    return {
        "number": number,
        "user": {"login": login, "type": "User"},
        "author_association": association,
        "labels": [],
        "state": "open",
        "merged": False,
        "body": "",
        "head": {"sha": f"sha{number}"},
        "base": {"ref": "master"},
        **kw,
    }


def run_pr(call, pr, *, members=NO_MEMBERS):
    """Run a pull request event; the fake serves ``pr`` as the API's copy."""
    call.pulls.setdefault(pr["number"], pr)
    return pr_policy.run(
        call,
        event_name="pull_request_target",
        event={"pull_request": {"number": pr["number"]}, "repository": REPOSITORY},
        owner=OWNER,
        repo=REPO,
        server_url=SERVER,
        members=members,
    )


def run_assignment(call, *, login, issue=5):
    return pr_policy.run(
        call,
        event_name="issues",
        event={
            "action": "assigned",
            "issue": {"number": issue},
            "assignee": {"login": login},
            "repository": REPOSITORY,
        },
        owner=OWNER,
        repo=REPO,
        server_url=SERVER,
        members=NO_MEMBERS,
    )


def problems(results, number=100):
    return results[number].get("problems", [])


def comment_bodies(writes):
    return [
        p["body"]
        for m, path, p in writes
        if m in ("POST", "PATCH")
        and (path.endswith("comments") or "/comments/" in path)
    ]


def state_changes(writes):
    return [
        (int(path.rsplit("/", 1)[1]), p["state"])
        for m, path, p in writes
        if m == "PATCH" and "/pulls/" in path
    ]


def statuses(writes):
    return [
        (path.rsplit("/", 1)[1], p["state"])
        for m, path, p in writes
        if m == "POST" and "/statuses/" in path
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Closes #12", {12: "any"}),
        ("see fury-gl/polyxios#7, not other/repo#9", {7: "any"}),
        ("https://github.com/fury-gl/polyxios/discussions/33", {33: "discussion"}),
        ("https://github.com/Fury-GL/polyxios/issues/4 and #4", {4: "issue"}),
        ("https://github.com/x/y/issues/4", {}),
        ("<!-- Closes #99 -->", {}),
        ("`#55` and\n```\n#56\n```", {}),
        ("~~~\nCloses #57\n~~~", {}),
        ("a#66 b/#67", {}),
        ("text\n\n    Closes #12\n\nmore", {}),
        ("\tCloses #13", {}),
        ("- item\n    Closes #14", {14: "any"}),
        ("- fix\n\n    Closes #14", {14: "any"}),
        ("1. fix\n\n\tCloses #14", {14: "any"}),
        ("- item\n\ntext\n\n    Closes #15", {}),
        ("text\n\n    a\n\n    Closes #16\nafter", {}),
        ("Closes #17\n```\nsee #18", {17: "any"}),
        ("Closes #17\n~~~python\n#18", {17: "any"}),
        ("use ``` to fence, Closes #19", {19: "any"}),
        ("Closes #17\n<!-- draft\nCloses #18", {17: "any"}),
        ("```html\n<!-- x\n```\nCloses #20", {20: "any"}),
        ("Type `<!--` to hide text. Closes #21", {21: "any"}),
        ("~~~\n```\n~~~\nCloses #22\n```", {22: "any"}),
        ("<!--\n```\n-->\nCloses #23", {23: "any"}),
        ("``code #3`` and #4", {4: "any"}),
        ("`a `` b` #9", {9: "any"}),
        ("`` a ` #10 `` #11", {11: "any"}),
        ("````\n```py\nCloses #12\n```\n````\nCloses #13", {13: "any"}),
        ("```\ncode ``` #7\n```\n#8", {8: "any"}),
        ("```\n#3\n  ```x\n#4", {}),
        ("```\n```\n#5", {5: "any"}),
        ("- step\n  ```\n  #6\n  ```\n- Closes #7", {7: "any"}),
        ("> ```\n> #8\n> ```\n#9", {9: "any"}),
        ("```py `x`\n#10", {10: "any"}),
        ("fixes GH-12", {12: "any"}),
        ("fixes gh-15 and Gh-16", {15: "any", 16: "any"}),
        ("xGH-12 a-GH-13 `GH-14`", {}),
        ("https://www.github.com/fury-gl/polyxios/issues/12", {12: "issue"}),
        (None, {}),
    ],
)
def test_extract_refs(body, expected):
    assert pr_policy.extract_refs(body, owner=OWNER, repo=REPO) == expected


def test_load_members(tmp_path):
    path = tmp_path / "members.txt"
    path.write_text("# header\n@Alice  # joined\n\nbob\n", encoding="utf-8")
    assert pr_policy.load_members(path) == {"alice", "bob"}
    assert pr_policy.load_members(tmp_path / "missing.txt") == frozenset()


def test_members_file_ships_empty():
    assert pr_policy.load_members(pr_policy.MEMBERS_FILE) == frozenset()


def test_no_reference_closes_and_welcomes():
    call, writes = fake_github(label=False)
    results = run_pr(call, make_pr())
    assert problems(results)
    (body,) = comment_bodies(writes)
    assert body.startswith(MARKER)
    assert "Welcome to polyxios, @alice" in body
    assert "references no issue" in body
    assert any(
        path.endswith("/labels") and p.get("name") == LABEL for _, path, p in writes
    )
    assert any(p == {"labels": [LABEL]} for _, _, p in writes)
    assert state_changes(writes) == [(100, "closed")]
    assert statuses(writes) == [("sha100", "failure")]


def test_status_carries_the_policy_context():
    call, writes = fake_github()
    run_pr(call, make_pr(html_url="https://github.com/x/y/pull/100"))
    ((_, _, payload),) = [w for w in writes if "/statuses/" in w[1]]
    assert payload["context"] == STATUS_CONTEXT
    assert payload["target_url"] == "https://github.com/x/y/pull/100"
    assert len(payload["description"]) <= 140


def test_verdict_comes_from_the_api_not_the_payload():
    call, writes = fake_github(issues={5: {"assignees": ["alice"]}})
    call.pulls[100] = make_pr(body="Closes #5")
    results = pr_policy.run(
        call,
        event_name="pull_request_target",
        event={
            "pull_request": make_pr(body="stale description"),
            "repository": REPOSITORY,
        },
        owner=OWNER,
        repo=REPO,
        server_url=SERVER,
        members=NO_MEMBERS,
    )
    assert results[100] == {"linked": [5]}
    assert statuses(writes) == [("sha100", "success")]


@pytest.mark.parametrize(
    ("issue", "message"),
    [
        ({"assignees": []}, "not assigned to you"),
        ({"assignees": ["bob"]}, "not assigned to you"),
        ({"assignees": ["alice"], "pull_request": True}, "is a pull request"),
        ({"assignees": ["alice"], "repo": "fury-gl/fury"}, "transferred"),
        ({"assignees": ["alice"], "state": "closed"}, "is closed"),
    ],
)
def test_unqualified_issue_closes(issue, message):
    call, writes = fake_github(issues={5: issue})
    results = run_pr(call, make_pr(body="Closes #5"))
    assert message in problems(results)[0]
    assert state_changes(writes) == [(100, "closed")]
    assert statuses(writes) == [("sha100", "failure")]


def test_discussion_url_does_not_count():
    call, _ = fake_github()
    results = run_pr(
        call, make_pr(body="https://github.com/fury-gl/polyxios/discussions/8")
    )
    (problem,) = problems(results)
    assert "#8 is a discussion." in problem


def test_bare_number_not_found_may_not_exist():
    call, _ = fake_github()
    (problem,) = problems(run_pr(call, make_pr(body="Closes #8")))
    assert "is a discussion or does not exist" in problem


def test_issue_url_that_does_not_exist():
    call, _ = fake_github()
    results = run_pr(call, make_pr(body="https://github.com/fury-gl/polyxios/issues/8"))
    (problem,) = problems(results)
    assert "does not exist" in problem


def test_self_reference_does_not_count():
    call, _ = fake_github(issues={100: {"assignees": ["alice"]}})
    (problem,) = problems(run_pr(call, make_pr(body="#100")))
    assert "references no issue" in problem


def test_assigned_issue_passes_case_insensitively():
    call, writes = fake_github(issues={5: {"assignees": ["bob", "Alice"]}})
    assert run_pr(call, make_pr(body="Closes #5")) == {100: {"linked": [5]}}
    (body,) = comment_bodies(writes)
    assert "Linked to issue #5, assigned to @alice" in body
    assert state_changes(writes) == []
    assert statuses(writes) == [("sha100", "success")]


def test_one_good_reference_is_enough():
    call, _ = fake_github(issues={5: {"assignees": []}, 6: {"assignees": ["alice"]}})
    assert run_pr(call, make_pr(body="#5 #6")) == {100: {"linked": [6]}}


def test_sidebar_link_counts():
    call, _ = fake_github(issues={9: {"assignees": ["alice"]}}, closing={100: [9]})
    assert run_pr(call, make_pr(body="no reference here")) == {100: {"linked": [9]}}


def test_returning_contributor_passing_gets_no_comment():
    call, writes = fake_github(issues={5: {"assignees": ["alice"]}})
    assert "linked" in run_pr(call, make_pr(body="#5", association="CONTRIBUTOR"))[100]
    assert comment_bodies(writes) == []
    assert statuses(writes) == [("sha100", "success")]


def test_returning_contributor_failing_gets_no_welcome():
    call, writes = fake_github()
    assert problems(run_pr(call, make_pr(association="CONTRIBUTOR")))
    (body,) = comment_bodies(writes)
    assert "Welcome" not in body
    assert "category=community-membership" in body


@pytest.mark.parametrize("association", ["OWNER", "MEMBER"])
def test_maintainers_skip(association):
    call, writes = fake_github()
    assert "skipped" in run_pr(call, make_pr(association=association))[100]
    assert [w for w in writes if "/statuses/" not in w[1]] == []
    assert statuses(writes) == [("sha100", "success")]


@pytest.mark.parametrize("role", ["read", "triage"])
def test_collaborator_without_write_access_is_checked(role):
    call, writes = fake_github(roles={"alice": role})
    results = run_pr(call, make_pr(association="COLLABORATOR"))
    assert problems(results)
    assert "Welcome" not in comment_bodies(writes)[0]
    assert state_changes(writes) == [(100, "closed")]


def test_collaborator_with_write_access_skips():
    call, _ = fake_github(roles={"alice": "write"})
    results = run_pr(call, make_pr(association="COLLABORATOR"))
    assert results[100] == {"skipped": "alice (write access)"}


def test_bots_skip():
    call, writes = fake_github()
    pr = make_pr(user={"login": "dependabot[bot]", "type": "Bot"}, association="NONE")
    assert "skipped" in run_pr(call, pr)[100]
    assert statuses(writes) == [("sha100", "success")]


def test_community_member_held_pr_reopens():
    call, writes = fake_github()
    pr = make_pr(state="closed", labels=[{"name": LABEL}])
    assert "skipped" in run_pr(call, pr, members=frozenset({"alice"}))[100]
    assert (
        "DELETE",
        f"/repos/{OWNER}/{REPO}/issues/100/labels/{LABEL}",
        None,
    ) in writes
    assert state_changes(writes) == [(100, "open")]
    assert comment_bodies(writes) == []


def test_sticky_comment_is_updated_not_duplicated():
    old = {"id": 1, "user": BOT, "body": f"{MARKER}\nold"}
    call, writes = fake_github(comments={100: [old]})
    run_pr(call, make_pr())
    patches = [path for m, path, _ in writes if m == "PATCH" and "/comments/" in path]
    posts = [
        path for m, path, _ in writes if m == "POST" and path.endswith("/comments")
    ]
    assert patches == [f"/repos/{OWNER}/{REPO}/issues/comments/1"]
    assert posts == []


def test_duplicate_sticky_comments_collapse_to_the_oldest():
    dupes = [
        {"id": 7, "user": BOT, "body": f"{MARKER}\nnewer"},
        {"id": 3, "user": BOT, "body": f"{MARKER}\nolder"},
        {"id": 9, "user": {"type": "User"}, "body": f"{MARKER}\nquoted"},
    ]
    delete = ("DELETE", f"/repos/{OWNER}/{REPO}/issues/comments/7")
    call, writes = fake_github(comments={100: dupes}, fail={delete: 404})
    run_pr(call, make_pr())
    patches = [path for m, path, _ in writes if m == "PATCH" and "/comments/" in path]
    assert patches == [f"/repos/{OWNER}/{REPO}/issues/comments/3"]


def test_unchanged_sticky_comment_is_left_alone():
    call, writes = fake_github(issues={5: {"assignees": ["alice"]}})
    run_pr(call, make_pr(body="#5"))
    (body,) = comment_bodies(writes)
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}},
        comments={100: [{"id": 1, "user": BOT, "body": body}]},
    )
    run_pr(call, make_pr(body="#5"))
    assert comment_bodies(writes) == []


def test_merged_pr_is_never_reopened():
    call, writes = fake_github(issues={5: {"assignees": ["alice"]}})
    pr = make_pr(body="#5", state="closed", merged=True, labels=[{"name": LABEL}])
    run_pr(call, pr)
    assert state_changes(writes) == []


def test_pr_a_person_closed_is_never_reopened():
    closed = [
        {"event": "closed", "actor": {"login": POLICY_ACTOR}},
        {"event": "reopened", "actor": {"login": "alice"}},
        {"event": "closed", "actor": {"login": "maintainer"}},
    ]
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}}, events={100: closed}
    )
    pr = make_pr(body="#5", state="closed", labels=[{"name": LABEL}])
    run_pr(call, pr)
    assert state_changes(writes) == []
    assert ("DELETE", f"/repos/{OWNER}/{REPO}/issues/100/labels/{LABEL}", None) in (
        writes
    )
    assert statuses(writes) == [("sha100", "success")]


def test_label_already_removed_is_not_an_error():
    path = f"/repos/{OWNER}/{REPO}/issues/100/labels/{LABEL}"
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}}, fail={("DELETE", path): 404}
    )
    pr = make_pr(body="#5", labels=[{"name": LABEL}])
    assert run_pr(call, pr) == {100: {"linked": [5]}}
    assert statuses(writes) == [("sha100", "success")]


def test_label_created_by_an_overlapping_run_is_not_an_error():
    path = f"/repos/{OWNER}/{REPO}/labels"
    call, _ = fake_github(label=False, fail={("POST", path): 422})
    assert problems(run_pr(call, make_pr()))


def test_assignment_reopens_held_pr_and_clears_its_status():
    pulls = {
        100: make_pr(100, body="Closes #5", state="closed", labels=[{"name": LABEL}]),
        101: make_pr(101, body="Closes #6"),
        102: make_pr(102, login="bob", body="Closes #5"),
    }
    call, writes = fake_github(issues={5: {"assignees": ["alice"]}}, pulls=pulls)
    assert run_assignment(call, login="alice") == {100: {"linked": [5]}}
    assert state_changes(writes) == [(100, "open")]
    assert statuses(writes) == [("sha100", "success")]
    assert all("/101" not in path and "/102" not in path for _, path, _ in writes)


def test_unassignment_closes_open_pr():
    pulls = {100: make_pr(100, body="Closes #5")}
    call, writes = fake_github(issues={5: {"assignees": []}}, pulls=pulls)
    assert problems(run_assignment(call, login="alice"))
    assert state_changes(writes) == [(100, "closed")]
    assert statuses(writes) == [("sha100", "failure")]


def test_unassignment_keeps_pr_linking_another_assigned_issue():
    pulls = {100: make_pr(100, body="Closes #5, part of #6")}
    call, writes = fake_github(
        issues={5: {"assignees": []}, 6: {"assignees": ["alice"]}}, pulls=pulls
    )
    assert run_assignment(call, login="alice") == {100: {"linked": [6]}}
    assert state_changes(writes) == []


def test_failed_reopen_does_not_stop_the_other_pulls():
    held = {"state": "closed", "labels": [{"name": LABEL}]}
    pulls = {
        100: make_pr(100, body="Closes #5", **held),
        101: make_pr(101, body="Closes #5", **held),
    }
    reopen = ("PATCH", f"/repos/{OWNER}/{REPO}/pulls/100")
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}}, pulls=pulls, fail={reopen: 500}
    )
    results = run_assignment(call, login="alice")
    assert "error" in results[100]
    assert results[101] == {"linked": [5]}
    assert state_changes(writes) == [(101, "open")]


def test_refused_reopen_drops_the_hold_and_says_why():
    pr = make_pr(
        body="Closes #5",
        association="CONTRIBUTOR",
        state="closed",
        labels=[{"name": LABEL}],
    )
    reopen = ("PATCH", f"/repos/{OWNER}/{REPO}/pulls/100")
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}}, pulls={100: pr}, fail={reopen: 422}
    )
    assert run_assignment(call, login="alice") == {100: {"linked": [5]}}
    assert ("DELETE", f"/repos/{OWNER}/{REPO}/issues/100/labels/{LABEL}", None) in (
        writes
    )
    (body,) = comment_bodies(writes)
    assert pr_policy.STUCK_NOTE in body
    assert pr_policy.REOPEN_NOTE not in body
    assert statuses(writes) == [("sha100", "success")]
    call.pulls[100] = {**pr, "labels": []}
    assert run_pr(call, call.pulls[100])[100] == {
        "skipped": "closed, not held by the policy"
    }


def test_main_fails_only_on_api_errors(monkeypatch, tmp_path):
    event = tmp_path / "event.json"
    event.write_text(
        '{"pull_request": {"number": 100}, "repository": {"default_branch": "master"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", f"{OWNER}/{REPO}")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_target")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    call, _ = fake_github(pulls={100: make_pr()})
    monkeypatch.setattr(pr_policy, "make_call", lambda **kw: call)
    assert pr_policy.main() == 0
    held = ("PATCH", f"/repos/{OWNER}/{REPO}/pulls/100")
    broken, _ = fake_github(pulls={100: make_pr()}, fail={held: 500})
    monkeypatch.setattr(pr_policy, "make_call", lambda **kw: broken)
    assert pr_policy.main() == 1


def test_assignment_event_on_pull_request_is_ignored():
    call, writes = fake_github()
    results = pr_policy.run(
        call,
        event_name="issues",
        event={"issue": {"number": 5, "pull_request": {}}, "assignee": {"login": "a"}},
        owner=OWNER,
        repo=REPO,
        server_url=SERVER,
        members=NO_MEMBERS,
    )
    assert results == {}
    assert writes == []


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def test_make_call_sends_json(monkeypatch):
    sent = []

    def urlopen(request, timeout):
        sent.append(request)
        return _Response(b'{"ok": true}')

    monkeypatch.setattr(pr_policy, "urlopen", urlopen)
    call = pr_policy.make_call(token="t", api_url=API + "/")
    assert call("POST", "/x", params={"a": 1}, payload={"k": "v"}) == {"ok": True}
    (request,) = sent
    assert request.full_url == f"{API}/x?a=1"
    assert request.get_header("Content-type") == "application/json"
    assert request.data == b'{"k": "v"}'


@pytest.mark.parametrize("error", [URLError("down"), TimeoutError("slow")])
def test_make_call_wraps_network_errors(monkeypatch, error):
    def urlopen(request, timeout):
        raise error

    monkeypatch.setattr(pr_policy, "urlopen", urlopen)
    call = pr_policy.make_call(token="t", api_url=API)
    with pytest.raises(ApiError) as info:
        call("GET", "/x")
    assert info.value.status == 0


def test_graphql_errors_raise():
    def call(method, path, *, params=None, payload=None):
        return {"errors": [{"message": "boom"}]}

    with pytest.raises(ApiError, match="boom"):
        pr_policy.graphql(call, "query", variables={})


def test_paginate_follows_full_pages(monkeypatch):
    monkeypatch.setattr(pr_policy, "PER_PAGE", 2)
    pages = {1: [1, 2], 2: [3, 4], 3: [5]}

    def call(method, path, *, params=None, payload=None):
        return pages[params["page"]]

    assert pr_policy.paginate(call, "/x", params={}) == [1, 2, 3, 4, 5]


def stamped(closed_at, text="old"):
    body = f"{MARKER}\n<!-- polyxios-pr-policy closed_at={closed_at} -->\n{text}"
    return {"id": 1, "user": BOT, "body": body}


def test_close_stamps_the_sticky_comment():
    call, writes = fake_github()
    run_pr(call, make_pr())
    (body,) = comment_bodies(writes)
    assert f"closed_at={CLOSED_AT}" in body


def test_still_held_pr_keeps_its_stamp():
    call, writes = fake_github(comments={100: [stamped(CLOSED_AT)]})
    pr = make_pr(state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}])
    assert problems(run_pr(call, pr))
    (body,) = comment_bodies(writes)
    assert f"closed_at={CLOSED_AT}" in body


def test_stamped_close_is_reopened():
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}}, comments={100: [stamped(CLOSED_AT)]}
    )
    pr = make_pr(
        body="#5", state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}]
    )
    run_pr(call, pr)
    assert state_changes(writes) == [(100, "open")]


def test_close_by_another_workflow_is_not_undone():
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}},
        comments={100: [stamped("2026-01-01T00:00:00Z")]},
    )
    pr = make_pr(
        body="#5", state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}]
    )
    run_pr(call, pr)
    assert state_changes(writes) == []
    assert ("DELETE", f"/repos/{OWNER}/{REPO}/issues/100/labels/{LABEL}", None) in (
        writes
    )


@pytest.mark.parametrize(
    ("association", "members"),
    [("CONTRIBUTOR", NO_MEMBERS), ("CONTRIBUTOR", frozenset({"alice"}))],
)
def test_reopen_by_the_policy_warns_that_ci_did_not_run(association, members):
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}}, comments={100: [stamped(CLOSED_AT)]}
    )
    pr = make_pr(
        body="#5",
        association=association,
        state="closed",
        closed_at=CLOSED_AT,
        labels=[{"name": LABEL}],
    )
    run_pr(call, pr, members=members)
    assert state_changes(writes) == [(100, "open")]
    (body,) = comment_bodies(writes)
    assert "GitHub starts no CI" in body
    assert "closed_at=" not in body


def test_pass_without_reopen_has_no_ci_warning():
    call, writes = fake_github(issues={5: {"assignees": ["alice"]}})
    run_pr(call, make_pr(body="#5"))
    (body,) = comment_bodies(writes)
    assert "GitHub starts no CI" not in body


@pytest.mark.parametrize("role", ["admin", "maintain", "write"])
def test_private_org_member_with_write_access_skips(role):
    call, writes = fake_github(roles={"alice": role})
    results = run_pr(call, make_pr(association="CONTRIBUTOR"))
    assert results[100] == {"skipped": "alice (write access)"}
    assert state_changes(writes) == []
    assert statuses(writes) == [("sha100", "success")]


@pytest.mark.parametrize("role", ["read", "triage"])
def test_read_access_does_not_skip(role):
    call, _ = fake_github(roles={"alice": role})
    assert problems(run_pr(call, make_pr(association="CONTRIBUTOR")))


def test_held_pr_of_a_new_maintainer_reopens():
    call, writes = fake_github(roles={"alice": "write"})
    pr = make_pr(association="COLLABORATOR", state="closed", labels=[{"name": LABEL}])
    assert "skipped" in run_pr(call, pr)[100]
    assert state_changes(writes) == [(100, "open")]
    assert ("DELETE", f"/repos/{OWNER}/{REPO}/issues/100/labels/{LABEL}", None) in (
        writes
    )


def test_indented_code_with_crlf_line_endings_does_not_count():
    body = "text\r\n\r\n    Closes #12\r\n"
    assert pr_policy.extract_refs(body, owner=OWNER, repo=REPO) == {}


def test_foreign_close_stamp_survives_a_failing_edit():
    old = "2026-01-01T00:00:00Z"
    call, writes = fake_github(comments={100: [stamped(old)]})
    pr = make_pr(state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}])
    assert problems(run_pr(call, pr))
    (body,) = comment_bodies(writes)
    assert f"closed_at={old}" in body


def test_foreign_close_stamp_survives_a_passing_edit():
    old = "2026-01-01T00:00:00Z"
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}}, comments={100: [stamped(old)]}
    )
    pr = make_pr(
        body="#5", state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}]
    )
    run_pr(call, pr)
    assert state_changes(writes) == []
    (body,) = comment_bodies(writes)
    assert f"closed_at={old}" in body


def test_references_checked_are_capped():
    seen = []
    call, _ = fake_github()
    inner = call

    def counting(method, path, *, params=None, payload=None):
        if method == "GET" and "/issues/" in path and path.count("/") == 5:
            seen.append(path)
        return inner(method, path, params=params, payload=payload)

    counting.pulls = inner.pulls
    body = " ".join(f"#{n}" for n in range(1, 51))
    (*_, last) = problems(run_pr(counting, make_pr(body=body)))
    assert len(seen) == pr_policy.MAX_REFS
    assert "40 more" in last


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("see #3 and #4. Closes #7", [7, 3, 4]),
        ("#1 fixes: #2, resolved GH-3", [2, 3, 1]),
        ("FIXES https://github.com/fury-gl/polyxios/issues/8 #1", [8, 1]),
        ("unclosed #1 prefixes #2", [1, 2]),
        ("closes other/repo#5 #5", [5]),
    ],
)
def test_closing_keyword_references_come_first(body, expected):
    assert list(pr_policy.extract_refs(body, owner=OWNER, repo=REPO)) == expected


def test_closing_keyword_reference_beats_the_cap():
    call, _ = fake_github(issues={60: {"assignees": ["alice"]}})
    mentions = " ".join(f"#{n}" for n in range(1, 21))
    pr = make_pr(body=f"Follows {mentions}.\n\nCloses #60")
    assert run_pr(call, pr) == {100: {"linked": [60]}}


def run_reopen(call, pr, *, sender):
    call.pulls.setdefault(pr["number"], pr)
    return pr_policy.run(
        call,
        event_name="pull_request_target",
        event={
            "action": "reopened",
            "pull_request": {"number": pr["number"]},
            "sender": {"login": sender},
            "repository": REPOSITORY,
        },
        owner=OWNER,
        repo=REPO,
        server_url=SERVER,
        members=NO_MEMBERS,
    )


@pytest.mark.parametrize(
    "pr",
    [
        make_pr(state="closed", merged=True, closed_at=CLOSED_AT),
        make_pr(state="closed", merged=True, labels=[{"name": LABEL}]),
        make_pr(state="closed", closed_at=CLOSED_AT),
    ],
)
def test_pr_not_held_and_not_open_is_left_alone(pr):
    call, writes = fake_github()
    assert "skipped" in run_pr(call, pr)[100]
    assert writes == []


def test_maintainer_reopen_waives_the_check():
    call, writes = fake_github(roles={"maint": "maintain"})
    pr = make_pr(labels=[{"name": LABEL}])
    results = run_reopen(call, pr, sender="maint")
    assert results[100] == {"skipped": "alice (waived by maint)"}
    assert state_changes(writes) == []
    assert ("DELETE", f"/repos/{OWNER}/{REPO}/issues/100/labels/{LABEL}", None) in (
        writes
    )
    (body,) = comment_bodies(writes)
    assert "waived_by=maint" in body
    assert statuses(writes) == [("sha100", "success")]


def test_maintainer_reopen_after_removing_the_label_still_waives():
    call, writes = fake_github(
        roles={"maint": "admin"}, comments={100: [stamped(CLOSED_AT)]}
    )
    results = run_reopen(call, make_pr(), sender="maint")
    assert results[100] == {"skipped": "alice (waived by maint)"}
    assert state_changes(writes) == []
    (body,) = comment_bodies(writes)
    assert "waived_by=maint" in body
    assert "closed_at=" not in body


def test_waiver_survives_later_edits_and_unassignment():
    waived = {
        "id": 1,
        "user": BOT,
        "body": f"{MARKER}\n<!-- polyxios-pr-policy waived_by=maint -->\nok",
    }
    pulls = {100: make_pr(body="Closes #5")}
    call, writes = fake_github(
        issues={5: {"assignees": []}}, pulls=pulls, comments={100: [waived]}
    )
    assert run_assignment(call, login="alice")[100] == {
        "skipped": "alice (waived by maint)"
    }
    assert run_pr(call, pulls[100])[100]["skipped"] == "alice (waived by maint)"
    assert state_changes(writes) == []
    assert all("waived_by=maint" in body for body in comment_bodies(writes))
    assert statuses(writes) == [("sha100", "success")] * 2


@pytest.mark.parametrize(
    ("sender", "labels"),
    [
        ("alice", [{"name": LABEL}]),
        ("reader", [{"name": LABEL}]),
        ("maint", []),
    ],
)
def test_reopen_that_does_not_waive(sender, labels):
    call, writes = fake_github(roles={"maint": "write", "alice": "read"})
    assert problems(run_reopen(call, make_pr(labels=labels), sender=sender))
    assert state_changes(writes) == [(100, "closed")]
    assert all("waived_by" not in body for body in comment_bodies(writes))


def test_another_bot_cannot_forge_a_close_stamp():
    forged = {
        "id": 1,
        "user": {"login": "quote-bot[bot]", "type": "Bot"},
        "body": f"{MARKER}\n<!-- polyxios-pr-policy closed_at={CLOSED_AT} -->\n",
    }
    closed = [{"event": "closed", "actor": {"login": "maintainer"}}]
    call, writes = fake_github(
        issues={5: {"assignees": ["alice"]}},
        comments={100: [forged]},
        events={100: closed},
    )
    pr = make_pr(
        body="#5", state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}]
    )
    run_pr(call, pr)
    assert state_changes(writes) == []
    assert all("/comments/1" not in path for _, path, _ in writes)


def test_sidebar_link_is_checked_before_the_reference_cap():
    call, _ = fake_github(issues={99: {"assignees": ["alice"]}}, closing={100: [99]})
    body = " ".join(f"#{n}" for n in range(1, 21))
    assert run_pr(call, make_pr(body=body))[100] == {"linked": [99]}


def test_assignment_fetches_only_pulls_that_reference_the_issue():
    pulls = {
        100: make_pr(100, body="Closes #5"),
        101: make_pr(101, body="Closes #6"),
    }
    call, _ = fake_github(issues={5: {"assignees": ["alice"]}}, pulls=pulls)
    fetched = []
    inner = call

    def counting(method, path, *, params=None, payload=None):
        if method == "GET" and "/pulls/" in path:
            fetched.append(int(path.rsplit("/", 1)[1]))
        return inner(method, path, params=params, payload=payload)

    counting.pulls = inner.pulls
    assert run_assignment(counting, login="alice") == {100: {"linked": [5]}}
    assert fetched == [100]


def test_waiver_from_a_cancelled_reopen_run_is_kept():
    reopened = [
        {"event": "closed", "actor": {"login": POLICY_ACTOR}},
        {"event": "reopened", "actor": {"login": "maint"}},
    ]
    call, writes = fake_github(roles={"maint": "maintain"}, events={100: reopened})
    pr = make_pr(labels=[{"name": LABEL}], closed_at=CLOSED_AT)
    results = run_pr(call, pr)
    assert results[100] == {"skipped": "alice (waived by maint)"}
    assert state_changes(writes) == []
    (body,) = comment_bodies(writes)
    assert "waived_by=maint" in body


WAIVED_THEN_CLOSED = [
    {"event": "closed", "actor": {"login": POLICY_ACTOR}},
    {"event": "reopened", "actor": {"login": "maint"}},
    {"event": "closed", "actor": {"login": POLICY_ACTOR}},
]


def test_waiver_does_not_undo_another_workflows_close():
    call, writes = fake_github(
        roles={"maint": "maintain"},
        events={100: WAIVED_THEN_CLOSED},
        comments={100: [stamped(CLOSED_AT)]},
    )
    pr = make_pr(
        state="closed", closed_at="2026-10-20T08:00:00Z", labels=[{"name": LABEL}]
    )
    assert run_pr(call, pr)[100] == {"skipped": "alice (waived by maint)"}
    assert state_changes(writes) == []


def test_waiver_that_reopens_warns_that_ci_did_not_run():
    call, writes = fake_github(
        roles={"maint": "maintain"},
        events={100: WAIVED_THEN_CLOSED},
        comments={100: [stamped(CLOSED_AT)]},
    )
    pr = make_pr(state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}])
    assert run_pr(call, pr)[100] == {"skipped": "alice (waived by maint)"}
    assert state_changes(writes) == [(100, "open")]
    (body,) = comment_bodies(writes)
    assert pr_policy.REOPEN_NOTE in body


@pytest.mark.parametrize(
    "events",
    [
        [
            {"event": "closed", "actor": {"login": "maint"}},
            {"event": "reopened", "actor": {"login": "maint"}},
        ],
        [
            {"event": "closed", "actor": {"login": POLICY_ACTOR}},
            {"event": "reopened", "actor": {"login": "alice"}},
        ],
        [
            {"event": "closed", "actor": {"login": POLICY_ACTOR}},
            {"event": "reopened", "actor": {"login": "reader"}},
        ],
    ],
)
def test_reopen_event_that_does_not_waive(events):
    call, _ = fake_github(
        roles={"maint": "write", "alice": "write"}, events={100: events}
    )
    pr = make_pr(labels=[{"name": LABEL}])
    assert (
        pr_policy.find_waiver(call, owner=OWNER, repo=REPO, pr=pr, reopened_by=None)
        is None
    )


def test_held_pr_does_not_look_the_label_up_again():
    call, writes = fake_github(label=False)
    pr = make_pr(state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}])
    assert problems(run_pr(call, pr))
    assert all(not path.endswith("/labels") for _, path, _ in writes)


def test_assignment_reads_sidebar_links_in_one_query():
    pulls = {n: make_pr(n, body="nothing") for n in (100, 101, 102)}
    call, _ = fake_github(
        issues={5: {"assignees": ["alice"]}}, pulls=pulls, closing={101: [5]}
    )
    assert run_assignment(call, login="alice") == {101: {"linked": [5]}}
    assert len(call.graphql_queries) == 2


def test_sidebar_links_are_read_in_batches(monkeypatch):
    monkeypatch.setattr(pr_policy, "CLOSING_BATCH", 2)
    call, _ = fake_github(closing={3: [7], 5: [8]})
    linked = pr_policy.closing_refs(
        call, owner=OWNER, repo=REPO, numbers=[1, 2, 3, 4, 5]
    )
    assert linked == {1: [], 2: [], 3: [7], 4: [], 5: [8]}
    assert len(call.graphql_queries) == 3


def test_deleted_pull_does_not_fail_its_sidebar_batch():
    pulls = {
        100: make_pr(100, body="Closes #5"),
        101: make_pr(101, state="closed", labels=[{"name": LABEL}]),
    }
    call, writes = fake_github(issues={5: {"assignees": ["alice"]}}, pulls=pulls)

    def deleted(method, path, *, params=None, payload=None):
        if path != "/graphql":
            return call(method, path, params=params, payload=payload)
        return {
            "data": {"repository": {"pr0": None}},
            "errors": [{"type": "NOT_FOUND", "message": "Could not resolve"}],
        }

    deleted.pulls = call.pulls
    assert run_assignment(deleted, login="alice") == {100: {"linked": [5]}}
    assert statuses(writes) == [("sha100", "success")]


def test_other_graphql_errors_still_raise():
    def call(method, path, *, params=None, payload=None):
        return {
            "data": {"repository": {"pr0": None}},
            "errors": [{"type": "FORBIDDEN", "message": "no"}],
        }

    with pytest.raises(ApiError, match="no"):
        pr_policy.closing_refs(call, owner=OWNER, repo=REPO, numbers=[1])


def test_workflow_runs_for_every_base_branch():
    workflow = pr_policy.MEMBERS_FILE.parent / "workflows/pr_policy.yml"
    assert "branches:" not in workflow.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "body",
    [
        "-" * 65536,
        ".a" * 32768,
        "<!--" * 16384,
        "`<!--" * 13107,
        "a```" * 16384,
        "```x`\n" * 10922,
        "> - ```\n" * 8192,
        "``x`" * 16384,
        "a".join("`" * k for k in range(1, 362)),
    ],
)
def test_long_description_is_scanned_in_linear_time(body):
    start = time.perf_counter()
    assert pr_policy.extract_refs(f"{body} #5", owner=OWNER, repo=REPO) in (
        {},
        {5: "any"},
    )
    assert time.perf_counter() - start < 1


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("(#5)", {5: "any"}),
        ("x.#5 y-#6", {5: "any", 6: "any"}),
        ("x.fury-gl/polyxios#7", {}),
    ],
)
def test_owner_starts_at_a_token_boundary(body, expected):
    assert pr_policy.extract_refs(body, owner=OWNER, repo=REPO) == expected


@pytest.mark.parametrize(
    ("role", "skipped"),
    [(("release-manager", "write"), True), (("auditor", "read"), False)],
)
def test_custom_role_is_judged_by_its_base_permission(role, skipped):
    call, _ = fake_github(roles={"alice": role})
    results = run_pr(call, make_pr(association="CONTRIBUTOR"))
    assert ("skipped" in results[100]) is skipped


def test_policy_reopen_never_waives_the_check():
    events = [
        {"event": "closed", "actor": {"login": POLICY_ACTOR}},
        {"event": "reopened", "actor": {"login": POLICY_ACTOR}},
        {"event": "closed", "actor": {"login": POLICY_ACTOR}},
    ]
    call, writes = fake_github(roles={POLICY_ACTOR: "write"}, events={100: events})
    pr = make_pr(state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}])
    assert problems(run_pr(call, pr))
    assert all("waived_by" not in body for body in comment_bodies(writes))
    assert statuses(writes) == [("sha100", "failure")]


def test_pull_into_another_branch_is_left_alone():
    pulls = {100: make_pr(100, body="Closes #5", base={"ref": "maint/0.5.x"})}
    call, writes = fake_github(issues={5: {"assignees": []}}, pulls=pulls)
    assert run_assignment(call, login="alice") == {
        100: {"skipped": "targets maint/0.5.x"}
    }
    assert run_pr(call, pulls[100]) == {100: {"skipped": "targets maint/0.5.x"}}
    assert writes == []


def racing(call, *, on, then):
    """Wrap a fake so that the write ``on`` changes the world through ``then``."""

    def wrapped(method, path, *, params=None, payload=None):
        result = call(method, path, params=params, payload=payload)
        if (method, path, (payload or {}).get("state")) == on:
            then()
        return result

    wrapped.pulls = call.pulls
    return wrapped


def test_assignment_during_the_close_reopens():
    issues = {5: {"assignees": []}}
    pr = make_pr(body="Closes #5", association="CONTRIBUTOR")
    call, writes = fake_github(issues=issues, pulls={100: pr})

    def assign():
        issues[5]["assignees"] = ["alice"]
        pr.update(state="closed", closed_at=CLOSED_AT, labels=[{"name": LABEL}])

    on = ("PATCH", f"/repos/{OWNER}/{REPO}/pulls/100", "closed")
    results = run_pr(racing(call, on=on, then=assign), pr)
    assert results == {100: {"linked": [5]}}
    assert state_changes(writes) == [(100, "closed"), (100, "open")]
    assert statuses(writes) == [("sha100", "success")]


def test_unassignment_during_the_reopen_closes_again():
    issues = {5: {"assignees": ["alice"]}}
    pr = make_pr(
        body="Closes #5",
        association="CONTRIBUTOR",
        state="closed",
        closed_at=CLOSED_AT,
        labels=[{"name": LABEL}],
    )
    call, writes = fake_github(issues=issues, pulls={100: pr})

    def unassign():
        issues[5]["assignees"] = []
        pr.update(state="open", closed_at=None, labels=[])

    on = ("PATCH", f"/repos/{OWNER}/{REPO}/pulls/100", "open")
    results = run_pr(racing(call, on=on, then=unassign), pr)
    assert problems(results)
    assert state_changes(writes) == [(100, "open"), (100, "closed")]
    assert statuses(writes) == [("sha100", "failure")]


def test_unchanged_verdict_reads_each_issue_once():
    call, _ = fake_github(issues={5: {"assignees": ["bob"]}})
    seen = []
    inner = call

    def counting(method, path, *, params=None, payload=None):
        if method == "GET" and path.endswith("/issues/5"):
            seen.append(path)
        return inner(method, path, params=params, payload=payload)

    counting.pulls = inner.pulls
    pr = make_pr(body="#5", state="closed", labels=[{"name": LABEL}])
    assert problems(run_pr(counting, pr))
    assert len(seen) == 1


def test_merged_bot_pull_gets_no_status():
    call, writes = fake_github()
    pr = make_pr(
        user={"login": "dependabot[bot]", "type": "Bot"},
        association="NONE",
        state="closed",
        merged=True,
    )
    assert run_pr(call, pr) == {100: {"skipped": "merged"}}
    assert writes == []


def test_comment_posted_in_a_race_converges_on_the_oldest():
    call, writes = fake_github()
    path = f"/repos/{OWNER}/{REPO}/issues/100/comments"
    posted = []

    def wrapped(method, p, *, params=None, payload=None):
        if method == "GET" and p == path:
            other = {"id": 1, "user": BOT, "body": f"{MARKER}\nother run"}
            return [other, *posted] if posted else []
        result = call(method, p, params=params, payload=payload)
        if method == "POST" and p == path:
            posted.append({"id": 2, "user": BOT, "body": payload["body"]})
        return result

    wrapped.pulls = call.pulls
    run_pr(wrapped, make_pr())
    (mine,) = [p["body"] for m, p_, p in writes if m == "POST" and p_ == path]
    patches = [
        (p_, p["body"]) for m, p_, p in writes if m == "PATCH" and "/comments/" in p_
    ]
    assert patches == [(f"/repos/{OWNER}/{REPO}/issues/comments/1", mine)]
    assert ("DELETE", f"/repos/{OWNER}/{REPO}/issues/comments/2", None) in writes

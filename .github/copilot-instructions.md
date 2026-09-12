# Commit message convention

Every commit subject starts with one of these prefixes followed by a colon
and a space: `NF:` new feature, `BF:` bug fix, `RF:` refactoring, `MNT:`
maintenance, `DOC:` documentation, `TEST:` tests, `OPT:` optimization,
`CI:` continuous integration, `STYLE:` formatting, `BW:`
backward-compatibility, `WIP:` work in progress.

The subject is at most 78 characters, has no trailing period, and the
second line is blank. A fix to a GitHub Actions workflow is `CI:`; a
dependency bump is `MNT:`. Do not put an alert or issue number in the
subject; reference it from the pull request description instead.

Do not add a `Co-authored-by` trailer for a bot or an AI assistant.

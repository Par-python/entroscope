# Security Policy

## Supported versions

entroscope is pre-1.0 and ships from a single line of development. Security fixes
are released against the latest published version on PyPI.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

Please upgrade to the latest release before reporting an issue, in case it has
already been fixed.

## Scope

entroscope is a pure-computation library (numpy / scipy / pandas / matplotlib).
It does not open network connections, run a server, handle authentication, or
deserialize untrusted data. The most likely "security-relevant" issues are
therefore things like:

- a denial-of-service from pathological input (e.g. an input that causes
  unbounded memory or runtime),
- a dependency vulnerability surfaced through entroscope,
- incorrect handling of crafted input that leads to a crash.

Plain numerical inaccuracies or ordinary bugs are not security issues; please
file those as a normal [bug report](.github/ISSUE_TEMPLATE/bug_report.md).

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Disclosing it
publicly before a fix is available puts users at risk.

Instead, report it privately one of two ways:

1. **GitHub private advisory (preferred):** go to the repository's **Security**
   tab and choose **Report a vulnerability**. This opens a private channel
   between you and the maintainer.
2. **Email:** pardojeromeimportant@gmail.com with the details.

Please include:

- a description of the issue and why you believe it is a security concern,
- a minimal reproduction (input and code) if possible,
- the entroscope version and your environment.

## What to expect

- An acknowledgement of your report within a few days.
- An assessment of whether it is in scope and a planned fix if so.
- A coordinated disclosure: the fix ships in a new release, and credit is given
  to the reporter unless you prefer to remain anonymous.

Thanks for helping keep entroscope and its users safe.

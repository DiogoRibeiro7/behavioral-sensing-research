# Security Policy

## Supported Versions

Security fixes are prioritized for the latest released version and the active
development branch.

| Version | Supported |
| ------- | --------- |
| `0.1.x` | Yes       |

## Reporting a Vulnerability

Do not open a public GitHub issue for suspected security vulnerabilities.

Report security concerns privately by email:

- Diogo Ribeiro: <dfr@esmad.ipp.pt>

Please include:

- A clear description of the issue.
- A minimal reproduction, if possible.
- Affected version, commit, or deployment context.
- Whether the issue affects the Flask web application, file upload handling,
  report generation, dependency handling, or data privacy.
- Any known workaround or mitigation.

## Response Expectations

The maintainer will make a best effort to:

- acknowledge the report within 7 days;
- assess severity and affected versions;
- coordinate a fix before public disclosure when appropriate;
- credit reporters when requested and appropriate.

This is a research toolkit, not a regulated medical device or hosted clinical
service. Security reports are still taken seriously, especially issues involving
file upload handling, generated reports, dependency supply chain risk, or
accidental exposure of sensitive sensor data.

## Public Disclosure

Please avoid public disclosure until a fix or mitigation is available. After a
fix is released, security-relevant changes should be summarized in
`CHANGELOG.md` without exposing exploit details that would put users at risk.

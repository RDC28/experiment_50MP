"""Offline analytic run report, derived only from the completion data."""

from __future__ import annotations

import html
import json

# Match the existing synthraw report's system typography and neutral table style.
_CSS = """
body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
margin:0 auto;max-width:1100px;padding:24px;color:#1d1d1f;background:#fff}
h1{font-size:24px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid #ddd}
.notice{background:#f4f4f6;border-left:4px solid #6b6b75;padding:12px}
table{border-collapse:collapse;width:100%;table-layout:fixed}
th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font-size:12px}
a{color:#175ca4}details{margin:12px 0}summary{cursor:pointer;font-weight:600}
"""


def render_report(report: dict) -> str:
    """Render escaped metadata without scripts, network assets or image decoding.

    No measured or effective state is inferred from requested settings. Detailed
    stage evidence is retained verbatim; scientific acceptance is separate from
    successful file roundtrips. Artifacts are listed without untrusted URL links.
    """

    def escape(value):
        return html.escape(str(value), quote=True)

    def detail(title, value):
        encoded = json.dumps(value, indent=2, allow_nan=False)
        return f"<details><summary>{escape(title)}</summary><pre>{escape(encoded)}</pre></details>"

    rows = "".join(
        f"<tr><th scope='row'>{escape(name)}</th><td>{escape(record['size_bytes'])}</td>"
        f"<td>{escape(record['sha256'])}</td></tr>"
        for name, record in report["artifacts"].items()
    )
    pending = "".join(f"<li>{escape(item)}</li>" for item in report["pending"])
    masks = "".join(detail(name, record) for name, record in report["masks"].items())
    stages = "".join(detail(name, record) for name, record in report["stages"].items())
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Analytic MVP run report</title><style>{_CSS}</style></head><body>"
        "<h1>Analytic MVP run report</h1>"
        "<p class='notice'>Synthetic camera hypothesis. This output does not establish recovered scene "
        "radiance, measured Sony calibration or learned-model inference.</p>"
        f"<p>{escape(report['pipeline'])}</p>"
        f"<p><strong>Scientific acceptance: {escape(report['scientific_acceptance'])}</strong></p>"
        "<p>File roundtrip checks are recorded below and do not establish scientific fidelity. "
        "The manifest is the machine-readable completion record.</p>"
        "<h2>Artifacts</h2><table><thead><tr><th scope='col'>File</th>"
        "<th scope='col'>Bytes</th><th scope='col'>SHA-256</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<h2>Stage evidence</h2>{stages}<h2>Mask semantics</h2>{masks}"
        + (detail("Runtime and replay limits", report["runtime"]) if "runtime" in report else "")
        + "<h2>Shared hypotheses and provenance</h2>"
        + "".join(detail(name, report[name]) for name in ("source", "geometry", "camera", "isp", "registry", "limits"))
        + f"<h2>Pending work</h2><ul>{pending}</ul></body></html>"
    )

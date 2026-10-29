"""Report metadata remains inert, complete and explicit about scientific limits."""

from html.parser import HTMLParser

from prd_v5.report import render_report


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)

    def handle_data(self, data):
        self.text.append(data)


def test_report_escapes_untrusted_metadata_and_keeps_evidence():
    payload = '<script>alert("source")</script>&'
    report = {
        "pipeline": payload,
        "scientific_acceptance": "not measured",
        "artifacts": {payload: {"size_bytes": 12, "sha256": "abc"}},
        "pending": [payload, "Colab full-scale validation"],
        "masks": {"source_clipped": {"limitations": payload}},
        "stages": {"roundtrip": {"independent_validation": "passed tiny fixture"}},
        **{name: {"value": payload} for name in ("source", "geometry", "camera", "isp", "registry", "limits")},
    }
    page = render_report(report)
    parser = Tags()
    parser.feed(page)
    assert "script" not in parser.tags
    assert "img" not in parser.tags
    assert "link" not in parser.tags
    assert "<script>" not in page
    text = "".join(parser.text)
    assert payload in text
    assert "passed tiny fixture" in text
    assert "Scientific acceptance: not measured" in text
    assert "Colab full-scale validation" in text
    assert "Mask semantics" in text
    assert "registry" in text

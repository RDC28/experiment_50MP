"""Video v1: one video clip to a synthetic Sony A1 II style RAW frame sequence, reusing prd_v6 (README.md).

Importing the package is cheap on purpose: `video_v1.delivery` and `video_v1.report` use only the
standard library (plus `prd_v6.delivery`, which does too), so the notebook's `finally` block and
`download_results` work even when the install failed.
"""

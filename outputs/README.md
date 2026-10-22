# outputs

Full-resolution masters, one folder per category and run: `outputs/<category>/<run id>/`. Each master has a
JSON sidecar with its validation results and provenance; `intermediates/` holds the working-space source.
Masters are large and are not stored in Git. In Colab, set `OUTPUT_ROOT` to a Google Drive folder to keep them.

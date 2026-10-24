# outputs

Full-resolution masters, one folder per category and run: `outputs/<category>/<run id>/`. Each master has a
JSON sidecar with its validation results and provenance; `intermediates/` holds the working-space source.
Every reconstruction run (notebooks 00 to 03) also keeps `final/`, the run's one final image: its 16-bit TIFF
and EXR masters, saved again with `final_` in front and verified by SHA-256. Masters are large and are not
stored in Git. In Colab, set `OUTPUT_ROOT` to a Google Drive folder to keep them.

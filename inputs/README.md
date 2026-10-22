# inputs

Read-only copies of ingested source images, one folder per category (`human`, `night`, `universal`).
`ingest()` names each copy `<sha256 prefix>_<original name>`, removes write permission, and writes a JSON
sidecar with the file and pixel hashes used for duplicate detection. Source images are not stored in Git.

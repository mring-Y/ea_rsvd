# Preparing GitHub and Zenodo releases

The release helper creates a clean GitHub repository staging directory and a
Zenodo source/evidence archive from this repository. Both contain the same
authored sources, release documentation, citation metadata, and selected,
sanitized numerical evidence. They omit `.git`, build products, Python caches,
downloaded matrices, runtime output, and machine-local paths.

From the repository root, prepare both artifacts with:

```bash
python scripts/prepare_release.py
```

By default, the helper creates a new `ea_het_rsvd_repro_release_20261017`
directory next to the repository. It refuses to overwrite an existing output
directory. Use `--output-dir PATH` to choose a different new directory. The
`github/ea_het_rsvd_repro` child is the repository root to add to GitHub Desktop
or use with Git; it has no `.git` directory or remote configured.
`release_manifest.json` lists each payload file, size, and SHA-256 hash.
`SHA256SUMS.txt` also hashes the manifest. The GitHub and Zenodo ZIPs contain
the same tree and each has a checksum sidecar. More than 100 files can be
committed together using Git or GitHub Desktop; the web file picker has a
separate file-count limit. Extract the GitHub ZIP and commit its contents so
that source files can be browsed in the repository.

To create the GitHub repository, add `github/ea_het_rsvd_repro` to GitHub
Desktop, or initialize it from a terminal after creating an empty repository:

```bash
cd PATH/TO/github/ea_het_rsvd_repro
git init -b main
git add .
git commit -m "Prepare reproducible experiment release"
git remote add origin https://github.com/ACCOUNT/REPOSITORY.git
git push -u origin main
```

For Zenodo, open the existing software record and prepare a new version when
this release continues the same project. For a separate new record, relate it
to the previous campaign. Upload the Zenodo ZIP as the source/evidence archive,
set the software license to MIT, and review the title, creator, version, and
related-publication metadata. The previously cited DOI
`10.5281/zenodo.23262787` identifies the earlier campaign archive; it does not
identify this snapshot. Use the DOI provided for the new record/version when
updating the manuscript and `CITATION.cff`; do not reuse the previous version
DOI as the new snapshot DOI. A reserved DOI permits citation metadata to be
updated before publication. Rebuild the final archives after those edits so
that the GitHub snapshot and Zenodo content agree.

MIT applies to the project's original code and documentation. Third-party
components keep their respective licenses and notices; see
`benchmarks/published_gpu/THIRD_PARTY.md` and the license texts beside the
corresponding materials. Matrix datasets are not redistributed in these
artifacts; download instructions and source references are retained.

The author name and affiliation in `CITATION.cff` and the copyright holder in
`LICENSE` are intentional public attribution. No Git configuration is copied.
For future Git commits, author identity is determined by the committing user's
Git configuration; a GitHub-provided `noreply` address can keep a private email
out of commit metadata.

The repository's `.gitattributes` preserves original file bytes for checksum
verification across systems. Shell runners use LF endings. Do not normalize
the archived evidence files' line endings before verifying their manifests.

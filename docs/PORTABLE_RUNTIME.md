# Portable local runtime packaging

This is packaging infrastructure for the experimental LongCat engine. It is not
an accepted-quality release. The production app still needs a complete model
render from the packaged installation, visual acceptance and release QA.

## Build and verify

Run on the target Apple Silicon Mac with the pinned Python 3.12 environment:

```sh
.venv-longcat/bin/python scripts/prepare_portable_runtime.py \
  --environment .venv-longcat --include-engine --output .local/new-portable-engine
```

The builder clones Python's framework, standard library and isolated dependencies;
keeps environment packages separate from user-site packages; bundles non-system
native-library dependencies; rewrites their loader-relative references; and
ad-hoc signs changed binaries. It includes engine code only from an explicit
allowlist and verifies the MLX vendor files against the existing source lock.
Model weights, photographs, audio, project databases and generation reports are
not included. Use a new output folder; existing artifacts are preserved.

The wrapper at `.venv-longcat/bin/python` selects the bundled Python standard
library and disables user-site packages. Child Python processes inherit that
configuration. Use the wrapper rather than launching the framework interpreter
with an unrelated Python environment or isolated-mode flags.

For a runtime probe, copy the public model tokenizer into `tokenizer/` inside the
artifact, move the artifact to a different path, then run:

```sh
.venv-longcat/bin/python scripts/verify_portable_runtime.py \
  '/absolute/path/to/relocated runtime'
```

The verifier starts from an unrelated temporary working directory, restricts PATH,
blocks Python socket connections and checks actual loaded dylib paths. It performs
a small real Metal calculation, image/audio codec tests, loudness analysis,
Whisper input-feature preparation, English/Hindi/Hinglish tokenization and an
FFmpeg launch. Synthetic codec fixtures remain in memory and are never avatar
output. This is not a full network-disabled neural-render test.

## Measured evidence

On macOS 27.2 / Apple M5, the artifact audit covered 338 native binaries and found
no remaining absolute non-system library dependencies or external symlinks. The
relocated functional probe passed with 562 loaded native images and none from the
development installation. The actual native queue also launched the relocated
LongCat worker from a path containing spaces, received its missing-input failure
and persisted the recoverable state. It did not generate a video.

The native app currently still points at the development runtime. Do not switch
its default or call this a complete offline installer until the quality-selected
engine, full model files and a successful native generation have been validated
from the release installation.

## Platform and notices

The pinned `mlx-metal` wheel declares `macosx_26_0_arm64`; this runtime therefore
requires macOS 26 or later. The actual tested system is macOS 27.2. The installer
now rejects older systems before modifying the runtime.

Dependency wheel notices are retained, and notices for copied Python, OpenSSL,
mpdecimal, SQLite and xz components are collected. The Python binary relocation
and launcher changes are summarized in the artifact, consistent with the
[Python license documentation](https://docs.python.org/3.12/license.html).
SQLite's upstream statement dedicates its deliverable code and documentation to
the public domain; the artifact includes the
[official source reference](https://www.sqlite.org/copyright.html).

The bundled FFmpeg 9.0 binary reports `--enable-gpl` and GPL v2-or-later terms;
its own `-L` output is retained. That binary's terms differ from the BSD license
of the Python imageio-ffmpeg wrapper. Keep `license_review_required` true until
model, transitive dependency and codec redistribution/source-availability review
is complete. Automatic notice collection is not proof of release compliance.

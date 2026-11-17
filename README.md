# Avatar Studio for macOS

Avatar Studio is a native SwiftUI app that animates a portrait using recorded speech. Rendering runs locally on Apple Silicon. The project is an engineering preview. Output realism varies by portrait, speech, and rendering engine.

## Requirements

- Apple Silicon Mac running macOS 14 or later.
- Apple Command Line Tools with Swift 5.10 or later. Run `xcode-select --install` if needed.
- Python 3.12. With Homebrew installed, run `brew install python@3.12`.
- Internet access for initial setup and model downloads. Allow at least 15 GB of free disk space for FlashHead, Python packages, and render outputs.

## Build and launch

Open Terminal, clone the repository, and run:

```sh
git clone https://github.com/cyberzapp/Dora.git
cd Dora
bash scripts/build_app.sh
open 'dist/Avatar Studio.app'
```

This builds the Swift app and signs it locally. Keep the checkout in place because the app uses it to find its Python runtime and models. If you move the checkout, update the runtime path in the app’s Settings.

## Set up local rendering

FlashHead is the recommended starting engine for this preview. From the same checkout:

```sh
python3.12 scripts/install_flashhead.py
.venv/bin/python backend/flashhead_doctor.py
```

The installer creates `.venv`, installs the Python dependencies, checks Apple Metal, and downloads pinned models to `.local/`. Model downloads can take a while. Run the same installer again if a transfer is interrupted. The diagnostics should report `installation_ready: true`. Run these commands in a normal Terminal session so PyTorch can access Metal.

In the app, choose **FlashHead • evaluation** in Settings. In **Avatar Library**, import a clear portrait that you own or have permission to use. In **Create Video**, choose the portrait and add a recorded voice-over. Start with **Render test clip** and check progress in **Render Queue**. Open the completed result in **Projects** to play or export it. Use a short clip first to evaluate lip sync and facial motion on your own inputs.

The app stores imported media, projects, and renders under `~/Library/Application Support/Avatar Studio/`. Initial setup downloads model assets. Your inputs and inference stay on your Mac.

## Other engines and tests

- Legacy MuseTalk uses `bash scripts/install_runtime.sh`. Run `.venv/bin/python backend/worker.py doctor` for its diagnostics.
- Experimental LongCat has separate platform and model requirements. See [LongCat setup](docs/LONGCAT_SETUP.md).
- Ditto evaluation currently depends on an experimental local runtime and research assets. The standard setup above installs FlashHead.
- The [validation notes](docs/VALIDATION.md) describe tested workflows and remaining quality gaps. The [model notes](docs/MODELS.md) cover model dependencies and upstream terms.

After setting up a Python runtime, run:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
swift build --disable-sandbox
```

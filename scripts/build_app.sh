#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --disable-sandbox
app="${AVATAR_STUDIO_APP_PATH:-$PWD/dist/Avatar Studio.app}"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp .build/release/AvatarStudio "$app/Contents/MacOS/AvatarStudio"
/usr/bin/python3 - "$app" "${AVATAR_STUDIO_RUNTIME_ROOT:-$PWD}" <<'PY'
import plistlib, sys
from pathlib import Path
plist = {'CFBundleName':'Avatar Studio','CFBundleDisplayName':'Avatar Studio','CFBundleIdentifier':'local.avatarstudio.mac','CFBundleVersion':'1','CFBundleShortVersionString':'0.1.0','CFBundleExecutable':'AvatarStudio','CFBundlePackageType':'APPL','LSMinimumSystemVersion':'14.0','NSHighResolutionCapable':True,'NSCameraUsageDescription':'Record your own facial expressions and head movement for local avatar animation.','NSMicrophoneUsageDescription':'Record audio with your motion reference locally.','AvatarRuntimeRoot':sys.argv[2]}
Path(sys.argv[1]+'/Contents/Info.plist').write_bytes(plistlib.dumps(plist))
PY
codesign --force --deep --sign - "$app"
echo "$app"

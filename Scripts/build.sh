#!/bin/bash
# Build a Release Wake.app into ./build.
set -euo pipefail
cd "$(dirname "$0")/.."

xcodebuild \
  -project Wake.xcodeproj \
  -scheme Wake \
  -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath build \
  CODE_SIGN_IDENTITY="-" \
  CODE_SIGNING_REQUIRED=NO \
  ONLY_ACTIVE_ARCH=YES \
  build

echo "Built build/Build/Products/Release/Wake.app"

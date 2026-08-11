#!/usr/bin/env bash
set -euo pipefail

# Mirrors the macOS branch of apple-dev.sh: build the Rust side, regenerate
# Swift bindings from that exact binary, then compile one Swift executable
# and run it. There is no Xcode project and no XCTest — `apps/apple/Tests/`
# is plain top-level code (see `Tests/main.swift`).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$repo_root/.build/apple/test"
generated_root="$build_root/generated"
ffi_root="$build_root/ffi"
source_root="$repo_root/apps/apple/Sources"
tests_root="$repo_root/apps/apple/Tests"
bindgen="$repo_root/target/debug/uniffi-bindgen-swift"

mkdir -p "$generated_root" "$ffi_root"

cargo build \
  --manifest-path "$repo_root/Cargo.toml" \
  --locked \
  -p uniffi-bindgen-swift

cargo build \
  --manifest-path "$repo_root/Cargo.toml" \
  --locked \
  --release \
  -p sinus-apple
rust_library="$repo_root/target/release/libsinus_apple.a"

"$bindgen" --swift-sources "$rust_library" "$generated_root"
"$bindgen" --headers "$rust_library" "$ffi_root"
"$bindgen" \
  --modulemap \
  --module-name SinusAppleFFI \
  --modulemap-filename module.modulemap \
  "$rust_library" \
  "$ffi_root"
"$repo_root/scripts/normalize-apple-bindings.sh" "$generated_root" "$ffi_root"

# Curated, not globbed (unlike apple-dev.sh, which globs all of Sources/):
# the test binary only needs the model layer this test suite exercises, plus
# whatever those files themselves need to compile. Globbing all of Sources/
# would pull SwiftUI/Charts/CoreML views into a binary that never uses them,
# and silently widen this list every time an unrelated view is added.
model_sources=(
  "$source_root/Models/FeedbackMessageFormatter.swift"
  "$source_root/Models/EventTypeDisplay.swift"
  "$source_root/Models/HistoryEngineProtocol.swift"
  "$source_root/Models/HistoryModel.swift"
  "$source_root/Models/TrainingEngineProtocol.swift"
  "$source_root/Models/TrainingModel.swift"
  # TrainingModel.attach()/finishTake() name this type; TrainingModelTests
  # constructs one (over a fake, handle-less AppleEngine) to satisfy it.
  "$source_root/Platform/AudioMonitoringService.swift"
)

test_sources=()
while IFS= read -r file; do
  test_sources+=("$file")
done < <(find "$tests_root" -name '*.swift' -type f | sort)
if [[ ${#test_sources[@]} -eq 0 ]]; then
  echo "no Swift sources found under $tests_root" >&2
  exit 1
fi

sdk_path="$(xcrun --sdk macosx --show-sdk-path)"
binary="$build_root/sinus-apple-tests"

xcrun --sdk macosx swiftc \
  -sdk "$sdk_path" \
  -module-name SinusSentinelTests \
  -I "$ffi_root" \
  "$generated_root/SinusApple.swift" \
  "${model_sources[@]}" \
  "${test_sources[@]}" \
  "$rust_library" \
  -framework AVFoundation \
  -o "$binary"

"$binary"

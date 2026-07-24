#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
action="${2:-run}"
case "$mode" in
  ios|macos) ;;
  *)
    echo "usage: $0 ios|macos [build|run]" >&2
    exit 2
    ;;
esac
case "$action" in
  build|run) ;;
  *)
    echo "usage: $0 ios|macos [build|run]" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$repo_root/.build/apple/$mode"
generated_root="$build_root/generated"
ffi_root="$build_root/ffi"
app_root="$build_root/SinusSentinel.app"
source_root="$repo_root/apps/apple/Sources"
bindgen="$repo_root/target/debug/uniffi-bindgen-swift"

mkdir -p "$generated_root" "$ffi_root"

cargo build \
  --manifest-path "$repo_root/Cargo.toml" \
  --locked \
  -p uniffi-bindgen-swift

if [[ "$mode" == "ios" ]]; then
  case "$(uname -m)" in
    arm64)
      rust_target="aarch64-apple-ios-sim"
      swift_target="arm64-apple-ios17.0-simulator"
      ;;
    x86_64)
      rust_target="x86_64-apple-ios"
      swift_target="x86_64-apple-ios17.0-simulator"
      ;;
    *)
      echo "unsupported Mac architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac
  rustup target add "$rust_target"
  cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --locked \
    --release \
    -p sinus-apple \
    --target "$rust_target"
  rust_library="$repo_root/target/$rust_target/release/libsinus_apple.a"
  sdk="iphonesimulator"
  info_plist="$repo_root/apps/apple/Resources/Info-iOS.plist"
else
  cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --locked \
    --release \
    -p sinus-apple
  rust_library="$repo_root/target/release/libsinus_apple.a"
  sdk="macosx"
  info_plist="$repo_root/apps/apple/Resources/Info-macOS.plist"
fi

"$bindgen" --swift-sources "$rust_library" "$generated_root"
"$bindgen" --headers "$rust_library" "$ffi_root"
"$bindgen" \
  --modulemap \
  --module-name SinusAppleFFI \
  --modulemap-filename module.modulemap \
  "$rust_library" \
  "$ffi_root"
"$repo_root/scripts/normalize-apple-bindings.sh" "$generated_root" "$ffi_root"

rm -rf "$app_root"
mkdir -p "$app_root/Contents/MacOS" "$app_root/Contents/Resources"

sdk_path="$(xcrun --sdk "$sdk" --show-sdk-path)"
swift_args=(
  -sdk "$sdk_path"
  -parse-as-library
  -module-name SinusSentinel
  -I "$ffi_root"
  "$generated_root/SinusApple.swift"
  "$source_root/AppModel.swift"
  "$source_root/AudioMonitoringService.swift"
  "$source_root/ContentView.swift"
  "$source_root/HistoryChartView.swift"
  "$source_root/ModelRunners.swift"
  "$source_root/SinusSentinelApp.swift"
  "$rust_library"
  -framework AVFoundation
  -framework Charts
  -framework CoreML
  -framework Security
  -framework SwiftUI
)

if [[ "$mode" == "ios" ]]; then
  rm -rf "$app_root/Contents"
  cp "$info_plist" "$app_root/Info.plist"
  if [[ -d "$repo_root/apps/apple/Resources/yamnet.mlmodelc" ]]; then
    cp -R "$repo_root/apps/apple/Resources/yamnet.mlmodelc" "$app_root/"
  fi
  xcrun --sdk "$sdk" swiftc \
    -target "$swift_target" \
    "${swift_args[@]}" \
    -o "$app_root/SinusSentinel"
  codesign --force --sign - "$app_root"
  if [[ "$action" == "build" ]]; then
    echo "built $app_root"
    exit 0
  fi

  simulator_id="$(
    xcrun simctl list devices booted |
      sed -nE 's/.*\(([0-9A-F-]{36})\) \(Booted\).*/\1/p' |
      head -n 1
  )"
  if [[ -z "$simulator_id" ]]; then
    simulator_id="$(
      xcrun simctl list devices available |
        sed -nE 's/.*\(([0-9A-F-]{36})\) \(Shutdown\).*/\1/p' |
        head -n 1
    )"
    if [[ -z "$simulator_id" ]]; then
      echo "no available iOS Simulator device found" >&2
      exit 1
    fi
    xcrun simctl boot "$simulator_id"
  fi
  open -a Simulator
  xcrun simctl bootstatus "$simulator_id" -b
  xcrun simctl install "$simulator_id" "$app_root"
  xcrun simctl launch "$simulator_id" com.sinussentinel.prototype
else
  cp "$info_plist" "$app_root/Contents/Info.plist"
  if [[ -d "$repo_root/apps/apple/Resources/yamnet.mlmodelc" ]]; then
    cp -R \
      "$repo_root/apps/apple/Resources/yamnet.mlmodelc" \
      "$app_root/Contents/Resources/"
  fi
  xcrun --sdk "$sdk" swiftc \
    "${swift_args[@]}" \
    -o "$app_root/Contents/MacOS/SinusSentinel"
  codesign --force --sign - "$app_root"
  if [[ "$action" == "run" ]]; then
    open "$app_root"
  else
    echo "built $app_root"
  fi
fi

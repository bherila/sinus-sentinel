import Foundation

// No XCTest: `scripts/apple-test.sh` compiles this directory with plain
// `swiftc`, so the harness is one executable rather than an Xcode test
// bundle. Without `-parse-as-library`, exactly one file may hold top-level
// statements — this one — and every test function below lives in a sibling
// file and gets called from here.

var passCount = 0
var failureCount = 0

/// The one assertion primitive every test file uses. Prints only on
/// failure, so a clean run's output is just the final tally.
func expect(_ condition: @autoclosure () -> Bool, _ label: String) {
    if condition() {
        passCount += 1
    } else {
        failureCount += 1
        print("FAIL: \(label)")
    }
}

// `HistoryModel`/`TrainingModel` are `@MainActor`; top-level code in a
// `main.swift` is not automatically isolated to it, so the test bodies run
// inside an explicit hop rather than reaching for `MainActor.assumeIsolated`
// on an unstarted actor.
await MainActor.run {
    testFeedbackMessageFormatter()
    testHistoryModel()
    testTrainingModel()
    testServerUrlNormalizer()
    testConnectionCheckMessage()
    testDevicePairing()
}

if failureCount > 0 {
    print("\(failureCount) failing, \(passCount) passing.")
    exit(1)
} else {
    print("\(passCount) passing.")
}

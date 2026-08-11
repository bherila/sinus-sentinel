/// Every `AppleConnectionCheck` outcome's exact message — a wording change
/// here is a wording change a user reads in Settings › PHR, so this checks
/// the literal strings rather than just "returns something".
func testConnectionCheckMessage() {
    expect(
        ConnectionCheckMessage.message(for: .ok) == "Connected — the server accepted the token.",
        "ok"
    )
    expect(
        ConnectionCheckMessage.message(for: .unauthenticated)
            == "The server rejected the stored token (expired, revoked, or wrong). Create a new API key in the PHR and save it here.",
        "unauthenticated"
    )
    expect(
        ConnectionCheckMessage.message(for: .patientNotFound)
            == "Connected and authenticated, but this patient id does not exist on that server.",
        "patientNotFound"
    )
    expect(
        ConnectionCheckMessage.message(for: .forbidden)
            == "Connected and authenticated, but the token does not grant access to this patient.",
        "forbidden"
    )
    expect(
        ConnectionCheckMessage.message(for: .noServerUrl) == "No server URL is set — add one above.",
        "noServerUrl"
    )
    expect(
        ConnectionCheckMessage.message(for: .noPatientId) == "No patient id is set — add one above.",
        "noPatientId"
    )
    expect(
        ConnectionCheckMessage.message(for: .noToken)
            == "No API token is set — add one in the API token section above.",
        "noToken"
    )
    expect(
        ConnectionCheckMessage.message(for: .offlineStrict)
            == "Sync mode is offline-strict, which never makes network calls. Switch modes above to test the connection.",
        "offlineStrict"
    )
    expect(
        ConnectionCheckMessage.message(for: .unreachable(detail: "connection refused"))
            == "Could not reach the server: connection refused",
        "unreachable carries its detail through verbatim"
    )
    expect(
        ConnectionCheckMessage.message(for: .http(status: 500)) == "Unexpected server response (HTTP 500).",
        "http carries its status through"
    )

    expect(!ConnectionCheckMessage.isFailure(.ok), "ok is the only non-failure")
    let failures: [AppleConnectionCheck] = [
        .unauthenticated, .patientNotFound, .forbidden,
        .noServerUrl, .noPatientId, .noToken, .offlineStrict,
        .unreachable(detail: "x"), .http(status: 404),
    ]
    for outcome in failures {
        expect(ConnectionCheckMessage.isFailure(outcome), "every non-ok outcome is a failure")
    }
}

import Foundation
import AuthenticationServices

#if os(macOS)
import AppKit
#endif

/// Drives "Sign in with PHR" end to end: builds the authorize request from
/// `DevicePairing`, runs it through `ASWebAuthenticationSession` so the
/// user's existing PHR browser session carries over (that reuse *is* the
/// smooth login — hence `prefersEphemeralWebBrowserSession = false`), then
/// exchanges the returned one-time code for a token.
///
/// Kept thin and UI-adjacent: the pure request/response shapes it calls into
/// live in `Models/DevicePairing.swift`, which the test harness links
/// directly. This file needs `AuthenticationServices`, which the harness does
/// not — see `scripts/apple-test.sh`'s curated source list, which leaves it
/// out on purpose.
@MainActor
final class PhrSignInSession {
    enum Failure: Error, Equatable {
        case cancelled
        case denied
        case invalidCode
        case network(String)
        case server(String)
    }

    enum Outcome: Equatable {
        case success(DevicePairing.ExchangeSuccess)
        case failure(Failure)
    }

    private let serverRoot: String
    private let deviceId: String
    private let deviceName: String

    /// Both held for the lifetime of the flow: `ASWebAuthenticationSession`
    /// only keeps a *weak* reference to its `presentationContextProvider`,
    /// and the session itself would be torn down mid-flight if nothing here
    /// retained it either.
    private var webAuthSession: ASWebAuthenticationSession?
    private var contextProvider: PresentationContextProvider?

    init(serverRoot: String, deviceId: String, deviceName: String) {
        self.serverRoot = serverRoot
        self.deviceId = deviceId
        self.deviceName = deviceName
    }

    /// Runs the whole flow and returns its outcome; never throws — every
    /// failure mode is a case of `Outcome`/`Failure` instead, so callers get
    /// one switch rather than a mix of `throws` and typed results.
    func signIn() async -> Outcome {
        guard let request = DevicePairing.makeRequest(
            serverRoot: serverRoot,
            deviceId: deviceId,
            deviceName: deviceName
        ) else {
            return .failure(.server("Could not build the sign-in request for that server URL."))
        }

        let callbackURL: URL
        do {
            callbackURL = try await authorize(url: request.url)
        } catch let failure as Failure {
            return .failure(failure)
        } catch {
            return .failure(.network(error.localizedDescription))
        }

        switch DevicePairing.code(fromCallback: callbackURL) {
        case .code(let code):
            return await exchange(code: code, verifier: request.verifier)
        case .denied:
            return .failure(.denied)
        case .malformed:
            return .failure(.server("The browser returned an unexpected response."))
        }
    }

    /// Presents the authorize URL and suspends until the callback (or a
    /// cancel/error) arrives.
    private func authorize(url: URL) async throws -> URL {
        #if os(macOS)
        // The Settings window is open whenever this runs (it is what the
        // user just clicked the button in), so failing here rather than
        // force-unwrapping is only a defense against a window closing out
        // from under this call, not the expected path.
        guard let anchor = NSApplication.shared.keyWindow ?? NSApplication.shared.windows.first else {
            throw Failure.server("No window is available to present sign-in from — open Settings and try again.")
        }
        let contextProvider = PresentationContextProvider(anchor: anchor)
        #else
        let contextProvider = PresentationContextProvider()
        #endif
        self.contextProvider = contextProvider

        return try await withCheckedThrowingContinuation { continuation in
            let session = ASWebAuthenticationSession(
                url: url,
                callbackURLScheme: "sinussentinel"
            ) { callbackURL, error in
                if let callbackURL {
                    continuation.resume(returning: callbackURL)
                    return
                }
                if let authError = error as? ASWebAuthenticationSessionError,
                    authError.code == .canceledLogin {
                    continuation.resume(throwing: Failure.cancelled)
                    return
                }
                continuation.resume(throwing: Failure.network(error?.localizedDescription ?? "Unknown error"))
            }
            session.prefersEphemeralWebBrowserSession = false
            session.presentationContextProvider = contextProvider
            webAuthSession = session
            if !session.start() {
                continuation.resume(throwing: Failure.network("Could not start the sign-in session."))
            }
        }
    }

    /// POSTs the exchange and maps its outcome onto `Outcome`/`Failure`.
    /// `.throttled` (429) has no dedicated `Failure` case per the spec — it
    /// folds into `.server` with its own wording, same as any other
    /// unrecognized server response.
    private func exchange(code: String, verifier: String) async -> Outcome {
        guard let exchangeURL = URL(string: "\(serverRoot)/api/device-pairing/exchange") else {
            return .failure(.server("Could not build the exchange request URL."))
        }
        var request = URLRequest(url: exchangeURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = DevicePairing.exchangeBody(code: code, verifier: verifier, deviceId: deviceId)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            return .failure(.network(error.localizedDescription))
        }
        guard let httpResponse = response as? HTTPURLResponse else {
            return .failure(.network("The server did not return a valid HTTP response."))
        }

        switch DevicePairing.parseExchangeResponse(status: httpResponse.statusCode, data: data) {
        case .success(let success):
            return .success(success)
        case .invalidCode:
            return .failure(.invalidCode)
        case .throttled:
            return .failure(.server("The server is throttling pairing attempts — wait a moment and try again."))
        case .serverError(let message):
            return .failure(.server(message))
        }
    }
}

/// `ASWebAuthenticationPresentationContextProviding`'s conformer, kept as its
/// own, non-`@MainActor` type — mirrors `SyncStatusBridge` in
/// `SyncModel.swift`: `AuthenticationServices` calls `presentationAnchor`
/// itself, so pinning the conformance to an actor here would fight the
/// framework rather than the framework fighting back at runtime.
private final class PresentationContextProvider: NSObject, ASWebAuthenticationPresentationContextProviding {
    #if os(macOS)
    private let anchor: NSWindow

    init(anchor: NSWindow) {
        self.anchor = anchor
    }

    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        anchor
    }
    #else
    // iOS: the default anchor is fine here — `ASWebAuthenticationSession`
    // presents from the active scene either way, and hunting for a specific
    // `UIWindow` would be extra window-scene plumbing this flow does not
    // need.
    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        ASPresentationAnchor()
    }
    #endif
}

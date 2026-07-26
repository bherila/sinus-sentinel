import SwiftUI
import SinusAppleFFI

/// Settings › PHR. Mirrors the desktop tray's PHR section (`app.rs:689-750`)
/// field-for-field, including its status strings, so a user running both
/// shells sees the same thing.
struct PhrSettingsView: View {
    @Environment(EngineHost.self) private var host

    @State private var serverUrlText = ""
    @State private var patientIdText = ""
    @State private var tokenText = ""
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case serverUrl
        case patientId
    }

    var body: some View {
        let sync = host.sync

        Form {
            Section("Connection") {
                TextField("Server URL", text: $serverUrlText)
                    .focused($focusedField, equals: .serverUrl)
                    .onSubmit { sync.setServerUrl(serverUrlText) }

                TextField("Patient id", text: $patientIdText)
                    .focused($focusedField, equals: .patientId)
                    .onSubmit { sync.setPatientId(patientIdText) }
                    .help("The PHR patient these events belong to")

                Picker("Mode", selection: Binding(
                    get: { sync.phr?.mode ?? .autoBatch },
                    set: { sync.setSyncMode($0) }
                )) {
                    Text("Auto-batch").tag(SyncMode.autoBatch)
                    Text("Offline-first").tag(SyncMode.offlineFirst)
                    Text("Offline-strict (never uploads)").tag(SyncMode.offlineStrict)
                }
            }

            Section("API token") {
                SecureField("API token", text: $tokenText)
                HStack {
                    Button("Save token") {
                        sync.saveToken(tokenText)
                        tokenText = ""
                    }
                    Button("Check token") {
                        sync.checkToken()
                    }
                    .help("Checks only whether a token exists; never displays it")
                    Button("Remove token") {
                        sync.removeToken()
                    }
                }
                Text(sync.tokenStatus)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            // Its own row rather than inside "API token": the same field also
            // carries the patient-id validation message, and a sectioned form
            // would otherwise blame the token for it.
            if let message = sync.message {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("Status") {
                LabeledContent("Sync", value: sync.stateLabel)
                // Deliberately two numbers, not one: pending events is the
                // badge count, pending work is what actually gates a flush
                // (it also counts tombstones, flags and enrollments) — a
                // user seeing "0 pending events" while a sync keeps running
                // deserves to see why.
                LabeledContent("Pending events", value: "\(sync.status.pendingEvents)")
                LabeledContent("Pending work (gates flush)", value: "\(sync.status.pendingWork)")
                LabeledContent("Last successful sync", value: sync.lastSuccessDescription)
                if sync.status.quiet {
                    Text("Quiet hours are suppressing sync right now.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                if let error = sync.mappedError {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
        }
        .formStyle(.grouped)
        .onChange(of: focusedField) { oldValue, newValue in
            if oldValue == .serverUrl && newValue != .serverUrl {
                sync.setServerUrl(serverUrlText)
            }
            if oldValue == .patientId && newValue != .patientId {
                sync.setPatientId(patientIdText)
            }
        }
        .onAppear {
            sync.reload()
            serverUrlText = sync.phr?.serverUrl ?? ""
            patientIdText = sync.phr?.patientId.map(String.init) ?? ""
        }
    }
}

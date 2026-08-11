/// User-facing names for the seven-class taxonomy. Split out of
/// `HistoryChartView` (which keeps `.color`) because feedback messaging in
/// `Models/` needs the name and nothing chart- or SwiftUI-related.
extension AppleEventType {
    var displayName: String {
        switch self {
        case .cough: "Cough"
        case .throatClearing: "Throat clearing"
        case .sniffle: "Sniffle"
        case .sneeze: "Sneeze"
        case .noseBlow: "Nose blow"
        case .hawk: "Hawk"
        case .snortSuck: "Snort / suck"
        }
    }
}

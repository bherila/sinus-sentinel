//! Core domain types: the event taxonomy (SPEC §2) and the stored event record
//! (SPEC §4.2).

use serde::{Deserialize, Serialize};

/// Target sample rate for the whole pipeline (SPEC §4.1).
pub const SAMPLE_RATE: u32 = 16_000;

/// The seven sinus/airway event types (SPEC §2). Serialized form matches the
/// `event_type` strings used by the store and the PHR API contract (SPEC §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    Cough,
    ThroatClearing,
    Sniffle,
    Sneeze,
    NoseBlow,
    Hawk,
    SnortSuck,
}

impl EventType {
    /// All variants, in taxonomy order.
    pub const ALL: [EventType; 7] = [
        EventType::Cough,
        EventType::ThroatClearing,
        EventType::Sniffle,
        EventType::Sneeze,
        EventType::NoseBlow,
        EventType::Hawk,
        EventType::SnortSuck,
    ];

    /// The stable snake_case identifier (matches serde + the API taxonomy).
    pub fn as_str(self) -> &'static str {
        match self {
            EventType::Cough => "cough",
            EventType::ThroatClearing => "throat_clearing",
            EventType::Sniffle => "sniffle",
            EventType::Sneeze => "sneeze",
            EventType::NoseBlow => "nose_blow",
            EventType::Hawk => "hawk",
            EventType::SnortSuck => "snort_suck",
        }
    }

    /// Parse from the snake_case identifier.
    pub fn parse(s: &str) -> Option<EventType> {
        EventType::ALL.into_iter().find(|e| e.as_str() == s)
    }

    /// Weight in the daily congestion score (SPEC §2). Blockage indicators
    /// (`sniffle`, `snort_suck`) weigh double; clearing actions weigh 1; cough 0.5.
    pub fn congestion_weight(self) -> f64 {
        match self {
            EventType::Sniffle | EventType::SnortSuck => 2.0,
            EventType::NoseBlow | EventType::Hawk | EventType::ThroatClearing => 1.0,
            EventType::Cough => 0.5,
            EventType::Sneeze => 0.0,
        }
    }

    /// Transient classes fire on a single window ≥ θ_c (SPEC §4.1 stage ④).
    pub fn is_transient(self) -> bool {
        matches!(
            self,
            EventType::Cough | EventType::Sneeze | EventType::NoseBlow | EventType::Hawk
        )
    }

    /// Weak/short classes require a coincident gate-energy peak (SPEC §4.1 ④).
    pub fn is_weak(self) -> bool {
        matches!(
            self,
            EventType::Sniffle | EventType::SnortSuck | EventType::ThroatClearing
        )
    }
}

/// Origin platform tag stored with each event (SPEC §4.2 `source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    DesktopMac,
    DesktopWin,
    MobileIos,
    MobileAndroid,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::DesktopMac => "desktop-mac",
            Source::DesktopWin => "desktop-win",
            Source::MobileIos => "mobile-ios",
            Source::MobileAndroid => "mobile-android",
        }
    }

    /// The source for the host this build runs on.
    pub fn current_desktop() -> Source {
        #[cfg(target_os = "windows")]
        {
            Source::DesktopWin
        }
        #[cfg(not(target_os = "windows"))]
        {
            Source::DesktopMac
        }
    }
}

/// A stored respiratory event — mirrors the `events` table (SPEC §4.2) and the
/// per-event payload of the batch upload contract (SPEC §7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// `client_event_uuid` — idempotency key.
    pub uuid: String,
    pub event_type: EventType,
    /// UTC ISO-8601.
    pub occurred_at: chrono::DateTime<chrono::Utc>,
    /// Local UTC offset in minutes, for "morning vs night" charts.
    pub tz_offset_min: i32,
    pub duration_ms: i64,
    pub confidence: f32,
    pub burst_count: i64,
    pub model_version: String,
    pub source: Source,
    /// Stable per-install UUID.
    pub device_id: String,
    /// `None` = pending upload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uploaded_at: Option<chrono::DateTime<chrono::Utc>>,
    /// User removed a local false positive.
    #[serde(default)]
    pub deleted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_type_roundtrips_through_str() {
        for et in EventType::ALL {
            assert_eq!(EventType::parse(et.as_str()), Some(et));
        }
        assert_eq!(EventType::parse("nope"), None);
    }

    #[test]
    fn serde_uses_snake_case() {
        let j = serde_json::to_string(&EventType::SnortSuck).unwrap();
        assert_eq!(j, "\"snort_suck\"");
    }

    #[test]
    fn transient_and_weak_partition_matches_spec() {
        // Every class is exactly one of transient or weak (sneeze counts transient).
        for et in EventType::ALL {
            assert_ne!(et.is_transient(), et.is_weak(), "{et:?}");
        }
    }

    #[test]
    fn congestion_weights_match_spec_formula() {
        assert_eq!(EventType::Sniffle.congestion_weight(), 2.0);
        assert_eq!(EventType::SnortSuck.congestion_weight(), 2.0);
        assert_eq!(EventType::Cough.congestion_weight(), 0.5);
        assert_eq!(EventType::Hawk.congestion_weight(), 1.0);
    }

    #[test]
    fn source_strings_match_schema() {
        assert_eq!(Source::DesktopMac.as_str(), "desktop-mac");
        assert_eq!(
            serde_json::to_string(&Source::MobileAndroid).unwrap(),
            "\"mobile-android\""
        );
    }
}

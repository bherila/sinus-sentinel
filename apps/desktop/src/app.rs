//! egui app + tray integration (SPEC §6). The tray gives at-a-glance status and
//! quick actions; a single egui window hosts the History and Settings views. All
//! data logic lives in [`crate::state`] (unit-tested); this file is the renderer.
//!
//! Mic permission and interactive tray behaviour can't be verified in CI — this
//! layer is kept compiling and clippy-clean, with the testable logic factored out.

use chrono::Utc;
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Plot};
use sinus_core::store::Store;
use sinus_core::sync::Mode;
use sinus_core::types::EventType;

use crate::shared::{ModelStatus, SharedStatus};
use crate::state::{self, PauseState};

/// Menu item ids.
mod ids {
    pub const PAUSE_15: &str = "pause_15";
    pub const PAUSE_60: &str = "pause_60";
    pub const PAUSE_INDEF: &str = "pause_indef";
    pub const RESUME: &str = "resume";
    pub const MODE_AUTO: &str = "mode_auto";
    pub const MODE_OFFLINE_FIRST: &str = "mode_offline_first";
    pub const MODE_OFFLINE_STRICT: &str = "mode_offline_strict";
    pub const SYNC_NOW: &str = "sync_now";
    pub const OPEN_HISTORY: &str = "open_history";
    pub const OPEN_SETTINGS: &str = "open_settings";
    pub const QUIT: &str = "quit";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    History,
    Settings,
}

/// Editable settings mirrored into the store.
#[derive(Debug, Clone, Default)]
struct SettingsForm {
    server_url: String,
    token: String,
    sensitivity: f32,
}

pub struct SinusApp {
    store: Store,
    pause: PauseState,
    mode: Mode,
    tab: Tab,
    form: SettingsForm,
    // The tray icon is held for its lifetime; menu events are polled globally.
    #[cfg(not(test))]
    _tray: Option<tray_icon::TrayIcon>,
    device_id: String,
    shared: SharedStatus,
}

impl SinusApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, store: Store, shared: SharedStatus) -> Self {
        let sensitivity = store
            .setting_get("sensitivity")
            .ok()
            .flatten()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let server_url = store
            .setting_get("server_url")
            .ok()
            .flatten()
            .unwrap_or_default();
        let mode = store
            .setting_get("mode")
            .ok()
            .flatten()
            .map(|s| match s.as_str() {
                "offline-first" => Mode::OfflineFirst,
                "offline-strict" => Mode::OfflineStrict,
                _ => Mode::AutoBatch,
            })
            .unwrap_or(Mode::AutoBatch);
        let device_id = ensure_device_id(&store);

        SinusApp {
            store,
            pause: PauseState::Running,
            mode,
            tab: Tab::History,
            form: SettingsForm {
                server_url,
                token: String::new(),
                sensitivity,
            },
            #[cfg(not(test))]
            _tray: build_tray().ok(),
            device_id,
            shared,
        }
    }

    fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        let _ = self.store.setting_set("mode", mode.as_str());
    }

    fn handle_menu_events(&mut self, ctx: &egui::Context) {
        while let Ok(event) = tray_icon::menu::MenuEvent::receiver().try_recv() {
            let id = event.id.as_ref();
            let now = Utc::now();
            match id {
                ids::PAUSE_15 => {
                    self.pause = PauseState::PausedUntil(now + chrono::Duration::minutes(15))
                }
                ids::PAUSE_60 => {
                    self.pause = PauseState::PausedUntil(now + chrono::Duration::minutes(60))
                }
                ids::PAUSE_INDEF => self.pause = PauseState::PausedIndefinite,
                ids::RESUME => self.pause = PauseState::Running,
                ids::MODE_AUTO => self.set_mode(Mode::AutoBatch),
                ids::MODE_OFFLINE_FIRST => self.set_mode(Mode::OfflineFirst),
                ids::MODE_OFFLINE_STRICT => self.set_mode(Mode::OfflineStrict),
                ids::SYNC_NOW => self.shared.request_sync_now(),
                ids::OPEN_HISTORY => self.tab = Tab::History,
                ids::OPEN_SETTINGS => self.tab = Tab::Settings,
                ids::QUIT => {
                    // Signal the sync thread to attempt a final flush (auto-batch
                    // flushes on quit, SPEC §4.3) before the window closes.
                    self.shared.set_quitting(true);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                _ => {}
            }
        }
    }

    fn status_glyph(&mut self) -> &'static str {
        let now = Utc::now();
        self.pause = self.pause.normalized(now);
        // A missing model (fail-soft fallback in the capture thread) is a warning
        // state (SPEC §6 tray "⚠"), shown ahead of the plain listening glyph.
        match self.mode {
            Mode::OfflineStrict => "📴",
            _ if self.pause.is_paused(now) => "⏸",
            _ if self.shared.model() == ModelStatus::Missing => "⚠",
            _ => "🟢",
        }
    }

    fn draw_history(&mut self, ui: &mut egui::Ui) {
        let now = Utc::now();
        let today = state::today_counts(&self.store, now);

        ui.heading("Today");
        ui.horizontal_wrapped(|ui| {
            for et in EventType::ALL {
                let n = today.get(&et).copied().unwrap_or(0);
                if n > 0 {
                    ui.label(format!("{}: {n}", et.as_str()));
                    ui.separator();
                }
            }
            if today.values().all(|&n| n == 0) {
                ui.label("no events yet today");
            }
        });
        let monitored_hours = (now - now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc())
            .num_minutes() as f64
            / 60.0;
        ui.label(format!(
            "congestion score: {:.2} / monitored hour",
            state::congestion_score(&today, monitored_hours.max(0.1))
        ));

        ui.separator();
        ui.heading("Last 7 days");
        let hist = state::daily_histogram(&self.store, 7, now);
        let bars: Vec<Bar> = hist
            .iter()
            .enumerate()
            .map(|(i, day)| Bar::new(i as f64, day.total() as f64).name(day.date.to_string()))
            .collect();
        Plot::new("trend_7d")
            .legend(Legend::default())
            .height(160.0)
            .show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new(bars).name("events/day"));
            });

        ui.separator();
        ui.heading("Recent events");
        let start = now - chrono::Duration::days(7);
        let events = self.store.events_in_range(start, now).unwrap_or_default();
        let mut to_delete: Option<String> = None;
        egui::ScrollArea::vertical()
            .max_height(180.0)
            .show(ui, |ui| {
                for e in events.iter().take(200) {
                    ui.horizontal(|ui| {
                        if ui
                            .button("✕")
                            .on_hover_text("mark false positive (tombstone)")
                            .clicked()
                        {
                            to_delete = Some(e.uuid.clone());
                        }
                        ui.label(format!(
                            "{}  {}  conf {:.2}  x{}",
                            e.occurred_at.format("%m-%d %H:%M:%S"),
                            e.event_type.as_str(),
                            e.confidence,
                            e.burst_count
                        ));
                    });
                }
            });
        if let Some(uuid) = to_delete {
            let _ = self.store.mark_deleted(&uuid);
        }
    }

    fn draw_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("PHR connection");
        ui.horizontal(|ui| {
            ui.label("Server URL");
            if ui
                .text_edit_singleline(&mut self.form.server_url)
                .lost_focus()
            {
                let _ = self.store.setting_set("server_url", &self.form.server_url);
            }
        });
        ui.horizontal(|ui| {
            ui.label("API token");
            ui.add(egui::TextEdit::singleline(&mut self.form.token).password(true));
            if ui.button("Save token").clicked() {
                // In production this writes to the OS keychain (KeyringTokenStore);
                // never persisted to the DB or config files (SPEC §8).
                self.form.token.clear();
            }
        });

        ui.separator();
        ui.heading("Detection");
        if ui
            .add(egui::Slider::new(&mut self.form.sensitivity, 0.0..=1.0).text("sensitivity"))
            .changed()
        {
            let _ = self
                .store
                .setting_set("sensitivity", &self.form.sensitivity.to_string());
        }

        ui.separator();
        ui.heading("Mode");
        let mut mode = self.mode;
        ui.radio_value(&mut mode, Mode::AutoBatch, "Auto-batch");
        ui.radio_value(&mut mode, Mode::OfflineFirst, "Offline-first");
        ui.radio_value(
            &mut mode,
            Mode::OfflineStrict,
            "Offline-strict (never uploads)",
        );
        if mode != self.mode {
            self.set_mode(mode);
        }

        ui.separator();
        ui.heading("Teach mode");
        ui.label("Enroll your own hawk / nose-blow / snort examples (Phase B-lite).");
        ui.label("(guided capture flow — placeholder)");

        ui.separator();
        ui.label(format!("device id: {}", self.device_id));
        ui.label("Privacy: only event metadata is stored/sent — never audio.");
    }
}

impl eframe::App for SinusApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_menu_events(ctx);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(self.status_glyph());
                ui.selectable_value(&mut self.tab, Tab::History, "History");
                ui.selectable_value(&mut self.tab, Tab::Settings, "Settings");
                ui.separator();
                ui.label(format!("mode: {}", self.mode.as_str()));
                ui.separator();
                ui.label(format!("model: {}", self.shared.model().label()));
                ui.separator();
                ui.label(format!(
                    "sync: {} ({} pending)",
                    self.shared.sync().label(),
                    self.shared.pending()
                ));
                if ui
                    .button("Sync now")
                    .on_hover_text("flush pending events to the PHR now")
                    .clicked()
                {
                    self.shared.request_sync_now();
                }
                if self.shared.quiet() {
                    ui.separator();
                    ui.label("🌙 quiet hours");
                }
                if let Some(rem) = self.pause.remaining(Utc::now()) {
                    ui.separator();
                    ui.label(format!("paused {}m", rem.num_minutes() + 1));
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::History => self.draw_history(ui),
            Tab::Settings => self.draw_settings(ui),
        });

        // Event-driven repaint only (SPEC §9): request a slow tick so counts and
        // pause countdown stay fresh without a busy render loop.
        ctx.request_repaint_after(std::time::Duration::from_secs(1));
    }
}

/// Ensure a stable per-install device id exists in settings.
fn ensure_device_id(store: &Store) -> String {
    if let Ok(Some(id)) = store.setting_get("device_id") {
        return id;
    }
    let id = uuid::Uuid::new_v4().to_string();
    let _ = store.setting_set("device_id", &id);
    id
}

/// Build the tray icon + menu (SPEC §6). Not exercised in tests.
#[cfg(not(test))]
fn build_tray() -> Result<tray_icon::TrayIcon, Box<dyn std::error::Error>> {
    use tray_icon::menu::{Menu, MenuItem, PredefinedMenuItem};
    use tray_icon::TrayIconBuilder;

    let menu = Menu::new();
    let pause15 = MenuItem::with_id(ids::PAUSE_15, "Pause 15 min", true, None);
    let pause60 = MenuItem::with_id(ids::PAUSE_60, "Pause 1 hour", true, None);
    let pause_indef = MenuItem::with_id(ids::PAUSE_INDEF, "Pause until resumed", true, None);
    let resume = MenuItem::with_id(ids::RESUME, "Resume", true, None);
    let mode_auto = MenuItem::with_id(ids::MODE_AUTO, "Mode: Auto-batch", true, None);
    let mode_of = MenuItem::with_id(ids::MODE_OFFLINE_FIRST, "Mode: Offline-first", true, None);
    let mode_os = MenuItem::with_id(ids::MODE_OFFLINE_STRICT, "Mode: Offline-strict", true, None);
    let sync_now = MenuItem::with_id(ids::SYNC_NOW, "Sync now", true, None);
    let history = MenuItem::with_id(ids::OPEN_HISTORY, "Open History", true, None);
    let settings = MenuItem::with_id(ids::OPEN_SETTINGS, "Settings", true, None);
    let quit = MenuItem::with_id(ids::QUIT, "Quit", true, None);

    menu.append_items(&[
        &pause15,
        &pause60,
        &pause_indef,
        &resume,
        &PredefinedMenuItem::separator(),
        &mode_auto,
        &mode_of,
        &mode_os,
        &PredefinedMenuItem::separator(),
        &sync_now,
        &history,
        &settings,
        &PredefinedMenuItem::separator(),
        &quit,
    ])?;

    let icon = status_icon([0x2e, 0xa0, 0x43]); // listening green
    let tray = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Sinus Sentinel")
        .with_icon(icon)
        .build()?;
    Ok(tray)
}

/// A simple filled-circle status icon in the given RGB color.
#[cfg(not(test))]
fn status_icon(rgb: [u8; 3]) -> tray_icon::Icon {
    let size = 32usize;
    let r = size as f32 / 2.0;
    let mut rgba = vec![0u8; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - r;
            let dy = y as f32 + 0.5 - r;
            let inside = dx * dx + dy * dy <= (r - 1.0) * (r - 1.0);
            let i = (y * size + x) * 4;
            if inside {
                rgba[i] = rgb[0];
                rgba[i + 1] = rgb[1];
                rgba[i + 2] = rgb[2];
                rgba[i + 3] = 0xff;
            }
        }
    }
    tray_icon::Icon::from_rgba(rgba, size as u32, size as u32).expect("valid rgba icon")
}

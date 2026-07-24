//! egui app + tray integration (SPEC §6). The tray gives at-a-glance status and
//! quick actions; a single egui window hosts the History and Settings views. All
//! data logic lives in [`crate::state`] (unit-tested); this file is the renderer.
//!
//! Mic permission and interactive tray behaviour can't be verified in CI — this
//! layer is kept compiling and clippy-clean, with the testable logic factored out.

use chrono::Utc;
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Plot};
use sinus_core::store::{EnrollmentInsert, Store, StoredEnrollment};
use sinus_core::sync::Mode;
use sinus_core::types::{Event, EventType};

use crate::instance::InstanceGuard;
use crate::shared::{ModelStatus, SharedStatus, TeachState};
use crate::state::{self, PauseState};

/// Menu item ids.
#[cfg_attr(test, allow(dead_code))]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrayState {
    Listening,
    Paused,
    Warning,
    Offline,
}

impl TrayState {
    fn glyph(self) -> &'static str {
        match self {
            TrayState::Listening => "🟢",
            TrayState::Paused => "⏸",
            TrayState::Warning => "⚠",
            TrayState::Offline => "📴",
        }
    }

    #[cfg(not(test))]
    fn color(self) -> [u8; 3] {
        match self {
            TrayState::Listening => [0x2e, 0xa0, 0x43],
            TrayState::Paused => [0xf0, 0xad, 0x4e],
            TrayState::Warning => [0xd9, 0x53, 0x4f],
            TrayState::Offline => [0x77, 0x77, 0x77],
        }
    }

    #[cfg(not(test))]
    fn tooltip(self) -> &'static str {
        match self {
            TrayState::Listening => "Sinus Sentinel — listening",
            TrayState::Paused => "Sinus Sentinel — paused",
            TrayState::Warning => "Sinus Sentinel — model unavailable",
            TrayState::Offline => "Sinus Sentinel — offline-strict",
        }
    }
}

/// A pending action from the recent-events list, applied after the row loop so
/// the store is not mutated while it is being iterated.
enum HistoryAction {
    Report(Event),
    Recharacterize(Event, EventType),
    Restore(Event),
}

/// Human-readable class name.
fn label(class: EventType) -> String {
    class.as_str().replace('_', " ")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnrollmentAction {
    One {
        id: i64,
        class: EventType,
    },
    Class(EventType),
    /// Forget learned false-positive suppressions, keeping taught takes.
    Negatives,
    All,
}

/// Editable settings mirrored into the store.
#[derive(Debug, Clone, Default)]
struct SettingsForm {
    server_url: String,
    patient_id: String,
    token: String,
    token_message: String,
    sensitivity: f32,
    pause_on_low_power: bool,
    token_status: String,
    enrollment_message: String,
}

#[derive(Default)]
struct HistorySnapshot {
    generation: usize,
    day: Option<chrono::NaiveDate>,
    refreshed_at: Option<std::time::Instant>,
    today: std::collections::HashMap<EventType, i64>,
    histogram: Vec<state::DayCount>,
    recent: Vec<Event>,
}

pub struct SinusApp {
    store: Store,
    pause: PauseState,
    mode: Mode,
    tab: Tab,
    form: SettingsForm,
    // The tray icon is held for its lifetime; menu events are polled globally.
    #[cfg(not(test))]
    tray: Option<tray_icon::TrayIcon>,
    #[cfg(not(test))]
    tray_state: TrayState,
    device_id: String,
    shared: SharedStatus,
    instance: InstanceGuard,
    pending_enrollment_action: Option<EnrollmentAction>,
    history_message: String,
    history: HistorySnapshot,
    window_visible: bool,
    #[cfg(not(test))]
    menu_events: std::sync::mpsc::Receiver<tray_icon::menu::MenuEvent>,
}

impl SinusApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        store: Store,
        shared: SharedStatus,
        instance: InstanceGuard,
    ) -> Self {
        shared.attach_repaint_context(cc.egui_ctx.clone());
        #[cfg(not(test))]
        let menu_events = {
            let (sender, receiver) = std::sync::mpsc::channel();
            let context = cc.egui_ctx.clone();
            tray_icon::menu::MenuEvent::set_event_handler(Some(move |event| {
                let _ = sender.send(event);
                context.request_repaint();
            }));
            receiver
        };
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
        let pause_on_low_power = store
            .setting_get("pause_low_power")
            .ok()
            .flatten()
            .is_none_or(|value| value != "false");
        let patient_id = store
            .setting_get("patient_id")
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
                patient_id,
                token: String::new(),
                token_message: String::new(),
                sensitivity,
                pause_on_low_power,
                token_status: "Token status not checked.".to_string(),
                enrollment_message: String::new(),
            },
            #[cfg(not(test))]
            tray: build_tray().ok(),
            #[cfg(not(test))]
            tray_state: TrayState::Listening,
            device_id,
            shared,
            instance,
            pending_enrollment_action: None,
            history_message: String::new(),
            history: HistorySnapshot::default(),
            window_visible: false,
            #[cfg(not(test))]
            menu_events,
        }
    }

    fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        let _ = self.store.setting_set("mode", mode.as_str());
        self.shared.notify_sync();
    }

    fn show_window(&mut self, ctx: &egui::Context) {
        self.window_visible = true;
        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
        ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
    }

    fn apply_enrollment_action(&mut self, action: EnrollmentAction) {
        let result = match action {
            EnrollmentAction::One { id, .. } => self.store.delete_enrollment(id).map(|()| 1usize),
            EnrollmentAction::Class(class) => self.store.delete_enrollments_for_class(class),
            EnrollmentAction::Negatives => self.store.delete_negative_enrollments(),
            EnrollmentAction::All => self.store.delete_all_enrollments(),
        };

        match result {
            Ok(deleted) => {
                let noun = match action {
                    EnrollmentAction::Negatives => "false-positive report",
                    _ => "saved take",
                };
                self.form.enrollment_message = format!(
                    "Removed {deleted} {noun}{}. Detection updated immediately.",
                    if deleted == 1 { "" } else { "s" }
                );
                self.shared.reset_teach_feedback();
                self.shared.request_enrollment_reload();
                // The removal has to reach the PHR too, or another machine
                // re-teaches this device on its next pull.
                self.shared.request_sync_now();
            }
            Err(error) => {
                self.form.enrollment_message = format!("Could not update training: {error}");
            }
        }
    }

    /// Report a misdetection: enroll the event's stored embedding as a negative
    /// for the class that fired (so the detector stops calling that sound *that*),
    /// then flag the event.
    ///
    /// The event is flagged, not deleted: a health record should keep the fact
    /// that the classifier got something wrong. It stops counting everywhere and
    /// the flag syncs to the PHR, where it is likewise retained rather than
    /// erased. The embedding is deliberately kept — the user may follow up by
    /// saying what the sound actually was.
    fn report_false_positive(&mut self, event: &Event) {
        let trained = self.enroll_negative(event, event.event_type, false);

        if let Err(error) = self.store.mark_false_positive(&event.uuid) {
            self.history_message = format!("Could not flag the event: {error}");
            return;
        }
        if trained {
            self.shared.request_enrollment_reload();
        }
        self.shared.notify_history_changed();
        self.shared.request_sync_now();

        let class = label(event.event_type);
        self.history_message = if trained {
            format!(
                "Reported the {class}: it no longer counts here or in the PHR, and the \
                 detector will stop labelling that sound {class}."
            )
        } else {
            format!(
                "Reported the {class}: it no longer counts here or in the PHR. No embedding \
                 was stored for it, so the detector was not adjusted."
            )
        };
    }

    /// Record what a misdetected sound actually was.
    ///
    /// Enrolls the embedding twice: as a negative against the class that fired,
    /// and as a positive for the corrected one. Negatives are class-scoped, so
    /// the wrong label is suppressed while the corrected label stays free to
    /// fire — a class-blind negative would silence the sound entirely, which is
    /// worse than the original mistake.
    fn recharacterize(&mut self, event: &Event, corrected: EventType) {
        // Correcting back to what the classifier originally said is an undo, not
        // a correction. Treating it as one would enroll a negative *and* a
        // positive for that class from the same embedding, and the veto would
        // then suppress the very label the user just confirmed.
        if corrected == event.event_type {
            self.clear_flag(event);
            return;
        }

        let embedding = self.store.get_event_embedding(&event.uuid).ok().flatten();

        let mut trained = false;
        if let Some(embedding) = &embedding {
            trained = self.enroll_negative(event, event.event_type, true);
            let positive = self.store.add_enrollment_full(EnrollmentInsert {
                class: corrected,
                embedding,
                is_negative: false,
                similarity: None,
                separation: None,
                peak_dbfs: event.peak_dbfs,
                model_version: Some(&event.model_version),
                source_event_uuid: Some(&event.uuid),
                negative_scoped: false,
            });
            trained |= positive.is_ok();
        }

        if let Err(error) = self.store.recharacterize(&event.uuid, corrected) {
            self.history_message = format!("Could not update the event: {error}");
            return;
        }
        if trained {
            self.shared.request_enrollment_reload();
        }
        self.shared.notify_history_changed();
        self.shared.request_sync_now();

        let was = label(event.event_type);
        let now = label(corrected);
        // Be honest about what one correction can do: the negative takes effect
        // immediately, but a personalized class needs three takes before it
        // matches on its own.
        self.history_message = format!(
            "Recorded as {now} instead of {was} — it now counts as {now} here and in the \
             PHR, and the detector will stop calling that sound {was}. Teach {now} a few \
             more times for it to be recognised on its own."
        );
    }

    /// Undo a false-positive report or a correction.
    fn clear_flag(&mut self, event: &Event) {
        if let Err(error) = self.store.clear_flag(&event.uuid) {
            self.history_message = format!("Could not restore the event: {error}");
            return;
        }
        self.shared.notify_history_changed();
        self.shared.request_sync_now();
        self.history_message =
            "Restored the event here and in the PHR. Any training it produced is kept — \
             use Settings › Teach mode to remove that too."
                .to_string();
    }

    /// Enroll this event's sound as "not `class`", if its embedding was retained.
    ///
    /// `scoped` decides how far the veto reaches: a plain false-positive report
    /// is unscoped (the sound is suppressed under every label, so a borderline
    /// sound cannot simply re-fire as a sibling class), while the negative half
    /// of a correction is scoped, because a positive for the corrected class
    /// carries the same embedding.
    fn enroll_negative(&mut self, event: &Event, class: EventType, scoped: bool) -> bool {
        let Ok(Some(embedding)) = self.store.get_event_embedding(&event.uuid) else {
            return false;
        };
        self.store
            .add_enrollment_full(EnrollmentInsert {
                class,
                embedding: &embedding,
                is_negative: true,
                similarity: None,
                separation: None,
                peak_dbfs: event.peak_dbfs,
                model_version: Some(&event.model_version),
                source_event_uuid: Some(&event.uuid),
                negative_scoped: scoped,
            })
            .is_ok()
    }

    #[cfg(not(test))]
    fn handle_menu_events(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.menu_events.try_recv() {
            let id = event.id.as_ref();
            let now = Utc::now();
            match id {
                ids::PAUSE_15 => {
                    let until = now + chrono::Duration::minutes(15);
                    self.pause = PauseState::PausedUntil(until);
                    self.shared.pause_until(until);
                }
                ids::PAUSE_60 => {
                    let until = now + chrono::Duration::minutes(60);
                    self.pause = PauseState::PausedUntil(until);
                    self.shared.pause_until(until);
                }
                ids::PAUSE_INDEF => {
                    self.pause = PauseState::PausedIndefinite;
                    self.shared.pause_indefinitely();
                }
                ids::RESUME => {
                    self.pause = PauseState::Running;
                    self.shared.resume_capture();
                }
                ids::MODE_AUTO => self.set_mode(Mode::AutoBatch),
                ids::MODE_OFFLINE_FIRST => self.set_mode(Mode::OfflineFirst),
                ids::MODE_OFFLINE_STRICT => self.set_mode(Mode::OfflineStrict),
                ids::SYNC_NOW => self.shared.request_sync_now(),
                ids::OPEN_HISTORY => {
                    self.tab = Tab::History;
                    self.show_window(ctx);
                }
                ids::OPEN_SETTINGS => {
                    self.tab = Tab::Settings;
                    self.show_window(ctx);
                }
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

    #[cfg(test)]
    fn handle_menu_events(&mut self, _ctx: &egui::Context) {}

    fn current_tray_state(&mut self) -> TrayState {
        let now = Utc::now();
        self.pause = self.pause.normalized(now);
        // A missing model (fail-soft fallback in the capture thread) is a warning
        // state (SPEC §6 tray "⚠"), shown ahead of the plain listening glyph.
        match self.mode {
            Mode::OfflineStrict => TrayState::Offline,
            _ if self.pause.is_paused(now) || self.shared.low_power() || self.shared.quiet() => {
                TrayState::Paused
            }
            _ if self.shared.model() == ModelStatus::Missing => TrayState::Warning,
            _ => TrayState::Listening,
        }
    }

    fn status_glyph(&mut self) -> &'static str {
        self.current_tray_state().glyph()
    }

    #[cfg(not(test))]
    fn update_tray_status(&mut self) {
        let state = self.current_tray_state();
        if state == self.tray_state {
            return;
        }
        if let Some(tray) = &self.tray {
            let _ = tray.set_icon(Some(status_icon(state.color())));
            let _ = tray.set_tooltip(Some(state.tooltip()));
        }
        self.tray_state = state;
    }

    #[cfg(test)]
    fn update_tray_status(&mut self) {}

    fn refresh_history_if_needed(&mut self, now: chrono::DateTime<Utc>) {
        let generation = self.shared.history_generation();
        let day = now.date_naive();
        let stale = self
            .history
            .refreshed_at
            .is_none_or(|at| at.elapsed() >= std::time::Duration::from_secs(60));
        if self.history.generation == generation && self.history.day == Some(day) && !stale {
            return;
        }

        self.history.today = state::today_counts(&self.store, now);
        self.history.histogram = state::daily_histogram(&self.store, 7, now);
        self.history.recent = self
            .store
            .recent_events(now - chrono::Duration::days(7), now)
            .unwrap_or_default();
        self.history.generation = generation;
        self.history.day = Some(day);
        self.history.refreshed_at = Some(std::time::Instant::now());
    }

    fn draw_history(&mut self, ui: &mut egui::Ui) {
        let now = Utc::now();
        self.refresh_history_if_needed(now);
        let today = &self.history.today;

        ui.heading("Today");
        ui.horizontal_wrapped(|ui| {
            let dark = ui.visuals().dark_mode;
            for et in EventType::ALL {
                let n = today.get(&et).copied().unwrap_or(0);
                if n > 0 {
                    // The colored square ties the count to its series in the
                    // chart below; the text itself stays in normal ink.
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("■").color(class_color(et, dark)));
                        ui.label(format!("{}: {n}", et.as_str().replace('_', " ")));
                    });
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
            state::congestion_score(today, monitored_hours.max(0.1))
        ));

        ui.separator();
        ui.heading("Last 7 days");
        let hist = &self.history.histogram;
        let dark = ui.visuals().dark_mode;
        let surface = ui.visuals().panel_fill;

        // One stacked series per class that occurred this week, in fixed class
        // order with class-bound colors — a week without sneezes must not
        // recolor the remaining series.
        let mut charts: Vec<BarChart> = Vec::new();
        for class in EventType::ALL {
            let counts: Vec<i64> = hist
                .iter()
                .map(|day| day.counts.get(&class).copied().unwrap_or(0))
                .collect();
            if counts.iter().all(|&count| count == 0) {
                continue;
            }
            let color = class_color(class, dark);
            let bars: Vec<Bar> = counts
                .iter()
                .enumerate()
                .map(|(i, &count)| {
                    let bar = Bar::new(i as f64, count as f64).width(0.72).fill(color);
                    // A surface-colored hairline separates stacked segments;
                    // zero-height bars get none so empty days stay blank.
                    if count > 0 {
                        bar.stroke(egui::Stroke::new(1.5_f32, surface))
                    } else {
                        bar
                    }
                })
                .collect();
            let label = class.as_str().replace('_', " ");
            let dates: Vec<String> = hist
                .iter()
                .map(|day| day.date.format("%b %-d").to_string())
                .collect();
            let mut chart = BarChart::new(bars)
                .name(&label)
                .color(color)
                .element_formatter(Box::new(move |bar, _| {
                    let date = dates
                        .get(bar.argument.round() as usize)
                        .cloned()
                        .unwrap_or_default();
                    format!("{label}: {} — {date}", bar.value.round() as i64)
                }));
            let stacked_below: Vec<&BarChart> = charts.iter().collect();
            chart = chart.stack_on(&stacked_below);
            charts.push(chart);
        }

        if charts.is_empty() {
            ui.label("no events in the last 7 days");
        } else {
            let day_labels: Vec<String> = hist
                .iter()
                .map(|day| day.date.format("%a").to_string())
                .collect();
            Plot::new("trend_7d")
                .legend(Legend::default())
                .height(180.0)
                // A fixed 7-day window has nothing to pan or zoom; leaving the
                // defaults on makes the chart eat window-scroll gestures.
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .allow_double_click_reset(false)
                .include_y(0.0)
                .x_axis_formatter(move |mark, _range| {
                    let index = mark.value;
                    if index.fract().abs() < f64::EPSILON && index >= 0.0 {
                        day_labels.get(index as usize).cloned().unwrap_or_default()
                    } else {
                        String::new()
                    }
                })
                // Event counts are integers; suppress fractional gridline labels.
                .y_axis_formatter(|mark, _range| {
                    if mark.value.fract().abs() < f64::EPSILON && mark.value >= 0.0 {
                        format!("{}", mark.value as i64)
                    } else {
                        String::new()
                    }
                })
                .show(ui, |plot_ui| {
                    for chart in charts {
                        plot_ui.bar_chart(chart);
                    }
                });
        }

        ui.separator();
        ui.heading("Recent events");

        // The gate can be open for a second or two before a classification
        // lands. Say so, rather than looking idle while the app is working.
        if self.shared.analyzing() {
            let heard = self
                .shared
                .last_heard()
                .map(|t| {
                    t.with_timezone(&chrono::Local)
                        .format("%H:%M:%S")
                        .to_string()
                })
                .unwrap_or_else(|| "now".to_string());
            ui.colored_label(
                egui::Color32::LIGHT_BLUE,
                format!("🔊 heard something at {heard} — classifying…"),
            );
        }

        // Flagged events stay in this list (struck through, with an undo) even
        // though they are excluded from every count above.
        let events = &self.history.recent;
        let mut action: Option<HistoryAction> = None;
        egui::ScrollArea::vertical()
            .max_height(220.0)
            .show(ui, |ui| {
                for e in events.iter().take(200) {
                    let flagged = e.false_positive_at.is_some();
                    let corrected = e.corrected_to.is_some();
                    ui.horizontal(|ui| {
                        // Both a report and a correction are undoable — a
                        // correction is just as much a user judgement that can
                        // be wrong.
                        if flagged || corrected {
                            let hint = if flagged {
                                "Undo: count this event again, here and in the PHR"
                            } else {
                                "Undo the correction, here and in the PHR"
                            };
                            if ui.button("↺").on_hover_text(hint).clicked() {
                                action = Some(HistoryAction::Restore(e.clone()));
                            }
                        }
                        if !flagged
                            && ui
                                .button("✕")
                                .on_hover_text(
                                    "Report false positive: stops this counting (here and in \
                                 the PHR) and teaches the detector not to label that sound \
                                 this way",
                                )
                                .clicked()
                        {
                            action = Some(HistoryAction::Report(e.clone()));
                        }

                        let shown = e.effective_type();
                        let mut text = format!(
                            "{}  {}",
                            e.occurred_at.format("%m-%d %H:%M:%S"),
                            label(shown),
                        );
                        if let Some(original) = e.corrected_to.map(|_| e.event_type) {
                            text.push_str(&format!(" (was {})", label(original)));
                        }
                        text.push_str(&format!("  conf {:.2}  x{}", e.confidence, e.burst_count));
                        if let Some(peak) = e.peak_dbfs {
                            text.push_str(&format!("  {peak:.0} dBFS"));
                        }

                        let mut rich = egui::RichText::new(text);
                        if flagged {
                            rich = rich.strikethrough().weak();
                        }
                        ui.label(rich);

                        // Recharacterize: say what the sound actually was.
                        if !flagged {
                            egui::ComboBox::from_id_salt(("recharacterize", &e.uuid))
                                .selected_text("…")
                                .width(30.0)
                                .show_ui(ui, |ui| {
                                    ui.label("Actually this was:");
                                    for class in EventType::ALL {
                                        if class == shown {
                                            continue;
                                        }
                                        if ui.selectable_label(false, label(class)).clicked() {
                                            action = Some(HistoryAction::Recharacterize(
                                                e.clone(),
                                                class,
                                            ));
                                        }
                                    }
                                })
                                .response
                                .on_hover_text("Recharacterize: record what this sound really was");
                        }
                    });
                }
            });

        match action {
            Some(HistoryAction::Report(event)) => self.report_false_positive(&event),
            Some(HistoryAction::Recharacterize(event, class)) => self.recharacterize(&event, class),
            Some(HistoryAction::Restore(event)) => self.clear_flag(&event),
            None => {}
        }
        if !self.history_message.is_empty() {
            ui.label(&self.history_message);
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
                self.shared.notify_sync();
            }
        });
        ui.horizontal(|ui| {
            ui.label("Patient id");
            if ui
                .add(egui::TextEdit::singleline(&mut self.form.patient_id).desired_width(80.0))
                .on_hover_text("The PHR patient these events belong to")
                .lost_focus()
            {
                let trimmed = self.form.patient_id.trim().to_string();
                if trimmed.is_empty() || trimmed.parse::<i64>().is_ok_and(|id| id > 0) {
                    self.form.patient_id = trimmed;
                    let _ = self.store.setting_set("patient_id", &self.form.patient_id);
                    self.shared.notify_sync();
                    self.form.token_message.clear();
                } else {
                    self.form.token_message =
                        "Patient id must be a number — nothing will sync until it is.".to_string();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("API token");
            ui.add(egui::TextEdit::singleline(&mut self.form.token).password(true));
            if ui.button("Save token").clicked() {
                self.form.token_message = match save_api_token(self.form.token.trim()) {
                    Ok(()) => {
                        self.form.token_status = "Token stored in the OS keychain.".to_string();
                        "Saved in the OS keychain.".to_string()
                    }
                    Err(error) => format!("Could not save token: {error}"),
                };
                self.form.token.clear();
            }
            if ui
                .button("Check token")
                .on_hover_text("Checks only whether a token exists; never displays it")
                .clicked()
            {
                self.form.token_status = match api_token_status() {
                    Ok(true) => "Token stored in the OS keychain.".to_string(),
                    Ok(false) => "No API token is stored.".to_string(),
                    Err(error) => format!("Could not check token: {error}"),
                };
            }
        });
        ui.label(&self.form.token_status);
        if !self.form.token_message.is_empty() {
            ui.label(&self.form.token_message);
        }

        ui.separator();
        ui.heading("Detection");
        let slider = ui
            .add(egui::Slider::new(&mut self.form.sensitivity, 0.0..=1.0).text("sensitivity"))
            .on_hover_text("Shared with your other machines through the PHR");
        if slider.changed() {
            let _ = self
                .store
                .setting_set("sensitivity", &self.form.sensitivity.to_string());
            // Apply it to the running detector: without this the slider would
            // only take effect on the next launch.
            self.shared.request_settings_reload();
        }
        if slider.drag_stopped() || slider.lost_focus() {
            // Only once the drag settles — syncing on every tick would push a
            // document per frame.
            self.shared.request_sync_now();
        }
        if ui
            .checkbox(
                &mut self.form.pause_on_low_power,
                "Pause microphone while OS low-power mode is active",
            )
            .on_hover_text("Releases the microphone and resumes automatically")
            .changed()
        {
            let _ = self.store.setting_set(
                "pause_low_power",
                if self.form.pause_on_low_power {
                    "true"
                } else {
                    "false"
                },
            );
            self.shared.request_settings_reload();
        }
        ui.label(
            "Sensitivity and quiet hours sync with the PHR, so they follow you between \
             machines. Server URL, patient id and sync mode stay on this device.",
        );

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
        ui.label("Teach your own sounds. Raw audio is discarded; only an embedding is saved — and synced to the PHR, if connected, so other machines inherit your training.");
        ui.label("Record one clear sound after the short get-ready countdown. Add 3–5 varied takes per class.");

        let feedback = self.shared.teach_feedback();
        let busy = matches!(feedback.state, TeachState::Armed | TeachState::Recording);
        let model_ready = self.shared.model() == ModelStatus::Onnx;
        let enrollments = self.store.enrollments().unwrap_or_default();

        egui::ScrollArea::vertical()
            .id_salt("teach_classes")
            .max_height(360.0)
            .show(ui, |ui| {
                for class in EventType::ALL {
                    let class_examples: Vec<_> = enrollments
                        .iter()
                        .filter(|stored| {
                            stored.enrollment.class == class && !stored.enrollment.is_negative
                        })
                        .collect();
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.strong(class.as_str().replace('_', " "));
                            ui.label(enrollment_status(&class_examples));
                            if ui
                                .add_enabled(model_ready && !busy, egui::Button::new("Record take"))
                                .clicked()
                            {
                                self.shared.request_teach(class);
                            }
                            if ui
                                .add_enabled(
                                    !busy && !class_examples.is_empty(),
                                    egui::Button::new("Reset class"),
                                )
                                .clicked()
                            {
                                self.pending_enrollment_action =
                                    Some(EnrollmentAction::Class(class));
                            }
                        });

                        for (index, stored) in class_examples.iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(format!(
                                    "Take {} • {}",
                                    index + 1,
                                    format_enrollment_time(&stored.created_at)
                                ));
                                match (stored.similarity, stored.separation) {
                                    (Some(similarity), Some(separation)) => {
                                        ui.label(format!(
                                            "repeat {similarity:.2} • separation {separation:+.2}"
                                        ));
                                    }
                                    _ => {
                                        ui.label("baseline take");
                                    }
                                }
                                if ui.add_enabled(!busy, egui::Button::new("Remove")).clicked() {
                                    self.pending_enrollment_action = Some(EnrollmentAction::One {
                                        id: stored.id,
                                        class,
                                    });
                                }
                            });
                        }
                    });
                }
            });

        let negative_count = enrollments
            .iter()
            .filter(|stored| stored.enrollment.is_negative)
            .count();
        if negative_count > 0 {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "{negative_count} reported false positive{} teaching the detector what to ignore.",
                    if negative_count == 1 { " is" } else { "s are" }
                ));
                if ui
                    .add_enabled(!busy, egui::Button::new("Forget reports"))
                    .clicked()
                {
                    self.pending_enrollment_action = Some(EnrollmentAction::Negatives);
                }
            });
        }

        if ui
            .add_enabled(
                !busy && !enrollments.is_empty(),
                egui::Button::new("Reset all training"),
            )
            .clicked()
        {
            self.pending_enrollment_action = Some(EnrollmentAction::All);
        }

        if let Some(action) = self.pending_enrollment_action {
            let mut confirm = false;
            let mut cancel = false;
            ui.group(|ui| {
                ui.colored_label(egui::Color32::YELLOW, enrollment_confirmation(action));
                ui.label("Only local embeddings and their metadata will be removed; event history is unchanged.");
                ui.horizontal(|ui| {
                    confirm = ui.button("Confirm removal").clicked();
                    cancel = ui.button("Cancel").clicked();
                });
            });
            if confirm {
                self.apply_enrollment_action(action);
                self.pending_enrollment_action = None;
            } else if cancel {
                self.pending_enrollment_action = None;
            }
        }

        if !self.form.enrollment_message.is_empty() {
            ui.label(&self.form.enrollment_message);
        }

        if !model_ready {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Teach mode needs the YAMNet model; restart after fixing a ‘model missing’ status.",
            );
        }
        match (feedback.state, feedback.class) {
            (TeachState::Armed, Some(class)) => {
                ui.label(format!(
                    "Get ready — {} recording starts in about one second…",
                    class.as_str()
                ));
            }
            (TeachState::Recording, Some(class)) => {
                ui.colored_label(
                    egui::Color32::LIGHT_GREEN,
                    format!("Recording {} now — make the sound once.", class.as_str()),
                );
            }
            (TeachState::Saved, Some(class)) if feedback.similarity < 0.0 => {
                ui.label(format!(
                    "Saved the first {} sample. Add at least two more for validation.",
                    class.as_str()
                ));
            }
            (TeachState::Saved, Some(class)) => {
                let good = feedback.examples >= 3
                    && feedback.similarity >= 0.75
                    && feedback.separation >= 0.05;
                let verdict = if good {
                    "good"
                } else {
                    "keep adding varied samples"
                };
                ui.label(format!(
                    "Saved {} sample #{} — repeat similarity {:.2}, class separation {:+.2}: {}.",
                    class.as_str(),
                    feedback.examples,
                    feedback.similarity,
                    feedback.separation,
                    verdict
                ));
            }
            (TeachState::Failed, Some(class)) => {
                ui.colored_label(
                    egui::Color32::RED,
                    format!(
                        "Could not save the {} sample; try again in a quieter moment.",
                        class.as_str()
                    ),
                );
            }
            _ => {}
        }

        ui.separator();
        ui.label(
            "Menu-bar status: 🟢 listening • ⏸ paused • ⚠ model unavailable • 📴 offline-strict.",
        );
        ui.label("macOS: Sinus Sentinel stays in the menu bar without a Dock icon. Closing this window hides it; use the menu-bar icon to reopen or quit.");
        ui.label(format!("device id: {}", self.device_id));
        ui.label(
            "Privacy: never audio. Event metadata is stored and synced; Teach-mode \
             embeddings sync too when a PHR is connected, so your training follows you \
             between machines. Embeddings are opaque vectors — audio cannot be \
             reconstructed from them.",
        );
    }
}

/// Fixed per-class chart colors. Bound to the class, never the visible-series
/// index, so a week without some class never repaints the others. Slots follow
/// the classes' declaration order; both mode variants were validated (CVD
/// adjacent-pair separation, normal-vision floor, chroma/lightness bands)
/// against the actual egui panel surfaces — light #f8f8f8, dark #1b1b1b.
fn class_color(class: EventType, dark: bool) -> egui::Color32 {
    match (class, dark) {
        (EventType::Cough, false) => egui::Color32::from_rgb(0x2a, 0x78, 0xd6),
        (EventType::Cough, true) => egui::Color32::from_rgb(0x39, 0x87, 0xe5),
        (EventType::ThroatClearing, _) => egui::Color32::from_rgb(0x00, 0x83, 0x00),
        (EventType::Sniffle, false) => egui::Color32::from_rgb(0xe8, 0x7b, 0xa4),
        (EventType::Sniffle, true) => egui::Color32::from_rgb(0xd5, 0x51, 0x81),
        (EventType::Sneeze, false) => egui::Color32::from_rgb(0xed, 0xa1, 0x00),
        (EventType::Sneeze, true) => egui::Color32::from_rgb(0xc9, 0x85, 0x00),
        (EventType::NoseBlow, false) => egui::Color32::from_rgb(0x1b, 0xaf, 0x7a),
        (EventType::NoseBlow, true) => egui::Color32::from_rgb(0x19, 0x9e, 0x70),
        (EventType::Hawk, false) => egui::Color32::from_rgb(0xeb, 0x68, 0x34),
        (EventType::Hawk, true) => egui::Color32::from_rgb(0xd9, 0x59, 0x26),
        (EventType::SnortSuck, false) => egui::Color32::from_rgb(0x4a, 0x3a, 0xa7),
        (EventType::SnortSuck, true) => egui::Color32::from_rgb(0x90, 0x85, 0xe9),
    }
}

fn enrollment_status(examples: &[&StoredEnrollment]) -> String {
    let count = examples.len();
    match count {
        0 => "not trained".to_string(),
        1 | 2 => format!("inactive • needs {} more", 3 - count),
        _ => {
            let quality_is_good = examples.last().is_some_and(|latest| {
                latest.similarity.is_some_and(|value| value >= 0.75)
                    && latest.separation.is_some_and(|value| value >= 0.05)
            });
            if quality_is_good {
                format!("ready • {count} takes")
            } else {
                format!("active • {count} takes • add varied takes")
            }
        }
    }
}

fn format_enrollment_time(created_at: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(created_at)
        .map(|time| {
            time.with_timezone(&chrono::Local)
                .format("%b %-d, %-I:%M %p")
                .to_string()
        })
        .unwrap_or_else(|_| "saved previously".to_string())
}

fn enrollment_confirmation(action: EnrollmentAction) -> String {
    match action {
        EnrollmentAction::One { class, .. } => {
            format!("Remove this {} take?", class.as_str().replace('_', " "))
        }
        EnrollmentAction::Class(class) => {
            format!("Reset every {} take?", class.as_str().replace('_', " "))
        }
        EnrollmentAction::Negatives => {
            "Forget every reported false positive? Previously suppressed sounds may be \
             detected again."
                .to_string()
        }
        EnrollmentAction::All => "Reset every saved Teach-mode take?".to_string(),
    }
}

#[cfg(feature = "keyring")]
fn save_api_token(token: &str) -> Result<(), String> {
    use sinus_core::token::{KeyringTokenStore, TokenStore};

    if token.is_empty() {
        return Err("token is empty".to_string());
    }
    KeyringTokenStore::new("SinusSentinel", "phr-api-token")
        .set_token(token)
        .map_err(|error| error.to_string())
}

#[cfg(feature = "keyring")]
fn api_token_status() -> Result<bool, String> {
    use sinus_core::token::{KeyringTokenStore, TokenStore};

    KeyringTokenStore::new("SinusSentinel", "phr-api-token")
        .get_token()
        .map(|token| token.is_some())
        .map_err(|error| error.to_string())
}

#[cfg(not(feature = "keyring"))]
fn save_api_token(_token: &str) -> Result<(), String> {
    Err("this build has no OS keychain support".to_string())
}

#[cfg(not(feature = "keyring"))]
fn api_token_status() -> Result<bool, String> {
    Err("this build has no OS keychain support".to_string())
}

impl eframe::App for SinusApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_menu_events(ctx);

        if self.instance.take_activation_request() {
            self.tab = Tab::History;
            self.show_window(ctx);
        }
        self.update_tray_status();

        // Closing the accessory window hides it but leaves monitoring and the
        // menu-bar item alive. Only the explicit Quit menu action exits.
        if ctx.input(|input| input.viewport().close_requested()) && !self.shared.quitting() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            self.window_visible = false;
        }

        // The tray and workers are event-driven. While hidden, avoid constructing
        // panels/charts or touching SQLite; this slow tick only preserves the
        // file-based second-instance activation fallback.
        if !self.window_visible {
            ctx.request_repaint_after(std::time::Duration::from_secs(5));
            return;
        }

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
                if self.shared.low_power() {
                    ui.separator();
                    ui.label("🔋 low-power pause");
                }
                if let Some(rem) = self.pause.remaining(Utc::now()) {
                    ui.separator();
                    ui.label(format!("paused {}m", rem.num_minutes() + 1));
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::History => self.draw_history(ui),
            Tab::Settings => {
                egui::ScrollArea::vertical()
                    .id_salt("settings_page")
                    .show(ui, |ui| self.draw_settings(ui));
            }
        });

        // Data changes request repaint through SharedStatus. A minute tick keeps
        // the monitored-hours label/date boundary fresh without a render loop;
        // timed pause expiry gets one precise wake at its deadline.
        let mut next = std::time::Duration::from_secs(60);
        if let Some(remaining) = self.pause.remaining(Utc::now()) {
            if let Ok(remaining) = remaining.to_std() {
                next = next.min(remaining);
            }
        }
        ctx.request_repaint_after(next.max(std::time::Duration::from_millis(1)));
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

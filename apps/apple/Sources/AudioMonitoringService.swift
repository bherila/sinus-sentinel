import AVFoundation
import Foundation

/// Native audio-session owner. Audio never persists here: each callback is
/// converted to 16 kHz mono PCM, sent to Rust, and then released.
final class AudioMonitoringService: @unchecked Sendable {
    var onEvents: (@Sendable ([AppleEvent]) -> Void)?
    var onFailure: (@Sendable (String) -> Void)?

    private let detector: AppleEngine
    private let audioEngine = AVAudioEngine()
    private let processingQueue = DispatchQueue(
        label: "com.sinussentinel.audio-processing",
        qos: .userInitiated
    )
    private var converter: AVAudioConverter?
    private var running = false

    init(engine: AppleEngine) {
        detector = engine
    }

    static func requestPermission() async -> Bool {
        #if os(iOS)
        if #available(iOS 17.0, *) {
            return await AVAudioApplication.requestRecordPermission()
        }
        return await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission {
                continuation.resume(returning: $0)
            }
        }
        #else
        return await AVCaptureDevice.requestAccess(for: .audio)
        #endif
    }

    func start() throws {
        guard !running else { return }

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setPreferredIOBufferDuration(0.05)
        try session.setActive(true)
        #endif

        let input = audioEngine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ), let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioMonitoringError.unsupportedInputFormat
        }
        self.converter = converter

        let now = Int64(Date().timeIntervalSince1970 * 1_000)
        let offset = Int32(TimeZone.current.secondsFromGMT() / 60)
        try detector.startMonitoring(
            startedAtEpochMs: now,
            timezoneOffsetMinutes: offset
        )

        input.installTap(
            onBus: 0,
            bufferSize: AVAudioFrameCount(max(1, inputFormat.sampleRate / 20)),
            format: inputFormat
        ) { [weak self] buffer, _ in
            guard let self, let copy = Self.copy(buffer) else { return }
            self.processingQueue.async {
                self.process(copy, outputFormat: outputFormat)
            }
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            running = true
        } catch {
            input.removeTap(onBus: 0)
            _ = try? detector.stopMonitoring()
            throw error
        }
    }

    func stop() throws -> [AppleEvent] {
        guard running else { return [] }
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        processingQueue.sync {}
        converter = nil
        running = false

        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
        #endif

        return try detector.stopMonitoring()
    }

    private func process(_ input: AVAudioPCMBuffer, outputFormat: AVAudioFormat) {
        guard let converter else { return }
        let ratio = outputFormat.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount(
            ceil(Double(input.frameLength) * ratio) + 32
        )
        guard let output = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: capacity
        ) else { return }

        var supplied = false
        var conversionError: NSError?
        let status = converter.convert(
            to: output,
            error: &conversionError
        ) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return input
        }
        guard status != .error, conversionError == nil,
              let channel = output.floatChannelData?.pointee else {
            onFailure?(conversionError?.localizedDescription ?? "Audio conversion failed.")
            return
        }

        let samples = Array(
            UnsafeBufferPointer(start: channel, count: Int(output.frameLength))
        )
        do {
            let events = try detector.pushPcm16k(samples: samples)
            if !events.isEmpty {
                onEvents?(events)
            }
        } catch {
            onFailure?(error.localizedDescription)
        }
    }

    private static func copy(_ source: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let destination = AVAudioPCMBuffer(
            pcmFormat: source.format,
            frameCapacity: source.frameLength
        ) else { return nil }
        destination.frameLength = source.frameLength

        let sourceBuffers = UnsafeMutableAudioBufferListPointer(
            source.mutableAudioBufferList
        )
        let destinationBuffers = UnsafeMutableAudioBufferListPointer(
            destination.mutableAudioBufferList
        )
        for index in 0..<min(sourceBuffers.count, destinationBuffers.count) {
            guard let sourceData = sourceBuffers[index].mData,
                  let destinationData = destinationBuffers[index].mData else {
                continue
            }
            memcpy(
                destinationData,
                sourceData,
                Int(sourceBuffers[index].mDataByteSize)
            )
        }
        return destination
    }
}

enum AudioMonitoringError: LocalizedError {
    case unsupportedInputFormat

    var errorDescription: String? {
        "The current microphone format could not be converted to 16 kHz mono."
    }
}

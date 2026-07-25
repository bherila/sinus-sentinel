import CoreML
import Foundation

/// Expected Core ML contract:
/// input `input` [1, 96, 64], outputs `scores` [1, 521] and
/// `embeddings` [1, 1024]. This matches `model/CONVERSION.md`.
final class CoreMLYamnetRunner: ModelRunner, @unchecked Sendable {
    private let model: MLModel

    init(compiledModelURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
    }

    func modelVersion() throws -> String {
        "yamnet-coreml@prototype"
    }

    func infer(
        logMel: [Float],
        frames: UInt32,
        bands: UInt32,
        energyPeak: Bool
    ) throws -> ModelOutput {
        guard logMel.count == Int(frames * bands) else {
            throw ModelError.Failed(message: "Core ML patch shape does not match its data.")
        }
        let input = try MLMultiArray(
            shape: [1, NSNumber(value: frames), NSNumber(value: bands)],
            dataType: .float32
        )
        let pointer = input.dataPointer.bindMemory(to: Float.self, capacity: logMel.count)
        logMel.withUnsafeBufferPointer { source in
            pointer.update(from: source.baseAddress!, count: logMel.count)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": input])
        let prediction = try model.prediction(from: provider)
        guard let scores = prediction.featureValue(for: "scores")?.multiArrayValue,
              let embeddings = prediction.featureValue(for: "embeddings")?.multiArrayValue else {
            throw ModelError.Failed(message: "Core ML model outputs do not match the bridge contract.")
        }
        return ModelOutput(
            audiosetScores: Self.floats(scores, count: 521),
            embedding: Self.floats(embeddings, count: 1_024)
        )
    }

    /// Copy an `MLMultiArray` out as `[Float]`.
    ///
    /// The subscript path boxes every element in an `NSNumber`; that is ~1,500
    /// allocations per inference, and inference runs for every gate-active
    /// analysis window. Read the backing storage directly for the contiguous
    /// float32 case the contract specifies, and keep the slow path only as a
    /// fallback for a model that was converted to some other layout or dtype.
    private static func floats(_ array: MLMultiArray, count: Int) -> [Float] {
        let count = min(count, array.count)
        let isContiguousFloat32 = array.dataType == .float32
            && array.strides.last?.intValue == 1
        guard isContiguousFloat32 else {
            return (0..<count).map { array[$0].floatValue }
        }
        return array.withUnsafeBufferPointer(ofType: Float.self) { buffer in
            Array(UnsafeBufferPointer(start: buffer.baseAddress, count: count))
        }
    }
}

/// Makes the UI runnable before the Core ML model is added on a Mac. It returns
/// correctly shaped neutral outputs, so no false detections are generated.
final class PreviewModelRunner: ModelRunner, @unchecked Sendable {
    func modelVersion() throws -> String {
        "preview-noop@1"
    }

    func infer(
        logMel: [Float],
        frames: UInt32,
        bands: UInt32,
        energyPeak: Bool
    ) throws -> ModelOutput {
        ModelOutput(
            audiosetScores: Array(repeating: 0, count: 521),
            embedding: Array(repeating: 0, count: 1_024)
        )
    }
}

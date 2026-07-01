@preconcurrency import CoreML
import Darwin
import Foundation

struct Fixture: Decodable {
    let promptLen: Int
    let hiddenSize: Int
    let headDim: Int
    let ropeTheta: Double
    let inputIds: [Int32]?
    let audioInputMask: [Bool]?
    let promptPrefixIds: [Int32]?
    let promptSuffixIds: [Int32]?
    let audioTokenCount: Int?
    let audioPlaceholderId: Int32?
    let audioDataShape: [Int]
    let audioData: [Float]
    let audioDataSeqlens: [Int32]
    let generatedIds: [Int32]

    enum CodingKeys: String, CodingKey {
        case promptLen = "prompt_len"
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case inputIds = "input_ids"
        case audioInputMask = "audio_input_mask"
        case promptPrefixIds = "prompt_prefix_ids"
        case promptSuffixIds = "prompt_suffix_ids"
        case audioTokenCount = "audio_token_count"
        case audioPlaceholderId = "audio_placeholder_id"
        case audioDataShape = "audio_data_shape"
        case audioData = "audio_data"
        case audioDataSeqlens = "audio_data_seqlens"
        case generatedIds = "generated_ids"
    }
}

struct RuntimeManifest: Decodable {
    let hiddenSize: Int
    let headDim: Int
    let ropeTheta: Double
    let promptPrefixIds: [Int32]
    let promptSuffixIds: [Int32]
    let audioPlaceholderId: Int32

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case promptPrefixIds = "prompt_prefix_ids"
        case promptSuffixIds = "prompt_suffix_ids"
        case audioPlaceholderId = "audio_placeholder_id"
    }
}

struct RuntimeContext {
    let sourcePath: String
    let promptLen: Int
    let hiddenSize: Int
    let headDim: Int
    let ropeTheta: Double
    let inputIds: [Int32]?
    let audioInputMask: [Bool]?
    let promptPrefixIds: [Int32]?
    let promptSuffixIds: [Int32]?
    let audioTokenCount: Int?
    let audioPlaceholderId: Int32?
    let audioDataShape: [Int]
    let audioData: [Float]
    let audioDataSeqlens: [Int32]
    let generatedIds: [Int32]
}

struct Prompt {
    let inputIds: [Int32]
    let audioInputMask: [Bool]
    let source: String

    var promptLen: Int {
        inputIds.count
    }

    var audioTokenCount: Int {
        audioInputMask.filter { $0 }.count
    }
}

struct Options {
    var packagesDir = URL(fileURLWithPath: "coreml/build")
    var fixture = URL(fileURLWithPath: "artifacts/coreml/moss_swift_fixture.json")
    var runtimeManifest: URL?
    var audio: URL?
    var audioMaxFrames: Int?
    var compareFixtureAudio = false
    var output: URL?
    var tokenPackage = "compiled/moss_token_embedding.mlmodelc"
    var audioPackage = "compiled_audio/moss_audio_encoder_adapter_fixture.mlmodelc"
    var decoderPackage = "compiled_stateful/moss_decoder_stateful_fused.mlmodelc"
    var prefillCachePackage: String?
    var prefillCacheSeqLen: Int?
    var stepPackage: String?
    var cacheLen = 768
    var tokenizer = URL(fileURLWithPath: "artifacts/coreml/moss_tokenizer.json")
    var referenceText: String?
    var referenceTextFile: URL?
    var batchManifest: URL?
    var batchOutputJsonl: URL?
    var computeUnits = MLComputeUnits.all
    var tokenMaxSeqLen = 512
    var maxNewTokens = 5
    var eosTokenId = 151645
}

struct BatchItem: Decodable {
    let rowIdx: Int?
    let id: String?
    let audio: String
    let referenceTextFile: String
    let output: String

    enum CodingKeys: String, CodingKey {
        case rowIdx = "row_idx"
        case id
        case audio
        case referenceTextFile = "reference_text_file"
        case output
    }
}

struct BatchLineResult: Encodable {
    let rowIdx: Int?
    let id: String?
    let output: String
    let promptLen: Int
    let audioTokenCount: Int
    let generatedTokenCount: Int
    let stoppedOnEos: Bool
    let normalizedWer: Double
    let normalizedCer: Double
    let rowWallSeconds: Double

    enum CodingKeys: String, CodingKey {
        case rowIdx = "row_idx"
        case id
        case output
        case promptLen = "prompt_len"
        case audioTokenCount = "audio_token_count"
        case generatedTokenCount = "generated_token_count"
        case stoppedOnEos = "stopped_on_eos"
        case normalizedWer = "normalized_wer"
        case normalizedCer = "normalized_cer"
        case rowWallSeconds = "row_wall_sec"
    }
}

struct LoadedModels {
    let tokenModel: MLModel
    let audioModel: MLModel
    let statefulDecoderModel: MLModel?
    let prefillCacheModel: MLModel?
    let stepModel: MLModel?
}

struct TopKEntry: Encodable {
    let index: Int
    let value: Float
}

struct Timing: Encodable {
    let audioFrontend: Double
    let tokenEmbeddingPrompt: Double
    let audioEncoderAdapter: Double
    let statefulDecoderPrefill: Double
    let tokenEmbeddingDecode: Double
    let statefulDecoderDecode: Double

    var total: Double {
        audioFrontend + tokenEmbeddingPrompt + audioEncoderAdapter + statefulDecoderPrefill
            + tokenEmbeddingDecode + statefulDecoderDecode
    }

    enum CodingKeys: String, CodingKey {
        case audioFrontend = "audio_frontend"
        case tokenEmbeddingPrompt = "token_embedding_prompt"
        case audioEncoderAdapter = "audio_encoder_adapter"
        case statefulDecoderPrefill = "stateful_decoder_prefill"
        case tokenEmbeddingDecode = "token_embedding_decode"
        case statefulDecoderDecode = "stateful_decoder_decode"
        case total
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(audioFrontend, forKey: .audioFrontend)
        try container.encode(tokenEmbeddingPrompt, forKey: .tokenEmbeddingPrompt)
        try container.encode(audioEncoderAdapter, forKey: .audioEncoderAdapter)
        try container.encode(statefulDecoderPrefill, forKey: .statefulDecoderPrefill)
        try container.encode(tokenEmbeddingDecode, forKey: .tokenEmbeddingDecode)
        try container.encode(statefulDecoderDecode, forKey: .statefulDecoderDecode)
        try container.encode(total, forKey: .total)
    }
}

struct Result: Encodable {
    let fixture: String
    let packagesDir: String
    let audioSource: String
    let audioDataShape: [Int]
    let audioDataSeqlens: [Int32]
    let audioFrontendDiff: AudioFeatureDiff?
    let promptSource: String
    let promptLen: Int
    let audioTokenCount: Int
    let firstTokenId: Int
    let secondTokenId: Int
    let maxNewTokens: Int
    let stoppedOnEos: Bool
    let decoderMode: String
    let generatedIds: [Int]
    let expectedGeneratedIds: [Int]
    let generatedPrefixMatchCount: Int
    let generatedPrefixMatchesExpected: Bool
    let generatedText: String
    let expectedText: String
    let expectedTextSource: String
    let normalizedGeneratedText: String
    let normalizedExpectedText: String
    let rawWer: Double
    let rawCer: Double
    let normalizedWer: Double
    let normalizedCer: Double
    let prefillTopK: [TopKEntry]
    let stepTopK: [TopKEntry]
    let prefillTop1MatchesFirstToken: Bool
    let stepTop1MatchesSecondToken: Bool
    let timingSeconds: Timing
    let rowWallSeconds: Double

    enum CodingKeys: String, CodingKey {
        case fixture
        case packagesDir = "packages_dir"
        case audioSource = "audio_source"
        case audioDataShape = "audio_data_shape"
        case audioDataSeqlens = "audio_data_seqlens"
        case audioFrontendDiff = "audio_frontend_diff"
        case promptSource = "prompt_source"
        case promptLen = "prompt_len"
        case audioTokenCount = "audio_token_count"
        case firstTokenId = "first_token_id"
        case secondTokenId = "second_token_id"
        case maxNewTokens = "max_new_tokens"
        case stoppedOnEos = "stopped_on_eos"
        case decoderMode = "decoder_mode"
        case generatedIds = "generated_ids"
        case expectedGeneratedIds = "expected_generated_ids"
        case generatedPrefixMatchCount = "generated_prefix_match_count"
        case generatedPrefixMatchesExpected = "generated_prefix_matches_expected"
        case generatedText = "generated_text"
        case expectedText = "expected_text"
        case expectedTextSource = "expected_text_source"
        case normalizedGeneratedText = "normalized_generated_text"
        case normalizedExpectedText = "normalized_expected_text"
        case rawWer = "raw_wer"
        case rawCer = "raw_cer"
        case normalizedWer = "normalized_wer"
        case normalizedCer = "normalized_cer"
        case prefillTopK = "prefill_topk"
        case stepTopK = "step_topk"
        case prefillTop1MatchesFirstToken = "prefill_top1_matches_first_token"
        case stepTop1MatchesSecondToken = "step_top1_matches_second_token"
        case timingSeconds = "timing_seconds"
        case rowWallSeconds = "row_wall_sec"
    }
}

enum RunnerError: Error, CustomStringConvertible {
    case missingArgument(String)
    case invalidArgument(String)
    case missingFeature(String)
    case unavailable(String)

    var description: String {
        switch self {
        case .missingArgument(let name):
            "missing value for \(name)"
        case .invalidArgument(let message):
            message
        case .missingFeature(let name):
            "CoreML output is missing feature \(name)"
        case .unavailable(let message):
            message
        }
    }
}

func parseOptions(_ arguments: [String]) throws -> Options {
    var options = Options()
    var index = 1
    while index < arguments.count {
        let argument = arguments[index]
        func value() throws -> String {
            guard index + 1 < arguments.count else {
                throw RunnerError.missingArgument(argument)
            }
            index += 1
            return arguments[index]
        }
        switch argument {
        case "--packages-dir":
            options.packagesDir = URL(fileURLWithPath: try value())
        case "--fixture":
            options.fixture = URL(fileURLWithPath: try value())
        case "--runtime-manifest":
            options.runtimeManifest = URL(fileURLWithPath: try value())
        case "--audio":
            options.audio = URL(fileURLWithPath: try value())
        case "--audio-max-frames":
            guard let parsed = Int(try value()) else {
                throw RunnerError.invalidArgument("invalid --audio-max-frames")
            }
            options.audioMaxFrames = parsed
        case "--compare-fixture-audio":
            options.compareFixtureAudio = true
        case "--output":
            options.output = URL(fileURLWithPath: try value())
        case "--token-package":
            options.tokenPackage = try value()
        case "--audio-package":
            options.audioPackage = try value()
        case "--decoder-package":
            options.decoderPackage = try value()
        case "--prefill-cache-package":
            options.prefillCachePackage = try value()
        case "--prefill-cache-seq-len":
            guard let parsed = Int(try value()) else {
                throw RunnerError.invalidArgument("invalid --prefill-cache-seq-len")
            }
            options.prefillCacheSeqLen = parsed
        case "--step-package":
            options.stepPackage = try value()
        case "--cache-len":
            guard let parsed = Int(try value()) else {
                throw RunnerError.invalidArgument("invalid --cache-len")
            }
            options.cacheLen = parsed
        case "--tokenizer":
            options.tokenizer = URL(fileURLWithPath: try value())
        case "--reference-text":
            options.referenceText = try value()
        case "--reference-text-file":
            options.referenceTextFile = URL(fileURLWithPath: try value())
        case "--batch-manifest":
            options.batchManifest = URL(fileURLWithPath: try value())
        case "--batch-output-jsonl":
            options.batchOutputJsonl = URL(fileURLWithPath: try value())
        case "--compute-units":
            options.computeUnits = try parseComputeUnits(try value())
        case "--token-max-seq-len":
            guard let parsed = Int(try value()) else {
                throw RunnerError.invalidArgument("invalid --token-max-seq-len")
            }
            options.tokenMaxSeqLen = parsed
        case "--max-new-tokens":
            guard let parsed = Int(try value()) else {
                throw RunnerError.invalidArgument("invalid --max-new-tokens")
            }
            options.maxNewTokens = parsed
        case "--eos-token-id":
            guard let parsed = Int(try value()) else {
                throw RunnerError.invalidArgument("invalid --eos-token-id")
            }
            options.eosTokenId = parsed
        case "--help", "-h":
            print(
                """
                Usage: moss-coreml-fixture [--packages-dir DIR]
                                           [--fixture JSON | --runtime-manifest JSON]
                                           [--output JSON]
                                           [--audio WAV] [--audio-max-frames N]
                                           [--compare-fixture-audio]
                                           [--token-package NAME] [--audio-package NAME]
                                           [--decoder-package NAME] [--token-max-seq-len N]
                                           [--prefill-cache-package NAME --step-package NAME]
                                           [--prefill-cache-seq-len N] [--cache-len N]
                                           [--max-new-tokens N] [--tokenizer JSON]
                                           [--reference-text TEXT | --reference-text-file FILE]
                                           [--batch-manifest JSONL --batch-output-jsonl JSONL]
                                           [--eos-token-id ID]
                                           [--compute-units all|cpu-only|cpu-gpu|cpu-ane]
                """
            )
            Darwin.exit(0)
        default:
            throw RunnerError.invalidArgument("unknown argument \(argument)")
        }
        index += 1
    }
    if options.referenceText != nil, options.referenceTextFile != nil {
        throw RunnerError.invalidArgument(
            "pass only one of --reference-text or --reference-text-file"
        )
    }
    if options.batchManifest != nil,
       (options.audio != nil || options.referenceText != nil || options.referenceTextFile != nil)
    {
        throw RunnerError.invalidArgument(
            "batch mode takes per-row audio/reference paths from --batch-manifest"
        )
    }
    return options
}

func parseComputeUnits(_ value: String) throws -> MLComputeUnits {
    switch value {
    case "all":
        return .all
    case "cpu-only":
        return .cpuOnly
    case "cpu-gpu":
        return .cpuAndGPU
    case "cpu-ane":
        return .cpuAndNeuralEngine
    default:
        throw RunnerError.invalidArgument("invalid --compute-units \(value)")
    }
}

func loadFixture(_ url: URL) throws -> Fixture {
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    return try decoder.decode(Fixture.self, from: data)
}

func loadRuntimeContext(options: Options) throws -> RuntimeContext {
    if let runtimeManifest = options.runtimeManifest {
        let data = try Data(contentsOf: runtimeManifest)
        let manifest = try JSONDecoder().decode(RuntimeManifest.self, from: data)
        return RuntimeContext(
            sourcePath: runtimeManifest.path,
            promptLen: 0,
            hiddenSize: manifest.hiddenSize,
            headDim: manifest.headDim,
            ropeTheta: manifest.ropeTheta,
            inputIds: nil,
            audioInputMask: nil,
            promptPrefixIds: manifest.promptPrefixIds,
            promptSuffixIds: manifest.promptSuffixIds,
            audioTokenCount: nil,
            audioPlaceholderId: manifest.audioPlaceholderId,
            audioDataShape: [],
            audioData: [],
            audioDataSeqlens: [],
            generatedIds: []
        )
    }
    let fixture = try loadFixture(options.fixture)
    return RuntimeContext(
        sourcePath: options.fixture.path,
        promptLen: fixture.promptLen,
        hiddenSize: fixture.hiddenSize,
        headDim: fixture.headDim,
        ropeTheta: fixture.ropeTheta,
        inputIds: fixture.inputIds,
        audioInputMask: fixture.audioInputMask,
        promptPrefixIds: fixture.promptPrefixIds,
        promptSuffixIds: fixture.promptSuffixIds,
        audioTokenCount: fixture.audioTokenCount,
        audioPlaceholderId: fixture.audioPlaceholderId,
        audioDataShape: fixture.audioDataShape,
        audioData: fixture.audioData,
        audioDataSeqlens: fixture.audioDataSeqlens,
        generatedIds: fixture.generatedIds
    )
}

func resolvePrompt(_ runtime: RuntimeContext, audioTokenCountOverride: Int? = nil) throws -> Prompt {
    if let prefixIds = runtime.promptPrefixIds,
       let suffixIds = runtime.promptSuffixIds
    {
        guard let fixtureAudioTokenCount = runtime.audioTokenCount ?? audioTokenCountOverride else {
            throw RunnerError.invalidArgument(
                "runtime manifest prompt construction requires audio input"
            )
        }
        let audioTokenCount = audioTokenCountOverride ?? fixtureAudioTokenCount
        guard audioTokenCount >= 0 else {
            throw RunnerError.invalidArgument("audio_token_count must be non-negative")
        }
        let placeholderId = runtime.audioPlaceholderId ?? 0
        let inputIds = prefixIds
            + [Int32](repeating: placeholderId, count: audioTokenCount)
            + suffixIds
        let audioInputMask = [Bool](repeating: false, count: prefixIds.count)
            + [Bool](repeating: true, count: audioTokenCount)
            + [Bool](repeating: false, count: suffixIds.count)
        if audioTokenCountOverride == nil {
            guard inputIds.count == runtime.promptLen else {
                throw RunnerError.invalidArgument(
                    "compact prompt length \(inputIds.count) != fixture prompt_len \(runtime.promptLen)"
                )
            }
        }
        let source = audioTokenCountOverride == nil ? "compact" : "compact_audio"
        return Prompt(inputIds: inputIds, audioInputMask: audioInputMask, source: source)
    }

    if audioTokenCountOverride != nil {
        throw RunnerError.invalidArgument(
            "audio override requires compact prompt_prefix_ids/prompt_suffix_ids fields"
        )
    }
    guard let inputIds = runtime.inputIds, let audioInputMask = runtime.audioInputMask else {
        throw RunnerError.invalidArgument(
            "fixture needs either compact prompt fields or input_ids/audio_input_mask"
        )
    }
    guard inputIds.count == runtime.promptLen else {
        throw RunnerError.invalidArgument(
            "input_ids length \(inputIds.count) != fixture prompt_len \(runtime.promptLen)"
        )
    }
    guard audioInputMask.count == runtime.promptLen else {
        throw RunnerError.invalidArgument(
            "audio_input_mask length \(audioInputMask.count) != fixture prompt_len \(runtime.promptLen)"
        )
    }
    return Prompt(inputIds: inputIds, audioInputMask: audioInputMask, source: "serialized")
}

func resolveExpectedText(
    options: Options,
    tokenizer: QwenByteLevelTokenizer,
    expectedGeneratedIds: [Int]
) throws -> (text: String, source: String) {
    if let referenceText = options.referenceText {
        return (referenceText, "reference_text")
    }
    if let referenceTextFile = options.referenceTextFile {
        let text = try String(contentsOf: referenceTextFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (text, "reference_text_file")
    }
    return (tokenizer.decode(expectedGeneratedIds), "fixture_tokens")
}

func resolveAudioFeatures(
    runtime: RuntimeContext,
    options: Options
) throws -> (features: MossAudioFeatures, diff: AudioFeatureDiff?, seconds: Double) {
    guard let audioURL = options.audio else {
        guard !runtime.audioData.isEmpty, !runtime.audioDataShape.isEmpty else {
            throw RunnerError.invalidArgument(
                "runtime manifest mode requires --audio; fixture audio is not available"
            )
        }
        return (
            features: MossAudioFeatures(
                source: "fixture",
                shape: runtime.audioDataShape,
                data: runtime.audioData,
                seqlens: runtime.audioDataSeqlens
            ),
            diff: nil,
            seconds: 0
        )
    }

    let start = DispatchTime.now().uptimeNanoseconds
    let samples = try MossAudioFile.loadMono16k(url: audioURL)
    let frontend = try WhisperLogMelFrontend()
    var features = try frontend.compute(samples: samples, source: audioURL.path)
    if let audioMaxFrames = options.audioMaxFrames {
        features = try padAudioFeatures(features, frames: audioMaxFrames)
    }
    let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
    let diff: AudioFeatureDiff?
    if options.compareFixtureAudio {
        guard !runtime.audioData.isEmpty, !runtime.audioDataShape.isEmpty else {
            throw RunnerError.invalidArgument(
                "--compare-fixture-audio requires fixture audio data"
            )
        }
        diff = compareAudioFeaturesPrefix(
            features.data,
            leftShape: features.shape,
            runtime.audioData,
            rightShape: runtime.audioDataShape
        )
    } else {
        diff = nil
    }
    return (
        features: features,
        diff: diff,
        seconds: seconds
    )
}

func makeFloatArray(shape: [Int], values: [Float]? = nil) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
    if let values {
        guard values.count == array.count else {
            throw RunnerError.invalidArgument("value count \(values.count) != array count \(array.count)")
        }
        values.withUnsafeBufferPointer { source in
            pointer.update(from: source.baseAddress!, count: values.count)
        }
    } else {
        for index in 0..<array.count {
            pointer[index] = 0
        }
    }
    return array
}

func makeIntArray(shape: [Int], values: [Int32]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
    guard values.count == array.count else {
        throw RunnerError.invalidArgument("value count \(values.count) != array count \(array.count)")
    }
    let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
    values.withUnsafeBufferPointer { source in
        pointer.update(from: source.baseAddress!, count: values.count)
    }
    return array
}

func paddedIds(_ ids: [Int32], maxSeqLen: Int) throws -> MLMultiArray {
    guard ids.count <= maxSeqLen else {
        throw RunnerError.invalidArgument("ids length \(ids.count) exceeds \(maxSeqLen)")
    }
    var padded = [Int32](repeating: 0, count: maxSeqLen)
    for (index, value) in ids.enumerated() {
        padded[index] = value
    }
    return try makeIntArray(shape: [1, maxSeqLen], values: padded)
}

func featureArray(_ provider: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
    guard let array = provider.featureValue(for: name)?.multiArrayValue else {
        throw RunnerError.missingFeature(name)
    }
    return array
}

func strides(_ array: MLMultiArray) -> [Int] {
    array.strides.map(\.intValue)
}

func offset(_ indices: [Int], strides: [Int]) -> Int {
    zip(indices, strides).reduce(0) { partial, pair in
        partial + pair.0 * pair.1
    }
}

func floatValue(_ array: MLMultiArray, _ indices: [Int]) -> Float {
    let storageIndex = offset(indices, strides: strides(array))
    switch array.dataType {
    case .float32:
        return array.dataPointer.bindMemory(to: Float.self, capacity: array.count)[storageIndex]
    case .float16:
        return Float(array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)[storageIndex])
    case .double:
        return Float(array.dataPointer.bindMemory(to: Double.self, capacity: array.count)[storageIndex])
    default:
        return array[indices.map { NSNumber(value: $0) }].floatValue
    }
}

func setFloat(_ array: MLMultiArray, _ indices: [Int], _ value: Float) {
    let storageIndex = offset(indices, strides: strides(array))
    array.dataPointer.bindMemory(to: Float.self, capacity: array.count)[storageIndex] = value
}

func shapeInts(_ array: MLMultiArray) -> [Int] {
    array.shape.map(\.intValue)
}

func padDecoderCache(_ cache: MLMultiArray, cacheLen: Int, featureName: String) throws -> MLMultiArray {
    let shape = shapeInts(cache)
    guard shape.count == 5 else {
        throw RunnerError.invalidArgument("\(featureName) must have rank 5, got \(shape)")
    }
    let pastLen = shape[3]
    guard cacheLen >= pastLen else {
        throw RunnerError.invalidArgument(
            "cache-len \(cacheLen) is shorter than \(featureName) prompt length \(pastLen)"
        )
    }
    let padded = try makeFloatArray(
        shape: [shape[0], shape[1], shape[2], cacheLen, shape[4]]
    )
    for layer in 0..<shape[0] {
        for batch in 0..<shape[1] {
            for head in 0..<shape[2] {
                for position in 0..<pastLen {
                    for dim in 0..<shape[4] {
                        setFloat(
                            padded,
                            [layer, batch, head, position, dim],
                            floatValue(cache, [layer, batch, head, position, dim])
                        )
                    }
                }
            }
        }
    }
    return padded
}

func buildPaddedStepInputs(cacheLen: Int, pastLen: Int) throws -> (
    cacheUpdateMask: MLMultiArray, attentionMask: MLMultiArray
) {
    guard pastLen >= 0, pastLen < cacheLen else {
        throw RunnerError.invalidArgument("past length \(pastLen) is outside cache-len \(cacheLen)")
    }
    let cacheUpdateMask = try makeFloatArray(shape: [1, 1, cacheLen, 1])
    setFloat(cacheUpdateMask, [0, 0, pastLen, 0], 1)

    let attentionMask = try makeFloatArray(shape: [1, 1, 1, cacheLen])
    for position in 0..<cacheLen {
        setFloat(attentionMask, [0, 0, 0, position], position <= pastLen ? 0 : -1_000_000_000.0)
    }
    return (cacheUpdateMask, attentionMask)
}

func buildMergedEmbeddings(
    tokenEmbeddings: MLMultiArray,
    audioEmbeddings: MLMultiArray,
    runtime: RuntimeContext,
    prompt: Prompt
) throws -> MLMultiArray {
    let merged = try makeFloatArray(shape: [1, prompt.promptLen, runtime.hiddenSize])
    var audioIndex = 0
    for position in 0..<prompt.promptLen {
        let sourceIsAudio = prompt.audioInputMask[position]
        for hidden in 0..<runtime.hiddenSize {
            let value: Float
            if sourceIsAudio {
                value = floatValue(audioEmbeddings, [audioIndex, hidden])
            } else {
                value = floatValue(tokenEmbeddings, [0, position, hidden])
            }
            setFloat(merged, [0, position, hidden], value)
        }
        if sourceIsAudio {
            audioIndex += 1
        }
    }
    guard audioIndex == prompt.audioTokenCount else {
        throw RunnerError.invalidArgument(
            "merged \(audioIndex) audio embeddings but prompt expects \(prompt.audioTokenCount)"
        )
    }
    return merged
}

func padMergedEmbeddings(
    _ mergedEmbeddings: MLMultiArray,
    promptLen: Int,
    seqLen: Int,
    hiddenSize: Int
) throws -> MLMultiArray {
    guard seqLen >= promptLen else {
        throw RunnerError.invalidArgument(
            "prefill cache seq len \(seqLen) is shorter than prompt length \(promptLen)"
        )
    }
    let padded = try makeFloatArray(shape: [1, seqLen, hiddenSize])
    for position in 0..<promptLen {
        for hidden in 0..<hiddenSize {
            setFloat(
                padded,
                [0, position, hidden],
                floatValue(mergedEmbeddings, [0, position, hidden])
            )
        }
    }
    return padded
}

func buildLastTokenMask(promptLen: Int, seqLen: Int) throws -> MLMultiArray {
    guard promptLen >= 1, promptLen <= seqLen else {
        throw RunnerError.invalidArgument(
            "prompt length \(promptLen) is outside prefill cache seq len \(seqLen)"
        )
    }
    let mask = try makeFloatArray(shape: [1, seqLen, 1])
    setFloat(mask, [0, promptLen - 1, 0], 1)
    return mask
}

func firstTokenEmbedding(_ tokenEmbeddings: MLMultiArray, hiddenSize: Int) throws -> MLMultiArray {
    let tokenEmbedding = try makeFloatArray(shape: [1, 1, hiddenSize])
    for hidden in 0..<hiddenSize {
        setFloat(
            tokenEmbedding,
            [0, 0, hidden],
            floatValue(tokenEmbeddings, [0, 0, hidden])
        )
    }
    return tokenEmbedding
}

func buildRope(length: Int, start: Int, headDim: Int, ropeTheta: Double) throws -> (
    cos: MLMultiArray, sin: MLMultiArray
) {
    let cosArray = try makeFloatArray(shape: [1, length, headDim])
    let sinArray = try makeFloatArray(shape: [1, length, headDim])
    let half = headDim / 2
    for localPosition in 0..<length {
        let position = Double(start + localPosition)
        for index in 0..<half {
            let exponent = Double(2 * index) / Double(headDim)
            let angle = position / pow(ropeTheta, exponent)
            let cosine = Float(Darwin.cos(angle))
            let sine = Float(Darwin.sin(angle))
            setFloat(cosArray, [0, localPosition, index], cosine)
            setFloat(cosArray, [0, localPosition, index + half], cosine)
            setFloat(sinArray, [0, localPosition, index], sine)
            setFloat(sinArray, [0, localPosition, index + half], sine)
        }
    }
    return (cosArray, sinArray)
}

func buildCausalMask(length: Int) throws -> MLMultiArray {
    let mask = try makeFloatArray(shape: [1, 1, length, length])
    for query in 0..<length {
        for key in 0..<length {
            setFloat(mask, [0, 0, query, key], key > query ? -1_000_000_000.0 : 0.0)
        }
    }
    return mask
}

func topK(_ logits: MLMultiArray, count: Int = 5) -> [TopKEntry] {
    var winners: [TopKEntry] = []
    for index in 0..<logits.count {
        let value: Float
        switch logits.dataType {
        case .float32:
            value = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)[index]
        case .float16:
            value = Float(logits.dataPointer.bindMemory(to: Float16.self, capacity: logits.count)[index])
        case .double:
            value = Float(logits.dataPointer.bindMemory(to: Double.self, capacity: logits.count)[index])
        default:
            value = logits[index].floatValue
        }
        if !value.isFinite {
            continue
        }
        winners.append(TopKEntry(index: index, value: value))
        winners.sort { $0.value > $1.value }
        if winners.count > count {
            winners.removeLast()
        }
    }
    return winners
}

func timedPrediction(
    model: MLModel,
    input: MLFeatureProvider,
    state: MLState? = nil
) async throws -> (MLFeatureProvider, Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let output: MLFeatureProvider
    if #available(macOS 15, *), let state {
        output = try await model.prediction(from: input, using: state)
    } else if state != nil {
        throw RunnerError.unavailable("MLState requires macOS 15+")
    } else {
        output = try await model.prediction(from: input)
    }
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
    return (output, elapsed)
}

func validateDecoderOptions(_ options: Options) throws -> Bool {
    let useExternalCache = options.prefillCachePackage != nil || options.stepPackage != nil
    if useExternalCache && (options.prefillCachePackage == nil || options.stepPackage == nil) {
        throw RunnerError.invalidArgument(
            "external cache mode requires both --prefill-cache-package and --step-package"
        )
    }
    return useExternalCache
}

func loadModels(options: Options, useExternalCache: Bool) throws -> LoadedModels {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = options.computeUnits
    let tokenModel = try MLModel(
        contentsOf: options.packagesDir.appendingPathComponent(options.tokenPackage),
        configuration: configuration
    )
    let audioModel = try MLModel(
        contentsOf: options.packagesDir.appendingPathComponent(options.audioPackage),
        configuration: configuration
    )
    if useExternalCache {
        return LoadedModels(
            tokenModel: tokenModel,
            audioModel: audioModel,
            statefulDecoderModel: nil,
            prefillCacheModel: try MLModel(
                contentsOf: options.packagesDir.appendingPathComponent(options.prefillCachePackage!),
                configuration: configuration
            ),
            stepModel: try MLModel(
                contentsOf: options.packagesDir.appendingPathComponent(options.stepPackage!),
                configuration: configuration
            )
        )
    }
    guard #available(macOS 15, *) else {
        throw RunnerError.unavailable("stateful decoder requires macOS 15+")
    }
    return LoadedModels(
        tokenModel: tokenModel,
        audioModel: audioModel,
        statefulDecoderModel: try MLModel(
            contentsOf: options.packagesDir.appendingPathComponent(options.decoderPackage),
            configuration: configuration
        ),
        prefillCacheModel: nil,
        stepModel: nil
    )
}

func runFixture(
    options: Options,
    runtime: RuntimeContext,
    tokenizer: QwenByteLevelTokenizer,
    models: LoadedModels,
    decoderMode: String
) async throws -> Result {
    let rowStart = DispatchTime.now().uptimeNanoseconds
    let (audioFeatures, audioFrontendDiff, audioFrontendSeconds) = try resolveAudioFeatures(
        runtime: runtime,
        options: options
    )
    guard let audioFrameCount = audioFeatures.seqlens.first else {
        throw RunnerError.invalidArgument("audio_data_seqlens is empty")
    }
    let audioTokenCountOverride: Int? = if options.audio == nil {
        nil
    } else {
        mossAudioTokenCount(melFrames: Int(audioFrameCount))
    }
    let prompt = try resolvePrompt(
        runtime,
        audioTokenCountOverride: audioTokenCountOverride
    )
    let maxNewTokens = options.maxNewTokens
    guard maxNewTokens > 0 else {
        throw RunnerError.invalidArgument("--max-new-tokens must be positive")
    }
    let firstToken = runtime.generatedIds.indices.contains(0) ? Int(runtime.generatedIds[0]) : -1
    let secondToken = runtime.generatedIds.indices.contains(1) ? Int(runtime.generatedIds[1]) : -1
    let audioTokenCount = prompt.audioTokenCount
    let useExternalCache = decoderMode == "external_cache"

    let tokenInput = try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": MLFeatureValue(
            multiArray: try paddedIds(prompt.inputIds, maxSeqLen: options.tokenMaxSeqLen)
        )
    ])
    let (tokenOutput, tokenSeconds) = try await timedPrediction(
        model: models.tokenModel,
        input: tokenInput
    )
    let tokenEmbeddings = try featureArray(tokenOutput, "token_embeddings")

    let audioInput = try MLDictionaryFeatureProvider(dictionary: [
        "audio_data": MLFeatureValue(
            multiArray: try makeFloatArray(shape: audioFeatures.shape, values: audioFeatures.data)
        ),
        "audio_data_seqlens": MLFeatureValue(
            multiArray: try makeIntArray(shape: [1], values: audioFeatures.seqlens)
        ),
    ])
    let (audioOutput, audioSeconds) = try await timedPrediction(
        model: models.audioModel,
        input: audioInput
    )
    let audioEmbeddings = try featureArray(audioOutput, "audio_embeddings")
    let mergedEmbeddings = try buildMergedEmbeddings(
        tokenEmbeddings: tokenEmbeddings,
        audioEmbeddings: audioEmbeddings,
        runtime: runtime,
        prompt: prompt
    )

    let prefillTopK: [TopKEntry]
    let prefillSeconds: Double
    var generatedIds: [Int]
    var stoppedOnEos: Bool
    var firstStepTopK: [TopKEntry] = []
    var currentToken: Int32
    var decodeTokenSeconds = 0.0
    var decodeStepSeconds = 0.0

    if useExternalCache {
        guard let prefillCacheModel = models.prefillCacheModel,
              let stepModel = models.stepModel
        else {
            throw RunnerError.invalidArgument("external cache models were not loaded")
        }
        var prefillFeatures: [String: MLFeatureValue] = [:]
        if let prefillCacheSeqLen = options.prefillCacheSeqLen {
            prefillFeatures["inputs_embeds"] = MLFeatureValue(
                multiArray: try padMergedEmbeddings(
                    mergedEmbeddings,
                    promptLen: prompt.promptLen,
                    seqLen: prefillCacheSeqLen,
                    hiddenSize: runtime.hiddenSize
                )
            )
            prefillFeatures["last_token_mask"] = MLFeatureValue(
                multiArray: try buildLastTokenMask(
                    promptLen: prompt.promptLen,
                    seqLen: prefillCacheSeqLen
                )
            )
        } else {
            prefillFeatures["inputs_embeds"] = MLFeatureValue(multiArray: mergedEmbeddings)
        }
        let prefillInput = try MLDictionaryFeatureProvider(dictionary: prefillFeatures)
        let (prefillOutput, elapsed) = try await timedPrediction(
            model: prefillCacheModel,
            input: prefillInput
        )
        prefillSeconds = elapsed
        prefillTopK = try topK(featureArray(prefillOutput, "logits"))
        guard let firstGenerated = prefillTopK.first?.index else {
            throw RunnerError.invalidArgument("prefill produced no logits")
        }
        generatedIds = [firstGenerated]
        stoppedOnEos = firstGenerated == options.eosTokenId
        currentToken = Int32(firstGenerated)
        var pastKeys = try padDecoderCache(
            try featureArray(prefillOutput, "past_keys"),
            cacheLen: options.cacheLen,
            featureName: "past_keys"
        )
        var pastValues = try padDecoderCache(
            try featureArray(prefillOutput, "past_values"),
            cacheLen: options.cacheLen,
            featureName: "past_values"
        )

        for stepIndex in 0..<max(0, maxNewTokens - 1) where !stoppedOnEos {
            let tokenInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(
                    multiArray: try paddedIds([currentToken], maxSeqLen: options.tokenMaxSeqLen)
                )
            ])
            let (tokenOutput, tokenDecodeSeconds) = try await timedPrediction(
                model: models.tokenModel,
                input: tokenInput
            )
            decodeTokenSeconds += tokenDecodeSeconds
            let tokenEmbedding = try firstTokenEmbedding(
                try featureArray(tokenOutput, "token_embeddings"),
                hiddenSize: runtime.hiddenSize
            )

            let stepPosition = prompt.promptLen + stepIndex
            let (stepCos, stepSin) = try buildRope(
                length: 1,
                start: stepPosition,
                headDim: runtime.headDim,
                ropeTheta: runtime.ropeTheta
            )
            let masks = try buildPaddedStepInputs(
                cacheLen: options.cacheLen,
                pastLen: stepPosition
            )
            let stepInput = try MLDictionaryFeatureProvider(dictionary: [
                "inputs_embeds": MLFeatureValue(multiArray: tokenEmbedding),
                "past_keys": MLFeatureValue(multiArray: pastKeys),
                "past_values": MLFeatureValue(multiArray: pastValues),
                "cache_update_mask": MLFeatureValue(multiArray: masks.cacheUpdateMask),
                "attention_mask": MLFeatureValue(multiArray: masks.attentionMask),
                "cos": MLFeatureValue(multiArray: stepCos),
                "sin": MLFeatureValue(multiArray: stepSin),
            ])
            let (stepOutput, stepSeconds) = try await timedPrediction(
                model: stepModel,
                input: stepInput
            )
            decodeStepSeconds += stepSeconds
            let stepTopK = try topK(featureArray(stepOutput, "logits"))
            if stepIndex == 0 {
                firstStepTopK = stepTopK
            }
            guard let nextToken = stepTopK.first?.index else {
                throw RunnerError.invalidArgument("decode step produced no logits")
            }
            generatedIds.append(nextToken)
            if nextToken == options.eosTokenId {
                stoppedOnEos = true
            }
            currentToken = Int32(nextToken)
            pastKeys = try featureArray(stepOutput, "updated_keys")
            pastValues = try featureArray(stepOutput, "updated_values")
        }
    } else {
        guard let decoderModel = models.statefulDecoderModel else {
            throw RunnerError.invalidArgument("stateful decoder model was not loaded")
        }
        guard #available(macOS 15, *) else {
            throw RunnerError.unavailable("stateful decoder requires macOS 15+")
        }
        let state = decoderModel.makeState()
        let (prefillCos, prefillSin) = try buildRope(
            length: prompt.promptLen,
            start: 0,
            headDim: runtime.headDim,
            ropeTheta: runtime.ropeTheta
        )
        let prefillMask = try buildCausalMask(length: prompt.promptLen)
        let prefillInput = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: mergedEmbeddings),
            "cos": MLFeatureValue(multiArray: prefillCos),
            "sin": MLFeatureValue(multiArray: prefillSin),
            "attention_mask": MLFeatureValue(multiArray: prefillMask),
        ])
        let (prefillOutput, elapsed) = try await timedPrediction(
            model: decoderModel,
            input: prefillInput,
            state: state
        )
        prefillSeconds = elapsed
        prefillTopK = try topK(featureArray(prefillOutput, "logits"))
        guard let firstGenerated = prefillTopK.first?.index else {
            throw RunnerError.invalidArgument("prefill produced no logits")
        }

        generatedIds = [firstGenerated]
        stoppedOnEos = firstGenerated == options.eosTokenId
        currentToken = Int32(firstGenerated)
        for stepIndex in 0..<max(0, maxNewTokens - 1) where !stoppedOnEos {
            let tokenInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(
                    multiArray: try paddedIds([currentToken], maxSeqLen: options.tokenMaxSeqLen)
                )
            ])
            let (tokenOutput, tokenDecodeSeconds) = try await timedPrediction(
                model: models.tokenModel,
                input: tokenInput
            )
            decodeTokenSeconds += tokenDecodeSeconds
            let tokenEmbedding = try firstTokenEmbedding(
                try featureArray(tokenOutput, "token_embeddings"),
                hiddenSize: runtime.hiddenSize
            )

            let stepPosition = prompt.promptLen + stepIndex
            let (stepCos, stepSin) = try buildRope(
                length: 1,
                start: stepPosition,
                headDim: runtime.headDim,
                ropeTheta: runtime.ropeTheta
            )
            let stepMask = try makeFloatArray(shape: [1, 1, 1, stepPosition + 1])
            let stepInput = try MLDictionaryFeatureProvider(dictionary: [
                "inputs_embeds": MLFeatureValue(multiArray: tokenEmbedding),
                "cos": MLFeatureValue(multiArray: stepCos),
                "sin": MLFeatureValue(multiArray: stepSin),
                "attention_mask": MLFeatureValue(multiArray: stepMask),
            ])
            let (stepOutput, stepSeconds) = try await timedPrediction(
                model: decoderModel,
                input: stepInput,
                state: state
            )
            decodeStepSeconds += stepSeconds
            let stepTopK = try topK(featureArray(stepOutput, "logits"))
            if stepIndex == 0 {
                firstStepTopK = stepTopK
            }
            guard let nextToken = stepTopK.first?.index else {
                throw RunnerError.invalidArgument("decode step produced no logits")
            }
            generatedIds.append(nextToken)
            if nextToken == options.eosTokenId {
                stoppedOnEos = true
            }
            currentToken = Int32(nextToken)
        }
    }
    let expectedGeneratedIds = runtime.generatedIds.prefix(maxNewTokens).map { Int($0) }
    var generatedPrefixMatchCount = 0
    for (actual, expected) in zip(generatedIds, expectedGeneratedIds) {
        guard actual == expected else { break }
        generatedPrefixMatchCount += 1
    }
    let generatedPrefixMatchesExpected = generatedPrefixMatchCount == expectedGeneratedIds.count
    let generatedText = tokenizer.decode(generatedIds)
    let expected = try resolveExpectedText(
        options: options,
        tokenizer: tokenizer,
        expectedGeneratedIds: expectedGeneratedIds
    )
    let expectedText = expected.text
    let normalizedGeneratedText = normalizedTranscript(generatedText)
    let normalizedExpectedText = normalizedTranscript(expectedText)
    let rowWallSeconds = Double(
        DispatchTime.now().uptimeNanoseconds - rowStart
    ) / 1_000_000_000.0

    return Result(
        fixture: runtime.sourcePath,
        packagesDir: options.packagesDir.path,
        audioSource: audioFeatures.source,
        audioDataShape: audioFeatures.shape,
        audioDataSeqlens: audioFeatures.seqlens,
        audioFrontendDiff: audioFrontendDiff,
        promptSource: prompt.source,
        promptLen: prompt.promptLen,
        audioTokenCount: audioTokenCount,
        firstTokenId: firstToken,
        secondTokenId: secondToken,
        maxNewTokens: maxNewTokens,
        stoppedOnEos: stoppedOnEos,
        decoderMode: decoderMode,
        generatedIds: generatedIds,
        expectedGeneratedIds: expectedGeneratedIds,
        generatedPrefixMatchCount: generatedPrefixMatchCount,
        generatedPrefixMatchesExpected: generatedPrefixMatchesExpected,
        generatedText: generatedText,
        expectedText: expectedText,
        expectedTextSource: expected.source,
        normalizedGeneratedText: normalizedGeneratedText,
        normalizedExpectedText: normalizedExpectedText,
        rawWer: wordErrorRate(reference: expectedText, hypothesis: generatedText),
        rawCer: characterErrorRate(reference: expectedText, hypothesis: generatedText),
        normalizedWer: wordErrorRate(
            reference: normalizedExpectedText,
            hypothesis: normalizedGeneratedText
        ),
        normalizedCer: characterErrorRate(
            reference: normalizedExpectedText,
            hypothesis: normalizedGeneratedText
        ),
        prefillTopK: prefillTopK,
        stepTopK: firstStepTopK,
        prefillTop1MatchesFirstToken: prefillTopK.first?.index == firstToken,
        stepTop1MatchesSecondToken: generatedIds.count > 1 && generatedIds[1] == secondToken,
        timingSeconds: Timing(
            audioFrontend: audioFrontendSeconds,
            tokenEmbeddingPrompt: tokenSeconds,
            audioEncoderAdapter: audioSeconds,
            statefulDecoderPrefill: prefillSeconds,
            tokenEmbeddingDecode: decodeTokenSeconds,
            statefulDecoderDecode: decodeStepSeconds
        ),
        rowWallSeconds: rowWallSeconds
    )
}

func encodedResult(_ result: Result) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try encoder.encode(result)
}

func writeResult(_ result: Result, output: URL?) throws -> Data {
    let data = try encodedResult(result)
    if let output {
        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: output)
    }
    return data
}

func loadBatchItems(_ manifest: URL) throws -> [BatchItem] {
    let decoder = JSONDecoder()
    return try String(contentsOf: manifest, encoding: .utf8)
        .split(separator: "\n")
        .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        .map { line in
            try decoder.decode(BatchItem.self, from: Data(line.utf8))
        }
}

func appendJSONLine<T: Encodable>(_ value: T, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(value)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    if !FileManager.default.fileExists(atPath: url.path) {
        FileManager.default.createFile(atPath: url.path, contents: nil)
    }
    let handle = try FileHandle(forWritingTo: url)
    try handle.seekToEnd()
    try handle.write(contentsOf: data)
    try handle.write(contentsOf: Data("\n".utf8))
    try handle.close()
}

func runBatch(
    options: Options,
    runtime: RuntimeContext,
    tokenizer: QwenByteLevelTokenizer,
    models: LoadedModels,
    decoderMode: String
) async throws {
    guard let manifest = options.batchManifest else {
        throw RunnerError.invalidArgument("missing --batch-manifest")
    }
    let batchOutput = options.batchOutputJsonl
        ?? manifest.deletingPathExtension().appendingPathExtension("results.jsonl")
    try FileManager.default.createDirectory(
        at: batchOutput.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    FileManager.default.createFile(atPath: batchOutput.path, contents: nil)
    let items = try loadBatchItems(manifest)
    for item in items {
        var rowOptions = options
        rowOptions.audio = URL(fileURLWithPath: item.audio)
        rowOptions.referenceText = nil
        rowOptions.referenceTextFile = URL(fileURLWithPath: item.referenceTextFile)
        rowOptions.output = URL(fileURLWithPath: item.output)
        rowOptions.batchManifest = nil
        rowOptions.batchOutputJsonl = nil
        let result = try await runFixture(
            options: rowOptions,
            runtime: runtime,
            tokenizer: tokenizer,
            models: models,
            decoderMode: decoderMode
        )
        _ = try writeResult(result, output: rowOptions.output)
        let line = BatchLineResult(
            rowIdx: item.rowIdx,
            id: item.id,
            output: item.output,
            promptLen: result.promptLen,
            audioTokenCount: result.audioTokenCount,
            generatedTokenCount: result.generatedIds.count,
            stoppedOnEos: result.stoppedOnEos,
            normalizedWer: result.normalizedWer,
            normalizedCer: result.normalizedCer,
            rowWallSeconds: result.rowWallSeconds
        )
        try appendJSONLine(line, to: batchOutput)
        let stdoutLine = try JSONEncoder().encode(line)
        print(String(decoding: stdoutLine, as: UTF8.self))
        fflush(stdout)
    }
}

@main
struct MossCoreMLFixture {
    static func main() async throws {
        let options = try parseOptions(CommandLine.arguments)
        let runtime = try loadRuntimeContext(options: options)
        let tokenizer = try QwenByteLevelTokenizer(tokenizerJSON: options.tokenizer)
        let useExternalCache = try validateDecoderOptions(options)
        let decoderMode = useExternalCache ? "external_cache" : "stateful"
        let models = try loadModels(options: options, useExternalCache: useExternalCache)
        if options.batchManifest != nil {
            try await runBatch(
                options: options,
                runtime: runtime,
                tokenizer: tokenizer,
                models: models,
                decoderMode: decoderMode
            )
            return
        }
        let result = try await runFixture(
            options: options,
            runtime: runtime,
            tokenizer: tokenizer,
            models: models,
            decoderMode: decoderMode
        )
        let data = try writeResult(result, output: options.output)
        print(String(decoding: data, as: UTF8.self))
    }
}

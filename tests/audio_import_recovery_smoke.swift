import Foundation
import AVFoundation

@main struct AudioImportRecoverySmoke {
    @MainActor static func wait(_ store: StudioStore) async throws {
        let end = Date().addingTimeInterval(10)
        while store.busy && Date() < end { try await Task.sleep(for: .milliseconds(20)) }
        precondition(!store.busy, "Import did not finish")
    }
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let suite = "local.avatarstudio.audio-recovery.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("right-channel-fixture.wav")
        let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 2)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 32000)!
        buffer.frameLength = 32000
        buffer.floatChannelData![0].initialize(repeating: 0, count: 32000)
        buffer.floatChannelData![1].initialize(repeating: 0.25, count: 32000)
        do { let file = try AVAudioFile(forWriting: original, settings: format.settings); try file.write(from: buffer) }
        let originalBytes = try Data(contentsOf: original)
        let store = StudioStore(storageRoot: root.appendingPathComponent("library"), preferences: preferences)
        store.importAudio(from: original); try await wait(store)
        precondition(store.error == nil && store.audioDuration == 2)
        precondition(store.waveform.max() == 0.25, "Right-channel voice was ignored")
        let goodPath = store.audio, goodWaveform = store.waveform
        let copied = try Data(contentsOf: URL(fileURLWithPath: goodPath))
        precondition(copied == originalBytes)
        let bad = root.appendingPathComponent("damaged.wav")
        try Data("Corrupt recording fixture".utf8).write(to: bad)
        store.importAudio(from: bad); try await wait(store)
        precondition(store.error?.contains("decoded") == true)
        precondition(store.audio == goodPath && store.waveform == goodWaveform && store.audioDuration == 2)
        let invalidSamples = root.appendingPathComponent("invalid-samples.wav")
        buffer.floatChannelData![0][0] = .nan
        do { let file = try AVAudioFile(forWriting: invalidSamples, settings: format.settings); try file.write(from: buffer) }
        store.importAudio(from: invalidSamples); try await wait(store)
        precondition(store.error?.contains("invalid audio samples") == true)
        precondition(store.audio == goodPath && store.waveform == goodWaveform)
        let files = try FileManager.default.contentsOfDirectory(atPath: store.root.appendingPathComponent("audio").path)
        precondition(files.count == 1, "Failed import left an orphan recording")
        let persisted = try Database(url: store.root.appendingPathComponent("studio.sqlite")).load()
        precondition(persisted.draft?.audio == goodPath)
        let unchanged = try Data(contentsOf: original); precondition(unchanged == originalBytes)
        store.shutdown()
        print("Audio replacement failure retains draft, cleans failed copy and uses both channels: passed")
    }
}

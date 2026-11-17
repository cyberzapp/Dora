import Foundation
import AVFoundation

@main struct AudioPlaybackSmoke {
    @MainActor static func main() async throws {
        let folder = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("silent-transport-fixture.wav")
        let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 48000)!
        buffer.frameLength = 48000
        buffer.floatChannelData![0].initialize(repeating: 0, count: 48000)
        do { let file = try AVAudioFile(forWriting: url, settings: format.settings); try file.write(from: buffer) }
        let player = AudioPlayback(muted: true)
        player.load(path: url.path, duration: 3); player.play()
        let deadline = Date().addingTimeInterval(5)
        while player.position < 0.2 && Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        precondition(player.position >= 0.2, "Playback clock did not advance")
        player.pause(); let paused = player.position
        try await Task.sleep(for: .milliseconds(250))
        precondition(abs(player.position-paused) < 0.15, "Pause did not hold the recording position")
        player.seek(to: 1.25); player.play()
        let resumeDeadline = Date().addingTimeInterval(3)
        while player.position < 1.4 && Date() < resumeDeadline { try await Task.sleep(for: .milliseconds(100)) }
        precondition(player.position >= 1.4, "Seek/resume did not advance from the selected position")
        player.seek(to: 2.8)
        let endDeadline = Date().addingTimeInterval(3)
        while player.isPlaying && Date() < endDeadline { try await Task.sleep(for: .milliseconds(100)) }
        precondition(!player.isPlaying && abs(player.position-3) < 0.1, "Playback completion state was not delivered")
        player.play()
        let restartDeadline = Date().addingTimeInterval(3)
        while player.position < 0.1 && Date() < restartDeadline { try await Task.sleep(for: .milliseconds(100)) }
        precondition(player.position > 0 && player.position < 1, "Completed recording did not restart")
        player.load(path: url.path, duration: 3)
        try await Task.sleep(for: .milliseconds(200))
        precondition(player.position == 0 && !player.isPlaying, "Replacing the recording retained old playback state")
        player.unload()
        precondition(AudioPlayback.clock(65) == "01:05")
        print("Actual AVPlayer clock, pause, seek/resume and replacement: passed (muted silence fixture; no generated avatar media)")
    }
}

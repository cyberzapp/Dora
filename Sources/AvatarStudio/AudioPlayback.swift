import Foundation
import Combine
import AVFoundation

@MainActor final class AudioPlayback: ObservableObject {
    @Published private(set) var position = 0.0
    @Published private(set) var duration = 0.0
    @Published private(set) var isPlaying = false
    @Published private(set) var error: String?
    private var player: AVPlayer?
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?
    private var statusObserver: NSKeyValueObservation?
    private var generation = UUID()
    private let muted: Bool

    init(muted: Bool = false) { self.muted = muted }

    func load(path: String, duration: Double) {
        unload()
        guard !path.isEmpty else { return }
        self.duration = duration.isFinite ? max(0, duration) : 0
        let item = AVPlayerItem(url: URL(fileURLWithPath: path))
        let player = AVPlayer(playerItem: item); player.isMuted = muted
        self.player = player
        let token = generation
        timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.1, preferredTimescale: 600), queue: .main) { [weak self] time in
            Task { @MainActor in
                guard let self, self.generation == token, time.seconds.isFinite else { return }
                self.position = min(self.duration, max(0, time.seconds))
            }
        }
        endObserver = NotificationCenter.default.addObserver(forName: AVPlayerItem.didPlayToEndTimeNotification, object: item, queue: .main) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation == token else { return }
                self.isPlaying = false; self.position = self.duration
            }
        }
        statusObserver = item.observe(\.status, options: [.initial, .new]) { [weak self] item, _ in
            guard item.status == .failed else { return }
            let message = item.error?.localizedDescription ?? "The recording could not be played. Import it again."
            Task { @MainActor in
                guard let self, self.generation == token else { return }
                self.error = message; self.isPlaying = false
            }
        }
    }

    func play() {
        guard let player, error == nil else { return }
        if position >= duration { seek(to: 0) }
        player.play(); isPlaying = true
    }
    func pause() { player?.pause(); isPlaying = false }
    func seek(to seconds: Double) {
        guard seconds.isFinite else { return }
        position = min(duration, max(0, seconds))
        player?.seek(to: CMTime(seconds: position, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
    }
    func unload() {
        generation = UUID()
        player?.pause()
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        timeObserver = nil; endObserver = nil; statusObserver = nil; player = nil
        position = 0; duration = 0; isPlaying = false; error = nil
    }
    static func clock(_ seconds: Double) -> String {
        let value = seconds.isFinite ? Int(max(0, seconds)) : 0
        return String(format: "%02d:%02d", value/60, value%60)
    }
    deinit {
        if let timeObserver { player?.removeTimeObserver(timeObserver) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
    }
}

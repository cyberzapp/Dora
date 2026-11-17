import SwiftUI
import AVKit

struct VideoFrameImportView: View {
    let source: URL
    let save: (Double) -> Void
    @Environment(\.dismiss) private var dismiss
    @ViewState<Double> private var seconds = 0
    @ViewState<Double> private var duration = 0
    @ViewState<String?> private var error = nil
    @ViewState<CGImage?> private var preview = nil
    @ViewState<String?> private var previewError = nil
    @ViewState<Bool> private var loading = true

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Choose an appearance frame").font(.title2.bold())
            Text("Choose a clear, forward-facing moment with open eyes and a relaxed mouth. The selected frame becomes an outfit appearance; the original video is kept locally.").foregroundStyle(.secondary)
            ZStack {
                Color.black
                if let preview { Image(decorative: preview, scale: 1).resizable().scaledToFit() }
                if loading { ProgressView() }
            }.frame(minHeight: 330)
            if let error { Text(error).foregroundStyle(.red) }
            if let previewError { Text(previewError).foregroundStyle(.red) }
            if duration > 0 {
                Slider(value: $seconds, in: 0...max(0.001, duration - 0.05)) {
                    Text("Frame position")
                }.accessibilityLabel("Appearance frame position")
                Text("\(seconds, specifier: "%.2f") s of \(duration, specifier: "%.2f") s").monospacedDigit().foregroundStyle(.secondary)
            }
            HStack {
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Spacer()
                Button("Use this frame") { save(seconds); dismiss() }
                    .buttonStyle(.borderedProminent).disabled(duration <= 0 || error != nil || previewError != nil || loading)
            }
        }.padding(24).frame(width: 760, height: 590)
            .task {
                do {
                    let asset = AVURLAsset(url: source)
                    let length = try await asset.load(.duration).seconds
                    guard length.isFinite, length >= 0.1 else { throw StudioError.message("This video has no readable duration.") }
                    try Task.checkCancellation()
                    duration = length
                } catch { self.error = error.localizedDescription }
            }
            .task(id: seconds) {
                loading = true; previewError = nil
                let generator = AVAssetImageGenerator(asset: AVURLAsset(url: source))
                generator.appliesPreferredTrackTransform = true
                generator.requestedTimeToleranceBefore = .zero
                generator.requestedTimeToleranceAfter = .zero
                do {
                    let (image, _) = try await generator.image(at: CMTime(seconds: seconds, preferredTimescale: 60000))
                    try Task.checkCancellation(); preview = image; loading = false
                } catch {
                    guard !Task.isCancelled else { return }
                    previewError = "This frame could not be read. Try another position."; loading = false
                }
            }
    }
}

import Foundation
import CryptoKit
import CoreImage

/// Actual native store → real Python inference → verified output → SQLite.
/// Uses existing consenting media; never activates camera or microphone.
@main struct NativeFlashHeadSmoke {
    @MainActor static func main() async throws {
        let parent = URL(fileURLWithPath: CommandLine.arguments[1])
        let runtime = CommandLine.arguments[2]
        let photo = URL(fileURLWithPath: CommandLine.arguments[3])
        let voice = URL(fileURLWithPath: CommandLine.arguments[4])
        let testClip = CommandLine.arguments.dropFirst(5).contains("--test-clip")
        let engine: RenderEngine = CommandLine.arguments.dropFirst(5).contains("--ditto") ? .ditto : .flashHead
        let root = parent.appendingPathComponent(UUID().uuidString)
        let suite = "local.avatarstudio.flashhead-test.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        let store = StudioStore(storageRoot: root, runtimeRoot: runtime, preferences: preferences)
        let outfit = try PhotoPreparation.prepare(source: photo, directory: root.appendingPathComponent("originals"))
        let sourceDigest = SHA256.hash(data: try Data(contentsOf: photo))
        let importedDigest = SHA256.hash(data: try Data(contentsOf: URL(fileURLWithPath: outfit.original)))
        precondition(importedDigest == sourceDigest)
        var avatar = Avatar(name: "Consenting native test", consent: true)
        avatar.outfits = [outfit]; avatar.defaultOutfit = outfit.id
        store.library.avatars = [avatar]; store.selectedAvatar = avatar.id; store.selectedOutfit = outfit.id
        store.renderEngine = engine; store.stabilizeHead = engine == .flashHead
        store.projectName = "Native \(engine.label) generation test"
        store.importAudio(from: voice)
        while store.busy { try await Task.sleep(for: .milliseconds(100)) }
        guard !store.audio.isEmpty, store.error == nil else { throw StudioError.message(store.error ?? "Audio import failed") }
        let importedAudio = store.audio, fullDuration = store.audioDuration
        store.enqueue(testClip: testClip)
        let deadline = Date().addingTimeInterval(max(20 * 60, (testClip ? StudioStore.testClipSeconds : fullDuration) * 100 + 300))
        var lastDetail = ""
        while Date() < deadline {
            if let job = store.library.jobs.first {
                if job.detail != lastDetail { print(job.detail); lastDetail = job.detail }
                if ["Completed", "Failed", "Cancelled"].contains(job.status) { break }
            } else if !store.busy, let error = store.error { throw StudioError.message(error) }
            try await Task.sleep(for: .seconds(1))
        }
        guard let result = store.library.jobs.first, result.status == "Completed" else {
            store.shutdown()
            throw StudioError.message("Native real render did not complete: \(store.library.jobs.first?.detail ?? store.error ?? "timeout")")
        }
        precondition(result.engine == engine)
        precondition((result.stabilizeHead ?? false) == (engine == .flashHead))
        let source = CIImage(contentsOf: URL(fileURLWithPath: result.photo))!
        let original = CIImage(contentsOf: photo)!
        precondition(source.extent.size == original.extent.size, "Native queue downscaled the source photo")
        let directory = root.appendingPathComponent("jobs/\(result.id)")
        let request = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("request.json"))) as! [String: Any]
        let size = engine.outputSize(portrait: false)
        precondition(request["engine"] as? String == engine.rawValue)
        precondition(request["width"] as? Int == size.width && request["height"] as? Int == size.height)
        precondition(request["stabilize_head"] as? Bool == (engine == .flashHead))
        precondition(store.audio == importedAudio && store.audioDuration == fullDuration)
        if testClip {
            precondition(fullDuration > StudioStore.testClipSeconds)
            precondition(result.audio == directory.appendingPathComponent("test-clip.wav").path)
            precondition(result.name.hasSuffix(" • Test clip"))
        } else {
            precondition(result.audio == importedAudio)
        }
        let persisted = try Database(url: root.appendingPathComponent("studio.sqlite")).load()
        precondition(persisted.jobs.first?.status == "Completed" && persisted.jobs.first?.output == result.output)
        precondition(FileManager.default.fileExists(atPath: result.output))
        let finalDigest = SHA256.hash(data: try Data(contentsOf: photo))
        precondition(finalDigest == sourceDigest, "Original photo changed")
        let report: [String: Any] = ["actual_native_queue": true, "actual_inference": true,
                                   "gui_click_test": false, "quality_passed": false,
                                   "output": result.output, "job": result.id, "storage_root": root.path,
                                   "source_size": [Int(source.extent.width), Int(source.extent.height)],
                                   "sqlite_completed": true, "camera_or_microphone_activated": false,
                                   "test_clip": testClip, "original_audio_duration": fullDuration,
                                   "render_audio": result.audio]
        try JSONSerialization.data(withJSONObject: report, options: .prettyPrinted).write(to: parent.appendingPathComponent("last-result.json"), options: .atomic)
        store.shutdown()
        print("Native \(engine.label) queue → actual inference → verified export → persisted completion: passed")
    }
}

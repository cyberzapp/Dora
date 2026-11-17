import Foundation

@main struct NativeQueueSmoke {
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        let runtime = CommandLine.arguments[2]
        let suite = "local.avatarstudio.test.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        let store = StudioStore(storageRoot: root, runtimeRoot: runtime, preferences: preferences)
        store.renderEngine = .longCat
        var job = RenderJob(name: "Missing-media recovery test", photo: root.appendingPathComponent("missing-photo.png").path,
                            audio: root.appendingPathComponent("missing-audio.wav").path, motion: "", script: "",
                            sourceVideo: nil, portrait: false, engine: .longCat)
        job.status = "Failed"
        store.library.jobs.append(job); store.save(); store.retry(job)
        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            if store.library.jobs.first?.status == "Failed" { break }
            try await Task.sleep(for: .milliseconds(100))
        }
        guard let result = store.library.jobs.first else { fatalError("Job disappeared") }
        precondition(result.id == job.id, "Resume must keep the same cache directory")
        precondition(result.status == "Failed", "Real worker failure did not reach native state")
        precondition(result.detail.contains("missing"), "Worker's actionable error was lost: \(result.detail)")
        precondition(result.output.isEmpty, "Failure exposed an output")
        store.cancel(result)
        precondition(store.library.jobs.first?.status == "Failed", "Cancellation overwrote a terminal state")
        let persisted = try Database(url: root.appendingPathComponent("studio.sqlite")).load()
        precondition(persisted.jobs.first?.status == "Failed")
        store.shutdown()
        print("Native queue → actual LongCat worker → failure → persisted recovery: passed")
    }
}

import Foundation

/// Exercises crash recovery with protocol fixtures only. No video is generated.
@main struct NativeRestartRecoverySmoke {
    @MainActor static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "local.avatarstudio.restart-test.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        var library = Library()
        func fixture(_ name: String, status: String = "Rendering", stale: Bool = false,
                     external: Bool = false, empty: Bool = false) throws {
            var job = RenderJob(name: name, photo: "", audio: "", motion: "", script: "", portrait: false, engine: .flashHead)
            job.status = status
            let directory = root.appendingPathComponent("jobs/\(job.id)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let output = (external ? root : directory).appendingPathComponent("output.mov")
            try Data(empty ? [] : [0]).write(to: output)
            let attempt = UUID().uuidString
            let completion: [String: Any] = ["attempt": stale ? UUID().uuidString : attempt,
                "status": "Completed", "detail": "Protocol fixture", "output": output.path]
            for (file, value) in [("request.json", ["attempt": attempt] as [String: Any]), ("status.json", completion)] {
                try JSONSerialization.data(withJSONObject: value).write(to: directory.appendingPathComponent(file))
            }
            library.jobs.append(job)
        }
        try fixture("Current attempt")
        try fixture("Stale completion", stale: true)
        try fixture("Explicitly cancelled", status: "Cancelled")
        try fixture("Other directory", external: true)
        try fixture("Empty output", empty: true)
        try fixture("Never started", status: "Queued")
        let evidence = CommandLine.arguments.dropFirst().first.map { URL(fileURLWithPath: $0) }
        if let evidence {
            let jobDirectory = evidence.deletingLastPathComponent()
            let status = try JSONSerialization.jsonObject(with: Data(contentsOf: jobDirectory.appendingPathComponent("status.json"))) as! [String: Any]
            precondition(status["status"] as? String == "Completed" && status["output"] as? String == evidence.path)
            let engine: RenderEngine = evidence.lastPathComponent == "ditto.mov" ? .ditto : .flashHead
            var job = RenderJob(name: "Previously verified neural output", photo: "", audio: "", motion: "", script: "", portrait: false, engine: engine)
            job.status = "Rendering"
            let directory = root.appendingPathComponent("jobs/\(job.id)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let output = directory.appendingPathComponent(evidence.lastPathComponent)
            try FileManager.default.copyItem(at: evidence, to: output)
            try FileManager.default.copyItem(at: jobDirectory.appendingPathComponent("request.json"), to: directory.appendingPathComponent("request.json"))
            var copiedStatus = status; copiedStatus["output"] = output.path
            try JSONSerialization.data(withJSONObject: copiedStatus).write(to: directory.appendingPathComponent("status.json"))
            library.jobs.append(job)
        }
        let database = try Database(url: root.appendingPathComponent("studio.sqlite"))
        try database.save(library)
        let store = StudioStore(storageRoot: root, runtimeRoot: root.path, preferences: preferences)
        defer { store.shutdown() }
        let restored = store.library.jobs
        let expected = ["Completed", "Interrupted", "Cancelled", "Interrupted", "Interrupted", "Interrupted"]
        precondition(restored.map(\.status) == expected + (evidence == nil ? [] : ["Completed"]))
        precondition(restored[0].output.hasSuffix("/output.mov") && restored[0].progress == 1)
        precondition(restored[1..<expected.count].allSatisfy { $0.output.isEmpty })
        if let evidence {
            let restoredBytes = try Data(contentsOf: URL(fileURLWithPath: restored.last!.output))
            let originalBytes = try Data(contentsOf: evidence)
            precondition(restoredBytes == originalBytes)
            print("Previously verified real neural export recovered byte-for-byte: passed")
        }
        let persisted = try database.load()
        precondition(persisted.jobs.map(\.status) == restored.map(\.status))
        print("Restart recovers only the current worker's expected completion and preserves cancellation: passed")
    }
}

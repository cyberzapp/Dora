import Foundation

/// Protocol-only subprocess fixture. It performs no inference and creates no video.
@main struct NativeAttemptRecoverySmoke {
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        let runtime = root.appendingPathComponent("fixture-runtime")
        for sub in [".venv-longcat/bin", "backend"] {
            try FileManager.default.createDirectory(at: runtime.appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        let python = URL(fileURLWithPath: CommandLine.arguments[2]).resolvingSymlinksInPath()
        try FileManager.default.createSymbolicLink(at: runtime.appendingPathComponent(".venv-longcat/bin/python"), withDestinationURL: python)
        let source = """
        import json, pathlib, sys, time
        request = json.loads(pathlib.Path(sys.argv[1]).read_text())
        directory = pathlib.Path(request['directory'])
        status = directory / 'status.json'
        status.write_text(json.dumps(dict(status='Completed', progress=1, output='/nonexistent-stale-output.mov', detail='Old attempt', attempt='previous-attempt')))
        (directory / 'fixture-ready').touch()
        time.sleep(1.8)
        status.write_text(json.dumps(dict(status='Failed', progress=0, output='', detail='Protocol fixture finished; no inference', attempt=request['attempt'])))
        """
        try source.write(to: runtime.appendingPathComponent("backend/longcat_render.py"), atomically: true, encoding: .utf8)
        let suite = "local.avatarstudio.attempt-test.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let store = StudioStore(storageRoot: root.appendingPathComponent("library"), runtimeRoot: runtime.path, preferences: preferences)
        var job = RenderJob(name: "Status protocol fixture", photo: "", audio: "", motion: "", script: "", portrait: false, engine: .longCat)
        job.status = "Interrupted"
        store.library.jobs.append(job); store.save(); store.retry(job)
        let directory = store.root.appendingPathComponent("jobs/\(job.id)")
        let deadline = Date().addingTimeInterval(10)
        while !FileManager.default.fileExists(atPath: directory.appendingPathComponent("fixture-ready").path) && Date() < deadline {
            try await Task.sleep(for: .milliseconds(50))
        }
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("fixture-ready").path) else {
            store.shutdown()
            throw StudioError.message("Protocol fixture did not start. Inspect its worker.log before evaluating status recovery.")
        }
        try await Task.sleep(for: .milliseconds(700))
        precondition(store.library.jobs.first?.status == "Rendering", "Stale attempt changed the resumed job's state")
        precondition(store.library.jobs.first?.output == "", "Stale output was exposed")
        while store.library.jobs.first?.status != "Failed" && Date() < deadline {
            try await Task.sleep(for: .milliseconds(100))
        }
        precondition(store.library.jobs.first?.detail.contains("Protocol fixture finished") == true)
        let firstRequest = try Data(contentsOf: directory.appendingPathComponent("request.json"))
        let firstValue = try JSONSerialization.jsonObject(with: firstRequest) as! [String: Any]
        let oldAttempt = firstValue["attempt"] as! String
        // Let the fixture process exit before asking the real queue to resume.
        try await Task.sleep(for: .milliseconds(300))
        store.retry(store.library.jobs[0])
        let archive = directory.appendingPathComponent("attempts/\(oldAttempt)")
        precondition(FileManager.default.fileExists(atPath: archive.appendingPathComponent("worker.log").path), "Retry lost the previous log")
        let archivedRequest = try Data(contentsOf: archive.appendingPathComponent("request.json"))
        precondition(archivedRequest == firstRequest, "Retry lost its previous input snapshot")
        let secondValue = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("request.json"))) as! [String: Any]
        precondition(secondValue["attempt"] as? String != oldAttempt)
        let secondDeadline = Date().addingTimeInterval(10)
        while store.library.jobs.first?.status != "Failed" && Date() < secondDeadline { try await Task.sleep(for: .milliseconds(100)) }
        precondition(store.library.jobs.first?.status == "Failed")
        store.shutdown()
        print("Native resume ignores stale status, accepts the current worker and retains previous request/logs: passed")
    }
}

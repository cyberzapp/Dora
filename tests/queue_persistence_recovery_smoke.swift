import Foundation
import CSQLite

/// Real queue/subprocess and SQLite failure test. The worker is a protocol
/// fixture only: it never loads a model, performs inference or writes a video.
@main struct QueuePersistenceRecoverySmoke {
    static func check(_ value: Bool, _ message: String = "Queue persistence check failed") { precondition(value, message) }
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        let runtime = root.appendingPathComponent("protocol-runtime")
        for sub in [".venv-longcat/bin", "backend"] { try FileManager.default.createDirectory(at: runtime.appendingPathComponent(sub), withIntermediateDirectories: true) }
        try FileManager.default.createSymbolicLink(at: runtime.appendingPathComponent(".venv-longcat/bin/python"),
            withDestinationURL: URL(fileURLWithPath: CommandLine.arguments[2]).resolvingSymlinksInPath())
        try """
        import json, pathlib, sys, time
        request=json.loads(pathlib.Path(sys.argv[1]).read_text())
        directory=pathlib.Path(request['directory'])
        (directory/'fixture-ready').touch()
        deadline=time.monotonic()+20
        while not (directory/'fixture-finish').exists() and time.monotonic()<deadline: time.sleep(.05)
        (directory/'status.json').write_text(json.dumps(dict(status='Failed',progress=0,detail='Protocol fixture only; no inference',attempt=request['attempt'])))
        """.write(to: runtime.appendingPathComponent("backend/longcat_render.py"), atomically: true, encoding: .utf8)
        let suite = "local.avatarstudio.queue-persistence.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let store = StudioStore(storageRoot: root.appendingPathComponent("library"), runtimeRoot: runtime.path, preferences: preferences)
        defer { store.shutdown() }
        func job(_ name: String, status: String) -> RenderJob {
            var value = RenderJob(name: name, photo: "", audio: "", motion: "", script: "", portrait: false, engine: .longCat)
            value.status = status; return value
        }
        let interrupted = job("Retry persistence fixture", status: "Interrupted")
        let completed = job("Copy persistence fixture", status: "Completed")
        store.library.jobs = [interrupted,completed]; check(store.save())
        var connection: OpaquePointer?
        check(sqlite3_open(store.root.appendingPathComponent("studio.sqlite").path, &connection) == SQLITE_OK)
        defer { sqlite3_close(connection) }
        func rejectWrites() {
            check(sqlite3_exec(connection,"CREATE TRIGGER reject_queue_save BEFORE UPDATE ON library BEGIN SELECT RAISE(ABORT, 'test write failure'); END",nil,nil,nil) == SQLITE_OK)
        }
        func allowWrites() { check(sqlite3_exec(connection,"DROP TRIGGER reject_queue_save",nil,nil,nil) == SQLITE_OK) }
        let directory = store.root.appendingPathComponent("jobs/\(interrupted.id)")
        rejectWrites()
        store.retry(interrupted)
        check(store.library.jobs[0].status == "Interrupted" && store.library.jobs.count == 2)
        check(!FileManager.default.fileExists(atPath: directory.path), "Failed save started preparing a worker")
        store.retry(completed, resume: false)
        check(store.library.jobs.count == 2, "Failed copy remained in the queue")
        check(try Database(url: store.root.appendingPathComponent("studio.sqlite")).load().jobs.map(\.status) == ["Interrupted","Completed"])
        allowWrites(); store.error = nil
        store.retry(interrupted)
        let ready = directory.appendingPathComponent("fixture-ready")
        let deadline = Date().addingTimeInterval(10)
        while !FileManager.default.fileExists(atPath: ready.path) && Date() < deadline { try await Task.sleep(for: .milliseconds(50)) }
        check(FileManager.default.fileExists(atPath: ready.path), "Real fixture subprocess did not start")
        let next = job("Must wait after storage failure", status: "Queued")
        store.library.jobs.append(next); check(store.save())
        rejectWrites()
        store.cancel(store.library.jobs[0])
        check(store.library.jobs[0].status == "Cancelled" && store.error != nil)
        let cancellation = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("cancellation.json"))) as! [String:Any]
        let request = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("request.json"))) as! [String:Any]
        check(cancellation["attempt"] as? String == request["attempt"] as? String)
        check(cancellation["job"] as? String == interrupted.id)
        try await Task.sleep(for: .milliseconds(800))
        check(store.library.jobs.last?.status == "Queued", "Worker exit attempted more jobs after persistence failed")
        check(!FileManager.default.fileExists(atPath: store.root.appendingPathComponent("jobs/\(next.id)/request.json").path))
        check(try Database(url: store.root.appendingPathComponent("studio.sqlite")).load().jobs[0].status == "Rendering")
        store.shutdown(); allowWrites()
        let restarted = StudioStore(storageRoot: store.root, runtimeRoot: runtime.path, preferences: preferences)
        check(restarted.library.jobs[0].status == "Cancelled", "Restart lost cancellation after failed SQLite save")
        check(restarted.library.jobs[0].detail.contains("Recovered"))
        check(restarted.library.jobs.last?.status == "Interrupted")
        restarted.shutdown()
        // A cancellation from another attempt must never cancel this attempt.
        var saved = try Database(url: store.root.appendingPathComponent("studio.sqlite")).load()
        saved.jobs[0].status = "Rendering"
        try Database(url: store.root.appendingPathComponent("studio.sqlite")).save(saved)
        try JSONSerialization.data(withJSONObject:["job":interrupted.id,"attempt":UUID().uuidString]).write(
            to:directory.appendingPathComponent("cancellation.json"),options:.atomic)
        let stale = StudioStore(storageRoot: store.root, runtimeRoot: runtime.path, preferences: preferences)
        check(stale.library.jobs[0].status == "Interrupted", "Stale cancellation affected another attempt")
        stale.shutdown()
        print("Retry/copy save failure prevents launch; active cancellation survives failed SQLite save; later jobs wait; stale cancellation rejected: passed")
    }
}

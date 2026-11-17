import Foundation
import IOKit.pwr_mgt

/// Queue/power-management fixtures only. No model or video result is produced.
@main struct NativeRenderActivitySmoke {
    static func ownAssertions() throws -> [[String: Any]] {
        var raw: Unmanaged<CFDictionary>?
        guard IOPMCopyAssertionsByProcess(&raw) == kIOReturnSuccess, let raw else {
            throw StudioError.message("Cannot inspect native power assertions")
        }
        return (raw.takeRetainedValue() as NSDictionary)[NSNumber(value: getpid())] as? [[String: Any]] ?? []
    }
    static func blocksSleep(_ type: String) throws -> Bool {
        try ownAssertions().contains {
            $0[kIOPMAssertionTypeKey as String] as? String == type &&
            ($0[kIOPMAssertionLevelKey as String] as? NSNumber)?.intValue == Int(kIOPMAssertionLevelOn)
        }
    }
    @MainActor static func wait(_ message: String, until condition: () throws -> Bool) async throws {
        let deadline = Date().addingTimeInterval(10)
        while !(try condition()) {
            guard Date() < deadline else { throw StudioError.message(message) }
            try await Task.sleep(for: .milliseconds(50))
        }
    }
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent(UUID().uuidString)
        let runtime = root.appendingPathComponent("fixture-runtime")
        for sub in [".venv-longcat/bin", "backend"] { try FileManager.default.createDirectory(at: runtime.appendingPathComponent(sub), withIntermediateDirectories: true) }
        try FileManager.default.createSymbolicLink(at: runtime.appendingPathComponent(".venv-longcat/bin/python"), withDestinationURL: URL(fileURLWithPath: CommandLine.arguments[2]).resolvingSymlinksInPath())
        let source = """
        import json, pathlib, sys, time
        request=json.loads(pathlib.Path(sys.argv[1]).read_text())
        directory=pathlib.Path(request['directory'])
        (directory/'fixture-ready').touch()
        deadline=time.monotonic()+15
        while not (directory/'fixture-finish').exists() and time.monotonic()<deadline: time.sleep(.05)
        (directory/'status.json').write_text(json.dumps(dict(status='Failed',progress=0,output='',detail='Protocol fixture only; no inference',attempt=request['attempt'])))
        """
        try source.write(to: runtime.appendingPathComponent("backend/longcat_render.py"), atomically: true, encoding: .utf8)
        let suite = "local.avatarstudio.activity-test.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        let store = StudioStore(storageRoot: root.appendingPathComponent("library"), runtimeRoot: runtime.path, preferences: preferences)
        defer { store.shutdown(); preferences.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        guard try !blocksSleep(kIOPMAssertPreventUserIdleSystemSleep) else { throw StudioError.message("Idle queue unexpectedly prevents sleep") }
        func start(_ name: String) -> RenderJob {
            var job = RenderJob(name: name, photo: "", audio: "", motion: "", script: "", portrait: false, engine: .longCat)
            job.status = "Interrupted"; store.library.jobs.append(job); store.save(); store.retry(job)
            return job
        }
        func ready(_ job: RenderJob) async throws {
            try await wait("Worker did not acquire its idle-sleep assertion") {
                guard FileManager.default.fileExists(atPath: store.root.appendingPathComponent("jobs/\(job.id)/fixture-ready").path) else { return false }
                return try blocksSleep(kIOPMAssertPreventUserIdleSystemSleep)
            }
            guard try !blocksSleep(kIOPMAssertPreventUserIdleDisplaySleep) else { throw StudioError.message("Rendering unexpectedly prevents display sleep") }
        }
        let normal = start("Worker exit fixture"); try await ready(normal)
        FileManager.default.createFile(atPath: store.root.appendingPathComponent("jobs/\(normal.id)/fixture-finish").path, contents: Data())
        try await wait("Worker exit retained its power assertion") {
            guard store.library.jobs.first(where: { $0.id == normal.id })?.status == "Failed" else { return false }
            return try !blocksSleep(kIOPMAssertPreventUserIdleSystemSleep)
        }
        let cancelled = start("Cancellation fixture"); try await ready(cancelled); store.cancel(cancelled)
        try await wait("Cancellation retained its power assertion") { try !blocksSleep(kIOPMAssertPreventUserIdleSystemSleep) }
        let failed = RenderJob(name: "Preparation failure fixture", photo: "", audio: "", motion: "", script: "", portrait: false, engine: .longCat)
        let invalidDirectory = store.root.appendingPathComponent("jobs/\(failed.id)")
        try Data().write(to: invalidDirectory)
        var interrupted = failed; interrupted.status = "Interrupted"; store.library.jobs.append(interrupted); store.save(); store.retry(interrupted)
        guard store.library.jobs.first(where: { $0.id == failed.id })?.status == "Failed",
              try !blocksSleep(kIOPMAssertPreventUserIdleSystemSleep) else { throw StudioError.message("Preparation failure leaked a sleep assertion") }
        let shutdown = start("Shutdown fixture"); try await ready(shutdown); store.shutdown()
        try await wait("Shutdown retained its power assertion") { try !blocksSleep(kIOPMAssertPreventUserIdleSystemSleep) }
        print("Real idle-sleep assertion during queued worker; display sleep allowed; worker exit, cancellation, preparation failure and shutdown release it: passed")
    }
}

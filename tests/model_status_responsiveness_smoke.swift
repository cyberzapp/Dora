import Foundation
import Darwin

/// Real blocked filesystem read; no inference, fake output or hardware capture.
@main struct ModelStatusResponsivenessSmoke {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let bad = root.appendingPathComponent("blocked-runtime")
        let folder = bad.appendingPathComponent(".local/longcat-models/q4")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let fifo = folder.appendingPathComponent("download-manifest.json")
        guard mkfifo(fifo.path, 0o600) == 0 else { throw StudioError.message("Cannot create the blocked-read fixture") }
        let suite = "local.avatarstudio.status-io.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let store = StudioStore(storageRoot: root.appendingPathComponent("library"), runtimeRoot: bad.path, preferences: preferences)
        store.renderEngine = .longCat
        // A writer eventually releases the real open/read, even if the tested
        // implementation accidentally blocks the calling thread.
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.7) {
            if let output = try? FileHandle(forWritingTo: fifo) {
                try? output.write(contentsOf: Data("{\"files\":[]}".utf8)); try? output.close()
            }
        }
        let start = Date(); store.refreshModelStatus()
        precondition(Date().timeIntervalSince(start) < 0.1, "Model setup IO blocked the UI caller")
        try await Task.sleep(for: .milliseconds(100))
        // Recover to a different runtime while the old read is still blocked.
        store.runtime = root.appendingPathComponent("new-runtime").path
        store.renderEngine = .flashHead; store.refreshModelStatus()
        try await Task.sleep(for: .milliseconds(250))
        precondition(store.modelStatus.detail.contains("FlashHead"), "New model setup read could not complete independently")
        try await Task.sleep(for: .seconds(1))
        precondition(store.modelStatus.detail.contains("FlashHead"), "Old runtime's late result replaced current settings")
        store.shutdown()
        print("Blocked setup IO preserves UI responsiveness and ignores stale results: passed")
    }
}

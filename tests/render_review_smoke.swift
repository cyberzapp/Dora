import Foundation

/// Local review storage only. No inference and no acceptance of user media.
@main struct RenderReviewSmoke {
    @MainActor static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "local.avatarstudio.review-test.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        var completed = RenderJob(name: "Review fixture", photo: "", audio: "", motion: "", script: "", portrait: false)
        completed.status = "Completed"
        let decoder = JSONDecoder(), encoder = JSONEncoder()
        var legacy = try JSONSerialization.jsonObject(with: encoder.encode(completed)) as! [String: Any]
        legacy.removeValue(forKey: "review")
        let old = try decoder.decode(RenderJob.self, from: JSONSerialization.data(withJSONObject: legacy))
        precondition(old.review == nil && old.status == "Completed")
        let store = StudioStore(storageRoot: root, runtimeRoot: root.path, preferences: preferences)
        store.library.jobs = [old]
        store.review(old, decision: .needsRefinement)
        precondition(store.library.jobs[0].status == "Completed" && store.library.jobs[0].review?.decision == .needsRefinement)
        let db = try Database(url: root.appendingPathComponent("studio.sqlite"))
        let saved = try db.load()
        precondition(saved.jobs[0].review == store.library.jobs[0].review)
        store.review(old, decision: .looksNatural)
        let accepted = try db.load()
        precondition(accepted.jobs[0].review?.decision == .looksNatural)
        store.shutdown()
        // A fresh output must not inherit acceptance of a previous output.
        store.retry(store.library.jobs[0], resume: false)
        precondition(store.library.jobs[0].review?.decision == .looksNatural)
        precondition(store.library.jobs[1].review == nil && store.library.jobs[1].status == "Queued")
        store.review(store.library.jobs[1], decision: .looksNatural)
        precondition(store.library.jobs[1].review == nil)
        store.review(old, decision: nil)
        precondition(store.library.jobs[0].review == nil)
        print("Legacy renders remain unreviewed; decisions persist separately; new renders do not inherit approval: passed")
    }
}

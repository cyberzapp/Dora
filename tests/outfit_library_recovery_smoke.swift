import Foundation
import CryptoKit
import CSQLite

@main struct OutfitLibraryRecoverySmoke {
    static func check(_ condition: Bool, _ message: String = "Outfit recovery check failed") { precondition(condition, message) }
    static func snapshot(_ library: Library) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
        return try encoder.encode(library)
    }
    @MainActor static func wait(_ store: StudioStore) async throws {
        let deadline = Date().addingTimeInterval(30)
        while store.busy && Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
        check(!store.busy, "Photo import did not finish")
    }
    @MainActor static func main() async throws {
        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let originalHash = SHA256.hash(data: try Data(contentsOf: source))
        let root = URL(fileURLWithPath: CommandLine.arguments[2]).appendingPathComponent(UUID().uuidString)
        let suite = "local.avatarstudio.outfit-recovery.\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let store = StudioStore(storageRoot: root, preferences: preferences)
        store.consent = true
        store.createAvatar(name: "Consenting library test")
        let avatarID = store.selectedAvatar
        store.importPhoto(from: source); try await wait(store)
        check(store.error == nil && store.outfit != nil)
        var first = store.outfit!
        store.renameOutfit("  My default outfit  ")
        check(store.outfit?.name == "My default outfit")
        store.importPhoto(from: source); try await wait(store)
        check(store.library.avatars[0].defaultOutfit == first.id, "New import changed the default outfit")
        let secondID = store.selectedOutfit
        store.renameOutfit("Replacement retains this name")
        let old = store.outfit!
        let oldBytes = try Data(contentsOf: URL(fileURLWithPath: old.original))
        let oldPrepared = try Data(contentsOf: URL(fileURLWithPath: old.prepared))
        store.importPhoto(from: source, replace: true); try await wait(store)
        check(store.outfit?.name == "Replacement retains this name")
        check(store.selectedOutfit != secondID && store.outfit?.prepared != old.prepared)
        check(store.library.avatars[0].outfits.count == 2 && store.library.avatars[0].defaultOutfit == first.id)
        check(try Data(contentsOf: URL(fileURLWithPath: old.original)) == oldBytes)
        check(try Data(contentsOf: URL(fileURLWithPath: old.prepared)) == oldPrepared)
        store.selectedOutfit = first.id
        store.importPhoto(from: source, replace: true); try await wait(store)
        check(store.outfit?.name == "My default outfit" && store.selectedOutfit != first.id)
        first = store.outfit!
        check(store.library.avatars[0].defaultOutfit == first.id, "Replacing the default lost its designation")
        let originals = root.appendingPathComponent("originals")
        let filesBefore = try FileManager.default.contentsOfDirectory(atPath: originals.path)
        let bad = root.appendingPathComponent("damaged.png")
        try Data("Invalid photo fixture".utf8).write(to: bad)
        store.importPhoto(from: bad, replace: true); try await wait(store)
        check(store.error != nil && store.selectedOutfit == first.id)
        check(Set(try FileManager.default.contentsOfDirectory(atPath: originals.path)) == Set(filesBefore))

        // Switching the selected avatar while Vision processes a photo must not
        // put the completed outfit ID into the other avatar's draft.
        store.createAvatar(name: "Second consenting test avatar")
        let otherID = store.selectedAvatar
        store.selectedAvatar = avatarID; store.selectedOutfit = first.id
        store.importPhoto(from: source)
        store.selectedAvatar = otherID; store.selectedOutfit = ""
        try await wait(store)
        check(store.selectedAvatar == otherID && store.selectedOutfit.isEmpty)
        check(store.library.avatars[0].outfits.count == 3 && store.library.avatars[1].outfits.isEmpty)
        check(try Database(url: root.appendingPathComponent("studio.sqlite")).load().draft?.avatar == otherID)

        store.selectedAvatar = avatarID; store.selectedOutfit = store.library.avatars[0].outfits[1].id
        check(store.save())
        var connection: OpaquePointer?
        check(sqlite3_open(root.appendingPathComponent("studio.sqlite").path, &connection) == SQLITE_OK)
        defer { sqlite3_close(connection) }
        check(sqlite3_exec(connection, "CREATE TRIGGER reject_library_save BEFORE UPDATE ON library BEGIN SELECT RAISE(ABORT, 'test write failure'); END", nil, nil, nil) == SQLITE_OK)
        let before = try snapshot(store.library)
        store.renameOutfit("Must not appear saved")
        check(try snapshot(store.library) == before)
        let selectionBefore = store.selectedOutfit
        store.defaultOutfit()
        check(try snapshot(store.library) == before)
        store.deleteOutfit()
        check(store.selectedOutfit == selectionBefore && store.library.avatars[0].outfits.count == 3)
        store.importPhoto(from: source, replace: true); try await wait(store)
        check(store.error != nil && store.selectedOutfit == selectionBefore)
        check(try snapshot(store.library) == before)
        store.createAvatar(name: "Must not appear saved either")
        check(store.selectedAvatar == avatarID && store.selectedOutfit == selectionBefore)
        check(try snapshot(store.library) == before)
        check(sqlite3_exec(connection, "DROP TRIGGER reject_library_save", nil, nil, nil) == SQLITE_OK)
        store.error = nil
        check(store.save())
        store.shutdown()
        let restarted = StudioStore(storageRoot: root, preferences: preferences)
        check(restarted.selectedAvatar == avatarID && restarted.selectedOutfit == selectionBefore)
        check(restarted.library.avatars[0].defaultOutfit == first.id)
        check(restarted.library.avatars[0].outfits.count == 3)
        restarted.shutdown()
        check(SHA256.hash(data: try Data(contentsOf: source)) == originalHash)
        print("Real-photo import/replacement, default preservation, avatar-switch isolation, failed SQLite save rollback and restart: passed")
    }
}

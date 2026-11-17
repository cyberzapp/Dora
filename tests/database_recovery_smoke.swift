import Foundation
import CSQLite

@main struct DatabaseRecoverySmoke {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("library.sqlite")
        let database = try Database(url: path)
        let empty = try database.load()
        precondition(empty.avatars.isEmpty)
        var library = Library()
        library.avatars.append(Avatar(name: "Consented test avatar", consent: true))
        try database.save(library)
        let loaded = try database.load()
        precondition(loaded.avatars.count == 1)
        var connection: OpaquePointer?
        precondition(sqlite3_open(path.path, &connection) == SQLITE_OK)
        defer { sqlite3_close(connection) }
        precondition(sqlite3_exec(connection, "UPDATE library SET payload='unreadable preserved payload'", nil, nil, nil) == SQLITE_OK)
        do {
            _ = try database.load()
            fatalError("Corrupt library was treated as empty")
        } catch { }
        var statement: OpaquePointer?
        precondition(sqlite3_prepare_v2(connection, "SELECT payload FROM library", -1, &statement, nil) == SQLITE_OK)
        defer { sqlite3_finalize(statement) }
        precondition(sqlite3_step(statement) == SQLITE_ROW)
        precondition(String(cString: sqlite3_column_text(statement, 0)) == "unreadable preserved payload")
        print("Database round-trip and failed-load preservation: passed")
    }
}

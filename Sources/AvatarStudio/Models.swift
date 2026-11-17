import Foundation
import CSQLite

enum RenderEngine: String, Codable, CaseIterable, Sendable {
    case museTalk = "musetalk", longCat = "longcat", flashHead = "flashhead", ditto
    var label: String {
        switch self { case .museTalk: return "MuseTalk • legacy"; case .longCat: return "LongCat • evaluation"; case .flashHead: return "FlashHead • evaluation"; case .ditto: return "Ditto • evaluation" }
    }
    var supportsResume: Bool { self == .longCat || self == .flashHead }
    var usesAutomaticMotion: Bool { self != .museTalk }
    func outputSize(portrait: Bool) -> (width: Int, height: Int) {
        let landscape: (Int, Int) = self == .flashHead ? (1920, 1080) : (self == .longCat ? (768, 432) : (1280, 720))
        return portrait ? (landscape.1, landscape.0) : landscape
    }
    func formatLabel(portrait: Bool) -> String {
        let size = outputSize(portrait: portrait)
        return "\(portrait ? "Portrait" : "Landscape") • \(size.width) × \(size.height)"
    }
}

struct Outfit: Codable, Identifiable, Hashable, Sendable {
    var id = UUID().uuidString
    var name: String
    var original: String
    var prepared: String
    var guidance: String
    var sourceVideo: String? = nil
    var sourceFrameSeconds: Double? = nil
    var created = Date()
}
struct Avatar: Codable, Identifiable {
    var id = UUID().uuidString
    var name: String
    var consent: Bool
    var outfits: [Outfit] = []
    var defaultOutfit: String?
    var created = Date()
    var updated = Date()
}
enum RenderReviewDecision: String, Codable {
    case looksNatural, needsRefinement
    var label: String { self == .looksNatural ? "Looks natural • reviewed by you" : "Needs refinement • reviewed by you" }
}
struct RenderReview: Codable, Equatable {
    var decision: RenderReviewDecision
    var updated = Date()
}
struct RenderJob: Codable, Identifiable {
    var id = UUID().uuidString
    var name: String
    var photo: String
    var audio: String
    var motion: String
    var script: String
    var sourceVideo: String?
    var portrait: Bool
    var engine: RenderEngine? = nil
    var faceBounds: [Double]? = nil
    var stabilizeHead: Bool? = nil
    var review: RenderReview? = nil
    var status = "Queued"
    var progress = 0.0
    var detail = "Waiting for local worker"
    var output = ""
    var created = Date()
}
struct VideoDraft: Codable, Equatable {
    var avatar: String
    var outfit: String
    var audio: String
    var motion: String
    var sourceVideo: String
    var script: String
    var name: String
    var portrait: Bool
    var waveform: [Float]
    var audioDuration: Double
    var backgroundMode: String
    var backgroundPath: String
    var backgroundRGBA: [Double]
    var engine: RenderEngine
}
struct Library: Codable {
    var avatars: [Avatar] = []
    var jobs: [RenderJob] = []
    var motions: [String] = []
    var draft: VideoDraft? = nil
}
enum StudioError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let text) = self { return text }; return nil }
}
final class Database {
    private var db: OpaquePointer?
    init(url: URL) throws {
        guard sqlite3_open(url.path, &db) == SQLITE_OK else { throw StudioError.message("Cannot open local database") }
        try execute("PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS library (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);")
    }
    deinit { sqlite3_close(db) }
    private func execute(_ sql: String) throws {
        guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else { throw StudioError.message(String(cString: sqlite3_errmsg(db))) }
    }
    func load() throws -> Library {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT payload FROM library WHERE id=1", -1, &statement, nil) == SQLITE_OK else { throw StudioError.message("Cannot read library") }
        defer { sqlite3_finalize(statement) }
        let result = sqlite3_step(statement)
        if result == SQLITE_DONE { return Library() }
        guard result == SQLITE_ROW, let text = sqlite3_column_text(statement, 0) else {
            throw StudioError.message("Cannot read the local library: \(String(cString: sqlite3_errmsg(db)))")
        }
        return try JSONDecoder().decode(Library.self, from: Data(String(cString: text).utf8))
    }
    func save(_ library: Library) throws {
        let data = try JSONEncoder().encode(library)
        let text = String(decoding: data, as: UTF8.self)
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, "INSERT INTO library(id,payload) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", -1, &statement, nil) == SQLITE_OK else { throw StudioError.message("Cannot prepare library save") }
        defer { sqlite3_finalize(statement) }
        let result = text.withCString { pointer -> Int32 in
            sqlite3_bind_text(statement, 1, pointer, -1, nil)
            return sqlite3_step(statement)
        }
        guard result == SQLITE_DONE else { throw StudioError.message("Could not save library") }
    }
}

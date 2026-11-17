import Foundation

@main struct ModelStatusSmoke {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent(".local/longcat-models/q4")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let manifest = "{\"files\":[{\"path\":\"weight\",\"bytes\":10}]}"
        try Data(manifest.utf8).write(to: model.appendingPathComponent("download-manifest.json"))
        try Data("[0,0,1,-1]".utf8).write(to: model.appendingPathComponent("weight.ranges.json"))
        precondition(LocalModelStatus.read(runtime: root.path).fraction == 0, "Orphan range state must not count")
        try Data(repeating: 0, count: 10).write(to: model.appendingPathComponent("weight.partial"))
        precondition(LocalModelStatus.read(runtime: root.path).fraction == 1, "Deduplicate ranges and bound final range")
        try FileManager.default.moveItem(at: model.appendingPathComponent("weight.partial"), to: model.appendingPathComponent("weight"))
        let status = LocalModelStatus.read(runtime: root.path)
        precondition(status.fraction == 1)
        precondition(status.evaluation == "Video evaluation pending", "Downloaded weights do not prove generation")
        print("Native setup status: passed")
    }
}

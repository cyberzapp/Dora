import Foundation

/// Reports transferred bytes separately from trained-video validation.
struct LocalModelStatus: Sendable {
    var fraction: Double = 0
    var detail = "LongCat models have not been installed."
    var evaluation = "Video evaluation pending"
    var evaluationLog = ".local/longcat-first-render.log"

    static func read(runtime: String, engine: RenderEngine = .longCat) -> LocalModelStatus {
        if engine == .flashHead { return readFlashHead(runtime: runtime) }
        let root = URL(fileURLWithPath: runtime)
        let folder = root.appendingPathComponent(".local/longcat-models/q4")
        struct Manifest: Decodable {
            struct File: Decodable { let path: String; let bytes: Int64 }
            let files: [File]
        }
        guard let data = try? Data(contentsOf: folder.appendingPathComponent("download-manifest.json")),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else { return LocalModelStatus() }
        var total: Int64 = 0, received: Int64 = 0
        let manager = FileManager.default
        for file in manifest.files {
            guard file.bytes >= 0, !file.path.hasPrefix("/"), !file.path.split(separator: "/").contains("..") else { continue }
            total += file.bytes
            let target = folder.appendingPathComponent(file.path)
            if let attributes = try? manager.attributesOfItem(atPath: target.path),
               (attributes[.size] as? NSNumber)?.int64Value == file.bytes {
                received += file.bytes
            } else if let partial = try? manager.attributesOfItem(atPath: target.path + ".partial"),
                      (partial[.size] as? NSNumber)?.int64Value == file.bytes,
                      let ranges = try? Data(contentsOf: URL(fileURLWithPath: target.path + ".ranges.json")),
                      let completed = try? JSONDecoder().decode([Int64].self, from: ranges) {
                let block: Int64 = 8 * 1024 * 1024
                for index in Set(completed) where index >= 0 && index < (file.bytes + block - 1) / block {
                    received += min(block, file.bytes - index * block)
                }
            }
        }
        var value = LocalModelStatus()
        value.fraction = total > 0 ? min(1, Double(received) / Double(total)) : 0
        let format = ByteCountFormatter(); format.countStyle = .file
        value.detail = "\(format.string(fromByteCount: received)) of \(format.string(fromByteCount: total)) transferred"
        if manager.fileExists(atPath: folder.appendingPathComponent("download-complete.json").path), received == total {
            value.detail = "Model download complete"
        }
        let progress = root.appendingPathComponent(".local/longcat-first-render/progress.json")
        if let data = try? Data(contentsOf: progress),
           let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let stage = result["stage"] as? String {
            switch stage {
            case "waiting_for_verified_weights": value.evaluation = "First video test is waiting for the model download"
            case "waiting_for_render_memory": value.evaluation = "Waiting for another local render to finish"
            case "cancelled": value.evaluation = "Video test cancelled • saved progress retained"
            case "generated_awaiting_review": value.evaluation = "Video generated • visual review pending"
            case "failed": value.evaluation = "Video test needs attention. Open the evaluation log."
            case "denoising": value.evaluation = "Generating test video • step \(result["step"] as? Int ?? 0) of 8"
            default: value.evaluation = "Preparing first video test"
            }
        }
        return value
    }

    private static func readFlashHead(runtime: String) -> LocalModelStatus {
        struct Manifest: Decodable {
            struct File: Decodable { let path: String; let bytes: Int64 }
            let files: [File]
        }
        let root = URL(fileURLWithPath: runtime), manager = FileManager.default
        var value = LocalModelStatus(), received: Int64 = 0, total: Int64 = 0, complete = true
        value.evaluationLog = ".local/flashhead-detail-five.log"
        value.detail = "FlashHead models have not been installed."
        value.evaluation = "Visual quality is still under review"
        for (name, manifestName) in [("flashhead-models", "flashhead-pro"), ("flashhead-audio", "flashhead-audio")] {
            let folder = root.appendingPathComponent(".local/\(name)")
            let canonical = root.appendingPathComponent("backend/model_manifests/\(manifestName).json")
            let manifestPath = manager.fileExists(atPath: canonical.path) ? canonical : folder.appendingPathComponent("download-manifest.json")
            guard let data = try? Data(contentsOf: manifestPath), let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else { complete = false; continue }
            complete = complete && manager.fileExists(atPath: folder.appendingPathComponent("download-complete.json").path)
            for file in manifest.files {
                guard file.bytes >= 0, !file.path.hasPrefix("/"), !file.path.split(separator: "/").contains("..") else { complete = false; continue }
                total += file.bytes
                let target = folder.appendingPathComponent(file.path)
                let size = (try? manager.attributesOfItem(atPath: target.path)[.size] as? NSNumber)?.int64Value
                if size == file.bytes { received += file.bytes; continue }
                complete = false
                let partialSize = (try? manager.attributesOfItem(atPath: target.path + ".partial")[.size] as? NSNumber)?.int64Value
                if partialSize == file.bytes,
                   let data = try? Data(contentsOf: URL(fileURLWithPath: target.path + ".ranges.json")),
                   let ranges = try? JSONDecoder().decode([Int64].self, from: data) {
                    let block: Int64 = 8 * 1024 * 1024
                    for index in Set(ranges) where index >= 0 && index < (file.bytes + block - 1) / block {
                        received += min(block, file.bytes - index * block)
                    }
                }
            }
        }
        value.fraction = total > 0 ? min(1, Double(received) / Double(total)) : 0
        if total > 0 {
            let format = ByteCountFormatter(); format.countStyle = .file
            value.detail = complete && received == total ? "Model download complete • files verified before each render" : "\(format.string(fromByteCount: received)) of \(format.string(fromByteCount: total)) transferred"
        }
        let evidence = root.appendingPathComponent(".local/flashhead-quality-five/report.json")
        if let data = try? Data(contentsOf: evidence), let report = try? JSONSerialization.jsonObject(with: data) as? [String: Any], report["generated_video"] as? Bool == true {
            value.evaluation = "Local video generation tested • visual quality still under review"
        }
        return value
    }
}

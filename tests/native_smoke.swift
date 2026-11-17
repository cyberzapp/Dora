import Foundation
import AppKit
import ImageIO

@main struct NativeSmoke {
    static func main() throws {
        let directory = URL(fileURLWithPath: CommandLine.arguments[1])
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let db = try Database(url: directory.appendingPathComponent("test.sqlite"))
        var library = Library()
        var avatar = Avatar(name: "Suman", consent: true)
        for path in CommandLine.arguments.dropFirst(2) {
            let source = URL(fileURLWithPath: path)
            let outfit = try PhotoPreparation.prepare(source: source, directory: directory)
            let a = try Data(contentsOf: source); let b = try Data(contentsOf: URL(fileURLWithPath: outfit.original)); precondition(a == b)
            let cachedSource = CGImageSourceCreateWithURL(URL(fileURLWithPath: outfit.prepared) as CFURL, nil)!
            let cached = CGImageSourceCreateImageAtIndex(cachedSource, 0, nil)!
            let input = CIImage(contentsOf: source, options: [.applyOrientationProperty: true])!
            precondition(cached.width == Int(input.extent.width) && cached.height == Int(input.extent.height), "Preparation discarded source resolution")
            let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("\(outfit.id)-landmarks.json"))) as! [String: Any]
            precondition(metadata["version"] as? Int == 2)
            avatar.outfits.append(outfit)
            print("Prepared \(outfit.name): \(outfit.guidance)")
        }
        avatar.defaultOutfit = avatar.outfits.first?.id
        library.avatars.append(avatar)
        try db.save(library)
        let loaded = try db.load()
        precondition(loaded.avatars.first?.outfits.count == avatar.outfits.count)
        if let first = avatar.outfits.first {
            try PhotoPreparation.composite(photo: URL(fileURLWithPath: first.prepared), background: nil, color: .darkGray, output: directory.appendingPathComponent("composite.png"))
            print("Person segmentation: passed")
        }
        print("Native photo validation, unchanged originals, and SQLite round-trip: passed")
    }
}

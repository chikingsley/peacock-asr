// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MossCoreMLFixture",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "moss-coreml-fixture", targets: ["MossCoreMLFixture"])
    ],
    targets: [
        .executableTarget(name: "MossCoreMLFixture")
    ]
)

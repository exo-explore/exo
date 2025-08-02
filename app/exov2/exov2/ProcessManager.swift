import Foundation
import OSLog
import SwiftUI
import AppKit
import ServiceManagement

extension NSApplication {
    func addTerminationHandler(_ handler: @escaping () -> Void) {
        NSApp.setActivationPolicy(.accessory)
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification,
                                            object: nil,
                                            queue: .main) { _ in
            handler()
        }
    }
}

class ProcessManager: ObservableObject {
    @Published var masterProcess: Process?
    @Published var workerProcess: Process?
    @Published var masterStatus: String = "Stopped"
    @Published var workerStatus: String = "Stopped"
    @Published var isLoginItemEnabled: Bool = false
    @Published var isMasterMode: Bool = false  // Default to replica mode (false)
    
    private var masterStdout: Pipe?
    private var workerStdout: Pipe?
    private let logger = Logger(subsystem: "exolabs.exov2", category: "ProcessManager")
    
    // Add file handle properties to track them
    private var masterFileHandle: FileHandle?
    private var workerFileHandle: FileHandle?
    
    private let loginService = SMAppService.mainApp
    
    // Find uv executable in common installation paths
    private var uvPath: String? {
        let commonPaths = [
            "/usr/local/bin/uv",
            "/opt/homebrew/bin/uv",
            "/usr/bin/uv",
            "/bin/uv",
            "/Users/\(NSUserName())/.cargo/bin/uv",
            "/Users/\(NSUserName())/.local/bin/uv"
        ]
        
        for path in commonPaths {
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        
        // Try using 'which uv' command as fallback
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = ["uv"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        
        do {
            try process.run()
            process.waitUntilExit()
            
            if process.terminationStatus == 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                if let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !path.isEmpty {
                    return path
                }
            }
        } catch {
            logger.error("Failed to run 'which uv': \(error.localizedDescription)")
        }
        
        return nil
    }
    
    // Project root path - assuming the app bundle is in the project directory
    private var projectPath: URL? {
        // Get the app bundle path and navigate to the project root
        // This assumes the app is built/run from within the project directory
        guard let bundlePath = Bundle.main.bundleURL.path as String? else { return nil }
        
        // Navigate up from the app bundle to find the project root
        // Look for pyproject.toml to identify the project root
        var currentPath = URL(fileURLWithPath: bundlePath)
        while currentPath.pathComponents.count > 1 {
            let pyprojectPath = currentPath.appendingPathComponent("pyproject.toml")
            if FileManager.default.fileExists(atPath: pyprojectPath.path) {
                return currentPath
            }
            currentPath = currentPath.deletingLastPathComponent()
        }
        
        // Fallback: try to find project in common development locations
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let commonPaths = [
            "exo",
            "Projects/exo", 
            "Documents/exo",
            "Desktop/exo"
        ]
        
        for path in commonPaths {
            let projectDir = homeDir.appendingPathComponent(path)
            let pyprojectPath = projectDir.appendingPathComponent("pyproject.toml")
            if FileManager.default.fileExists(atPath: pyprojectPath.path) {
                return projectDir
            }
        }
        
        return nil
    }
    
    init() {
        // Add termination handler
        NSApplication.shared.addTerminationHandler { [weak self] in
            self?.stopAll()
        }
        
        // Check if login item is enabled
        isLoginItemEnabled = (loginService.status == .enabled)
        
        // Start processes automatically
        startMaster()
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.startWorker()
        }
    }
    
    private func handleProcessOutput(_ pipe: Pipe, processName: String) -> FileHandle {
        let fileHandle = pipe.fileHandleForReading
        fileHandle.readabilityHandler = { [weak self] handle in
            guard let data = try? handle.read(upToCount: 1024),
                  let output = String(data: data, encoding: .utf8) else {
                return
            }
            
            DispatchQueue.main.async {
                self?.logger.info("\(processName) output: \(output)")
                print("[\(processName)] \(output)")
            }
        }
        return fileHandle
    }
    
    private func cleanupProcess(process: Process?, fileHandle: FileHandle?, pipe: Pipe?) {
        // Remove readability handler
        fileHandle?.readabilityHandler = nil
        
        // Close file handles
        try? fileHandle?.close()
        try? pipe?.fileHandleForReading.close()
        try? pipe?.fileHandleForWriting.close()
        
        // Terminate process if still running
        if process?.isRunning == true {
            process?.terminate()
        }
    }
    
    func startMaster() {
        guard let projectPath = self.projectPath else {
            masterStatus = "Error: Project directory not found"
            logger.error("Could not find project directory with pyproject.toml")
            return
        }
        
        guard let uvPath = self.uvPath else {
            masterStatus = "Error: uv not found"
            logger.error("Could not find uv executable in common paths")
            return
        }
        
        // Cleanup any existing process
        cleanupProcess(process: masterProcess, fileHandle: masterFileHandle, pipe: masterStdout)
        
        masterProcess = Process()
        masterStdout = Pipe()
        
        // Use uv to run the master module
        masterProcess?.executableURL = URL(fileURLWithPath: uvPath)
        masterProcess?.arguments = ["run", "python", "-m", "master.main"]
        masterProcess?.standardOutput = masterStdout
        masterProcess?.standardError = masterStdout
        
        // Set up environment
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = projectPath.path
        
        // Set replica mode if not in master mode
        if !self.isMasterMode {
            env["EXO_RUN_AS_REPLICA"] = "1"
        }
        
        masterProcess?.environment = env
        
        // Set working directory to project root
        masterProcess?.currentDirectoryURL = projectPath
        
        // Store the file handle
        masterFileHandle = handleProcessOutput(masterStdout!, processName: "Master")
        
        do {
            logger.info("Starting master process with \(uvPath) run python -m master.main at \(projectPath.path)")
            try masterProcess?.run()
            masterStatus = "Running"
            
            masterProcess?.terminationHandler = { [weak self] process in
                DispatchQueue.main.async {
                    let status = "Stopped (exit: \(process.terminationStatus))"
                    self?.masterStatus = status
                    self?.logger.error("Master process terminated: \(status)")
                    // Cleanup on termination
                    self?.cleanupProcess(process: self?.masterProcess, 
                                      fileHandle: self?.masterFileHandle, 
                                      pipe: self?.masterStdout)
                }
            }
        } catch {
            masterStatus = "Error: \(error.localizedDescription)"
            logger.error("Failed to start master: \(error.localizedDescription)")
            cleanupProcess(process: masterProcess, fileHandle: masterFileHandle, pipe: masterStdout)
        }
    }
    
    func startWorker() {
        guard let projectPath = self.projectPath else {
            workerStatus = "Error: Project directory not found"
            logger.error("Could not find project directory with pyproject.toml")
            return
        }
        
        guard let uvPath = self.uvPath else {
            workerStatus = "Error: uv not found"
            logger.error("Could not find uv executable in common paths")
            return
        }
        
        // Cleanup any existing process
        cleanupProcess(process: workerProcess, fileHandle: workerFileHandle, pipe: workerStdout)
        
        workerProcess = Process()
        workerStdout = Pipe()
        
        // Use uv to run the worker module
        workerProcess?.executableURL = URL(fileURLWithPath: uvPath)
        workerProcess?.arguments = ["run", "python", "-m", "worker.main"]
        workerProcess?.standardOutput = workerStdout
        workerProcess?.standardError = workerStdout
        
        // Set up environment
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = projectPath.path
        workerProcess?.environment = env
        
        // Set working directory to project root
        workerProcess?.currentDirectoryURL = projectPath
        
        // Store the file handle
        workerFileHandle = handleProcessOutput(workerStdout!, processName: "Worker")
        
        do {
            logger.info("Starting worker process with \(uvPath) run python -m worker.main at \(projectPath.path)")
            try workerProcess?.run()
            workerStatus = "Running"
            
            workerProcess?.terminationHandler = { [weak self] process in
                DispatchQueue.main.async {
                    let status = "Stopped (exit: \(process.terminationStatus))"
                    self?.workerStatus = status
                    self?.logger.error("Worker process terminated: \(status)")
                    // Cleanup on termination
                    self?.cleanupProcess(process: self?.workerProcess, 
                                      fileHandle: self?.workerFileHandle, 
                                      pipe: self?.workerStdout)
                }
            }
        } catch {
            workerStatus = "Error: \(error.localizedDescription)"
            logger.error("Failed to start worker: \(error.localizedDescription)")
            cleanupProcess(process: workerProcess, fileHandle: workerFileHandle, pipe: workerStdout)
        }
    }
    
    func stopAll() {
        logger.info("Stopping all processes")
        
        // Clean up master process
        cleanupProcess(process: masterProcess, fileHandle: masterFileHandle, pipe: masterStdout)
        masterProcess = nil
        masterStdout = nil
        masterFileHandle = nil
        masterStatus = "Stopped"
        
        // Clean up worker process
        cleanupProcess(process: workerProcess, fileHandle: workerFileHandle, pipe: workerStdout)
        workerProcess = nil
        workerStdout = nil
        workerFileHandle = nil
        workerStatus = "Stopped"
    }
    
    func checkBinaries() -> Bool {
        guard let projectPath = self.projectPath else {
            logger.error("Could not find project directory")
            return false
        }
        
        guard let uvPath = self.uvPath else {
            logger.error("Could not find uv executable")
            return false
        }
        
        let fileManager = FileManager.default
        let pyprojectPath = projectPath.appendingPathComponent("pyproject.toml").path
        let masterPath = projectPath.appendingPathComponent("master/main.py").path
        let workerPath = projectPath.appendingPathComponent("worker/main.py").path
        
        let uvExists = fileManager.fileExists(atPath: uvPath)
        let pyprojectExists = fileManager.fileExists(atPath: pyprojectPath)
        let masterExists = fileManager.fileExists(atPath: masterPath)
        let workerExists = fileManager.fileExists(atPath: workerPath)
        
        if !uvExists {
            logger.error("uv not found at \(uvPath)")
        }
        if !pyprojectExists {
            logger.error("pyproject.toml not found at \(pyprojectPath)")
        }
        if !masterExists {
            logger.error("master/main.py not found at \(masterPath)")
        }
        if !workerExists {
            logger.error("worker/main.py not found at \(workerPath)")
        }
        
        return uvExists && pyprojectExists && masterExists && workerExists
    }
    
    func toggleLoginItem() {
        do {
            if isLoginItemEnabled {
                try loginService.unregister()
            } else {
                try loginService.register()
            }
            isLoginItemEnabled = (loginService.status == .enabled)
        } catch {
            logger.error("Failed to toggle login item: \(error.localizedDescription)")
        }
    }
    
    func toggleMasterMode() {
        isMasterMode.toggle()
        logger.info("Toggling master mode to: \(self.isMasterMode ? "Master" : "Replica")")
        
        // Restart master process with new mode
        if masterProcess?.isRunning == true {
            // Clean up current master process
            cleanupProcess(process: masterProcess, fileHandle: masterFileHandle, pipe: masterStdout)
            masterProcess = nil
            masterStdout = nil
            masterFileHandle = nil
            masterStatus = "Stopped"
            
            // Start master with new mode after a brief delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.startMaster()
            }
        }
    }
} 
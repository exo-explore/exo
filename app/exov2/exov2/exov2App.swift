//
//  exov2App.swift
//  exov2
//
//  Created by Sami Khan on 2025-07-27.
//

import SwiftUI
import AppKit
import Foundation
import OSLog
import ServiceManagement

@main
struct exov2App: App {
    @StateObject private var processManager = ProcessManager()
    
    private func resizedMenuBarIcon(named: String, size: CGFloat = 18.0) -> NSImage? {
        guard let original = NSImage(named: named) else {
            print("Failed to load image named: \(named)")
            return nil
        }
        
        let resized = NSImage(size: NSSize(width: size, height: size), flipped: false) { rect in
            NSGraphicsContext.current?.imageInterpolation = .high
            original.draw(in: rect)
            return true
        }
        
        resized.isTemplate = false
        resized.size = NSSize(width: size, height: size)
        return resized
    }
    
    var body: some Scene {
        MenuBarExtra {
            MenuBarView(processManager: processManager)
        } label: {
            if let resizedImage = resizedMenuBarIcon(named: "menubar-icon") {
                Image(nsImage: resizedImage)
                    .opacity(processManager.masterStatus == "Running" ? 1.0 : 0.5)
            }
        }
        .menuBarExtraStyle(.window)
    }
}

struct MenuBarView: View {
    @ObservedObject var processManager: ProcessManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            StatusSection(processManager: processManager)
            
            Divider()
            
            Toggle("Launch at Login", isOn: Binding(
                get: { processManager.isLoginItemEnabled },
                set: { _ in processManager.toggleLoginItem() }
            ))
            .padding(.horizontal)
            
            Toggle("Is Master?", isOn: Binding(
                get: { processManager.isMasterMode },
                set: { _ in processManager.toggleMasterMode() }
            ))
            .padding(.horizontal)
            
            Divider()
            
            Button("Quit") {
                NSApplication.shared.terminate(nil)
            }
        }
        .padding()
        .frame(width: 250)
        .onAppear {
            if !processManager.checkBinaries() {
                showEnvironmentError()
            }
        }
    }
    
    private func showEnvironmentError() {
        let alert = NSAlert()
        alert.messageText = "Python Environment Error"
        alert.informativeText = "Could not find the required Python environment, uv, or project files. Please ensure uv is installed and the project directory is accessible."
        alert.alertStyle = .critical
        alert.addButton(withTitle: "OK")
        alert.runModal()
        NSApplication.shared.terminate(nil)
    }
}

struct StatusSection: View {
    @ObservedObject var processManager: ProcessManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Master:")
                    .bold()
                Text(processManager.masterStatus)
                    .foregroundColor(processManager.masterStatus == "Running" ? .green : .red)
            }
            
            HStack {
                Text("Worker:")
                    .bold()
                Text(processManager.workerStatus)
                    .foregroundColor(processManager.workerStatus == "Running" ? .green : .red)
            }
        }
    }
}

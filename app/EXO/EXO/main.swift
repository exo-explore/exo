//
//  main.swift
//  EXO
//
//  Created by Jake Hillion on 2026-02-03.
//

import Foundation

/// Command line options for the EXO app
enum CLICommand {
    case install
    case uninstall
    case help
    case none
}

/// Parse command line arguments to determine the CLI command
func parseArguments() -> CLICommand {
    let args = CommandLine.arguments
    if args.contains("--help") || args.contains("-h") {
        return .help
    }
    if args.contains("--install") {
        return .install
    }
    if args.contains("--uninstall") {
        return .uninstall
    }
    return .none
}

/// Print usage information
func printUsage() {
    let programName = (CommandLine.arguments.first as NSString?)?.lastPathComponent ?? "EXO"
    print(
        """
        Usage: \(programName) [OPTIONS]

        Options:
          --install     Install EXO network configuration (requires root)
          --uninstall   Uninstall EXO network configuration (requires root)
          --help, -h    Show this help message

        When run without options, starts the normal GUI application.

        Examples:
          sudo \(programName) --install    Install network components as root
          sudo \(programName) --uninstall  Remove network components as root
        """)
}

/// Check if running as root
func isRunningAsRoot() -> Bool {
    return getuid() == 0
}

// Main entry point
let command = parseArguments()

switch command {
case .help:
    printUsage()
    exit(0)

case .install:
    if !isRunningAsRoot() {
        fputs("Error: --install requires root privileges. Run with sudo.\n", stderr)
        exit(1)
    }
    let success = NetworkSetupHelper.installDirectly()
    exit(success ? 0 : 1)

case .uninstall:
    if !isRunningAsRoot() {
        fputs("Error: --uninstall requires root privileges. Run with sudo.\n", stderr)
        exit(1)
    }
    let success = NetworkSetupHelper.uninstallDirectly()
    exit(success ? 0 : 1)

case .none:
    // Start normal GUI application
    EXOApp.main()
}

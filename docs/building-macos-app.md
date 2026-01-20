# Building the macOS App

This guide explains how to build the exo macOS app from source. The app provides a native macOS interface for running exo in the background with system integration.

## Prerequisites

Before building the app, ensure you have the following tools installed:

### Required Tools

- **Xcode 26.2 or later**: Download from the Mac App Store or [Apple Developer](https://developer.apple.com/xcode/)
- **Xcode Command Line Tools**: Install with `xcode-select --install`
- **Homebrew**: Package manager for macOS dependencies
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- **Node.js**: For building the dashboard (version 18 or higher)
- **UV**: Python dependency manager
- **macmon**: Hardware monitoring tool for Apple Silicon
- **Rust**: For building Rust bindings (nightly toolchain)

Install the required packages:

```bash
# Install Homebrew packages
brew install uv macmon node

# Install Rust with nightly toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup toolchain install nightly
```

### Verify Xcode Installation

Ensure Xcode is properly configured:

```bash
# Select Xcode 26.2
sudo xcode-select -s /Applications/Xcode_26.2.app

# Verify Metal toolchain is available
xcrun -f metal
```

## Build Steps

### 1. Clone the Repository

```bash
git clone https://github.com/exo-explore/exo
cd exo
```

### 2. Build the Dashboard

The dashboard is a web interface embedded in the app:

```bash
cd dashboard
npm install
npm run build
cd ..
```

This creates the dashboard build artifacts in `dashboard/build/`.

### 3. Build the PyInstaller Bundle

The Python backend is packaged using PyInstaller:

```bash
# Install Python dependencies
uv sync --locked

# Build the PyInstaller bundle
uv run pyinstaller packaging/pyinstaller/exo.spec
```

This creates the bundled Python runtime in `dist/exo/`.

### 4. Build the Swift App

Open the Xcode project and build the app:

```bash
cd app/EXO
xcodebuild clean build \
  -scheme EXO \
  -configuration Release \
  -derivedDataPath build
```

The built app will be located at `build/Build/Products/Release/EXO.app`.

### 5. Inject the PyInstaller Runtime

The Python runtime needs to be embedded in the app bundle:

```bash
# From the exo root directory
rm -rf app/EXO/build/Build/Products/Release/EXO.app/Contents/Resources/exo
mkdir -p app/EXO/build/Build/Products/Release/EXO.app/Contents/Resources
cp -R dist/exo app/EXO/build/Build/Products/Release/EXO.app/Contents/Resources/exo
```

### 6. Run the App

You can now run the app from the build directory:

```bash
open app/EXO/build/Build/Products/Release/EXO.app
```

## Development Workflow

For development, you can use Xcode directly:

1. Open `app/EXO/EXO.xcodeproj` in Xcode
2. Build and run the app using Xcode's build system (⌘R)
3. The app will look for the Python runtime in these locations (in order):
   - `$EXO_RUNTIME_DIR` environment variable
   - `EXO.app/Contents/Resources/exo` (bundled runtime)
   - `dist/exo` (development build)

For faster iteration during development:

```bash
# Rebuild only the Python runtime
uv run pyinstaller packaging/pyinstaller/exo.spec

# Rebuild only the dashboard
cd dashboard && npm run build && cd ..
```

## App Structure

### Swift Source Code

The Swift app source code is located in `app/EXO/EXO/`:

- **EXOApp.swift**: Main app entry point
- **ExoProcessController.swift**: Manages the Python backend process
- **Views/**: SwiftUI views for the app interface
- **ViewModels/**: View models for managing app state
- **Services/**: Service classes for app functionality
- **Models/**: Data models

### Python Backend Integration

The app launches the Python backend as a subprocess:

1. The `ExoProcessController` locates the bundled Python runtime
2. It sets up the environment (PATH, EXO_RUNTIME_DIR, EXO_LIBP2P_NAMESPACE)
3. It launches the `exo` executable from the runtime directory
4. The app monitors the process and handles termination

### Dashboard Integration

The dashboard is built as static files and embedded in the app bundle. The Python backend serves these files through its web server at `http://localhost:52415`.

## Code Signing and Distribution

### Development Signing

For local development, Xcode will automatically sign the app with your development certificate.

### Distribution Signing

For distribution, you need:

1. **Apple Developer Account**: Required for code signing and notarization
2. **Developer ID Application Certificate**: For signing the app
3. **Provisioning Profile**: For app capabilities
4. **Notarization**: Required for distribution outside the Mac App Store

The GitHub Actions workflow (`.github/workflows/build-app.yml`) shows the complete process for building, signing, and notarizing the app for distribution.

### Creating a DMG

To create a distributable DMG:

```bash
cd app/EXO/build/Build/Products/Release
mkdir -p dmg-root
cp -R EXO.app dmg-root/
ln -s /Applications dmg-root/Applications
hdiutil create -volname "EXO" -srcfolder dmg-root -ov -format UDZO EXO.dmg
```

## Troubleshooting

### Build Failures

**Metal toolchain not found:**
```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install

# Verify Metal is available
xcrun -f metal
```

**PyInstaller bundle fails:**
```bash
# Ensure all dependencies are installed
uv sync --locked

# Check that macmon is available
which macmon
```

**Dashboard build fails:**
```bash
# Ensure Node.js version is 18 or higher
node --version

# Clean and rebuild
cd dashboard
rm -rf node_modules build
npm install
npm run build
```

### Runtime Issues

**App fails to launch Python backend:**

Check the app logs for errors:
```bash
log show --predicate 'subsystem == "net.exolabs.EXO"' --last 5m
```

**Python runtime not found:**

Ensure the PyInstaller bundle is properly injected:
```bash
ls -la app/EXO/build/Build/Products/Release/EXO.app/Contents/Resources/exo
```

## CI/CD

The project uses GitHub Actions for automated builds. See `.github/workflows/build-app.yml` for the complete CI/CD pipeline, which includes:

1. Building the dashboard
2. Building the PyInstaller bundle
3. Building the Swift app
4. Code signing and notarization
5. Creating and uploading the DMG
6. Publishing releases

## Additional Resources

- [Xcode Documentation](https://developer.apple.com/documentation/xcode)
- [PyInstaller Documentation](https://pyinstaller.org/)
- [Code Signing Guide](https://developer.apple.com/support/code-signing/)
- [Notarization Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)

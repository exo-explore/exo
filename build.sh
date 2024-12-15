#!/bin/bash
set -e

# Configuration
VERSION="0.1.0"
APP_NAME="exo"
DIST_DIR="dist"
PACKAGE_NAME="${APP_NAME}-${VERSION}-darwin-arm64"

# 1. Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist

# 2. Run PyInstaller
echo "Building with PyInstaller..."
pyinstaller exo.spec

# 3. Create a clean distribution directory
echo "Creating distribution package..."
mkdir -p "${DIST_DIR}/${PACKAGE_NAME}"
cp -r "dist/${APP_NAME}/"* "${DIST_DIR}/${PACKAGE_NAME}/"

# 4. Create ZIP file
echo "Creating ZIP archive..."
cd "${DIST_DIR}"
zip -r "${PACKAGE_NAME}.zip" "${PACKAGE_NAME}"
cd ..

# 5. Calculate SHA256
echo "Calculating SHA256..."
SHA256=$(shasum -a 256 "${DIST_DIR}/${PACKAGE_NAME}.zip" | cut -d' ' -f1)

# 6. Generate Homebrew Cask formula
echo "Generating Homebrew formula..."
cat > Formula/exo.rb << EOL
cask "exo" do
  version "${VERSION}"
  sha256 "${SHA256}"

  url "https://github.com/sethburkart123/exo/releases/download/test/exo-0.1.0-darwin-arm64.zip"
  name "Exo"
  desc "MLX-powered AI assistant"
  homepage "https://github.com/exo-explorer/exo"

  depends_on macos: ">= :ventura"
  depends_on arch: :arm64

  binary "#{staged_path}/exo-${VERSION}-darwin-arm64/exo"

  postflight do
    set_permissions "#{staged_path}/exo-${VERSION}-darwin-arm64/exo", "0755"
  end
end
EOL

echo "Done! Package created at: ${DIST_DIR}/${PACKAGE_NAME}.zip"
echo "SHA256: ${SHA256}"
echo ""
echo "Next steps:"
echo "1. Upload ${PACKAGE_NAME}.zip to GitHub releases"
echo "2. Update the URL in the formula with your actual GitHub repository"
echo "3. Test the formula locally with: brew install --cask ./Formula/exo.rb"
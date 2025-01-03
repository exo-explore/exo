cask "exo" do
  version "0.1.0"
  sha256 "57fc9b838688a4dbd4842db4a96888f7627d5df16fd633bf2401340a7388cba6"

  url "http://localhost:8000/exo-0.1.0-darwin-arm64.zip"
  name "Exo"
  desc "MLX-powered AI assistant"
  homepage "https://github.com/exo-explorer/exo"

  depends_on macos: ">= :ventura"
  depends_on arch: :arm64

  binary "#{staged_path}/exo-0.1.0-darwin-arm64/exo"

  postflight do
    set_permissions "#{staged_path}/exo-0.1.0-darwin-arm64/exo", "0755"
  end
end
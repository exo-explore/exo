# PowerShell script to compile gRPC protobuf files

Write-Host "Compiling gRPC protobuf files..." -ForegroundColor Blue

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Virtual environment not detected. Activating..." -ForegroundColor Yellow
    if (Test-Path "venv-windows\Scripts\Activate.ps1") {
        .\venv-windows\Scripts\Activate.ps1
    } else {
        Write-Host "Virtual environment not found. Please run install.ps1 first." -ForegroundColor Red
        exit 1
    }
}

# Change to the grpc directory
Push-Location "exo\networking\grpc"

try {
    # Compile protobuf files
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. node_service.proto

    # Fix import in generated file (Windows equivalent of sed)
    $filePath = "node_service_pb2_grpc.py"
    if (Test-Path $filePath) {
        $content = Get-Content $filePath -Raw
        $content = $content -replace "import node_service_pb2", "from . import node_service_pb2"
        Set-Content $filePath $content
        Write-Host "Fixed import in $filePath" -ForegroundColor Green
    } else {
        Write-Host "Generated file $filePath not found" -ForegroundColor Red
        exit 1
    }

    Write-Host "gRPC compilation completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error during gRPC compilation: $_" -ForegroundColor Red
    exit 1
} finally {
    # Return to original directory
    Pop-Location
}

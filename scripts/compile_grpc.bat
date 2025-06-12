@echo off
REM Windows batch script to compile gRPC protobuf files

echo Compiling gRPC protobuf files...

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Virtual environment not detected. Activating...
    if exist "venv-windows\Scripts\activate.bat" (
        call venv-windows\Scripts\activate.bat
    ) else (
        echo Virtual environment not found. Please run install.bat first.
        pause
        exit /b 1
    )
)

REM Change to the grpc directory
pushd exo\networking\grpc

REM Compile protobuf files
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. node_service.proto
if %errorlevel% neq 0 (
    echo Error during gRPC compilation
    popd
    pause
    exit /b 1
)

REM Fix import in generated file (Windows equivalent of sed)
if exist "node_service_pb2_grpc.py" (
    powershell -Command "(Get-Content 'node_service_pb2_grpc.py') -replace 'import node_service_pb2', 'from . import node_service_pb2' | Set-Content 'node_service_pb2_grpc.py'"
    echo Fixed import in node_service_pb2_grpc.py
) else (
    echo Generated file node_service_pb2_grpc.py not found
    popd
    pause
    exit /b 1
)

REM Return to original directory
popd

echo gRPC compilation completed successfully!
pause

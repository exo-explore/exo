#!/usr/bin/env bash
source ./install.sh
pushd exo/networking/grpc
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. node_service.proto
sed -i '' "s/import\ node_service_pb2/from . &/" node_service_pb2_grpc.py
popd


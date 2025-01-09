#!/usr/bin/env bash

echo "Starting node 1"
DEBUG_DISCOVERY=7 DEBUG=7 python3 main.py --node-id "node1" --listen-port 5678 --broadcast-port 5679 --chatgpt-api-port 52415 --chatgpt-api-response-timeout 900 > output1.log 2>&1 &
PID1=$!
echo "Started node 1 PID: $PID1"
echo "Starting node 2"
DEBUG_DISCOVERY=7 DEBUG=7 python3 main.py --node-id "node2" --listen-port 5679 --broadcast-port 5678 --chatgpt-api-port 8001 --chatgpt-api-response-timeout 900 > output2.log 2>&1 &
PID2=$!
echo "Started node 2 PID: $PID2"
sleep 5
kill $PID2
sleep 5
echo "Starting node 2 again..."
DEBUG_DISCOVERY=7 DEBUG=7 python3 main.py --node-id "node2" --listen-port 5679 --broadcast-port 5678 --chatgpt-api-port 8001 --chatgpt-api-response-timeout 900 > output3.log 2>&1 &
PID2=$!
sleep 5
echo "Killing nodes and ending test..."
kill $PID1
kill $PID2
echo "Test complete."
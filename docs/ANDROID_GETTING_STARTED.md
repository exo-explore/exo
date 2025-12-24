# EXO on Android: Getting Started Guide

Run distributed LLM inference across multiple Android devices using EXO and llama.cpp RPC.

## Prerequisites

- Android device(s) with **Termux** installed from [F-Droid](https://f-droid.org/en/packages/com.termux/)
- All devices on the **same WiFi network**
- At least **4GB RAM** per device recommended
- Storage space for models (~500MB for small models, larger for bigger ones)

---

## Quick Install (One Command)

After setting up Termux basics, run:

```bash
curl -fsSL https://raw.githubusercontent.com/lukewrightmain/exo/main/scripts/install_exo-termux.sh | bash
```

---

## Full Step-by-Step Installation

### Step 1: Update Termux Packages

Open Termux and run:

```bash
pkg update
```

```bash
pkg upgrade
```

Press `Y` when prompted to confirm updates.

### Step 2: Select Package Mirrors

Run the repository selector to choose faster mirrors near you:

```bash
termux-change-repo
```

- Use arrow keys to navigate
- Press **Space** to select your local region's servers
- Press **Enter** to confirm

### Step 3: Setup Storage Access

Grant Termux access to device storage:

```bash
termux-setup-storage
```

A permission dialog will appear - tap **Allow**.

### Step 4: Install EXO

Run the installer script:

```bash
curl -fsSL https://raw.githubusercontent.com/lukewrightmain/exo/main/scripts/install_exo-termux.sh | bash
```

This will:
- Install Python, Rust, and required dependencies
- Clone the EXO repository
- Build llama.cpp with RPC support
- Configure your environment

**This takes 10-20 minutes** depending on your device.

### Step 5: Start EXO

After installation completes:

```bash
cd ~/exo-termux
python -m exo
```

You should see:
```
Node elected Master
API unpaused
```

---

## Accessing the Dashboard

### Find Your Device's IP

In the EXO logs, look for lines like:
```
External IPs for RPC: 10.99.0.14:60000
```

Or run:
```bash
ifconfig wlan0 | grep "inet "
```

### Open the Dashboard

On any device (phone, tablet, computer) on the same network, open a browser and go to:

```
http://<DEVICE_IP>:52415
```

Example: `http://10.99.0.14:52415`

---

## Running on Multiple Devices

### Setup Each Device

1. Install EXO on each Android device using the steps above
2. Connect all devices to the **same WiFi network**
3. Start EXO on each device: `python -m exo`

### Cluster Formation

Devices automatically discover each other! You'll see:
```
Subscribed to topics: connection_messages, election_messages...
Node elected Master  (on one device)
Started worker process  (on other devices)
```

### Launch a Model

1. Open the dashboard from any device
2. Select a model (e.g., `Qwen 2.5 0.5B`)
3. Set **Minimum Nodes** to match your device count
4. Select **Tensor** sharding and **llama.cpp** instance type
5. Click the model to **Launch**

The model will:
- Download to all devices (if not already present)
- Distribute across all devices via RPC
- Show **READY** when loaded

### Chat!

Type a message in the chat box and hit **Send**. All devices work together to generate the response!

---

## Troubleshooting

### "No devices found"
- Ensure all devices are on the same WiFi network
- Check that no firewall is blocking ports 52415 and 60000
- Restart EXO on all devices

### "Model loading timeout"
- Distributed loading takes 2-3 minutes - be patient!
- Check the terminal logs for progress
- Ensure sufficient RAM on all devices

### "Connection refused" errors
- Make sure rpc-server is running on worker devices
- Verify the IPs shown in logs are correct
- Try restarting EXO on all devices

### Rebuild llama.cpp (if needed)

```bash
cd ~/llama.cpp
rm -rf build
cmake -B build -DGGML_RPC=ON -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j4
```

---

## Tips for Best Performance

1. **Keep devices plugged in** - Inference uses significant power
2. **Use similar devices** - Load balancing works best with comparable hardware
3. **Close other apps** - Free up RAM for the model
4. **Stay on WiFi** - Mobile data won't work for device discovery
5. **Start with small models** - Qwen 2.5 0.5B is great for testing

---

## Next Steps

- Read [How Distributed Inference Works](./DISTRIBUTED_LLAMACPP_ARCHITECTURE.md)
- Try larger models with more devices
- Experiment with different tensor split configurations

Happy inferencing! 🚀


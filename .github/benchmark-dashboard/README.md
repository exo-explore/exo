# EXO Benchmark Dashboard

A fully self-contained, browser-based dashboard for tracking EXO benchmark performance over time.

## Features

- 📊 **Success Rate Tracking**: Monitor cluster reliability across commits
- ⚡ **Response Time Analysis**: Track average request completion times  
- 🎯 **Throughput Metrics**: Tokens per second visualization
- 📈 **Request Distribution**: Success/failure breakdown over time
- 🔄 **Auto-Refresh**: Updates every 60 seconds
- 📺 **TV-Ready**: Large, clear visualizations perfect for display
- 🔐 **Secure**: Credentials stored in browser localStorage only
- 🌐 **No Backend**: Directly accesses S3 from the browser

## Quick Start

### Option 1: Direct File Access (Simplest)

Just open the HTML file directly in your browser:

```bash
open .github/benchmark-dashboard/index.html
```

Then click "Configure AWS Credentials" and enter your keys.

### Option 2: URL Parameters (For Quick Setup)

```bash
# Serve with credentials in URL (they'll be moved to localStorage)
open ".github/benchmark-dashboard/index.html?accessKey=YOUR_KEY&secretKey=YOUR_SECRET&region=us-east-1"
```

The credentials will be saved to localStorage and removed from the URL immediately.

### Option 3: Simple HTTP Server

```bash
# From repo root
python3 -m http.server 8080

# Then open: http://localhost:8080/.github/benchmark-dashboard/
```

## AWS Credentials

The dashboard needs read-only access to the `exo-benchmark-results` S3 bucket.

### Required IAM Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::exo-benchmark-results",
        "arn:aws:s3:::exo-benchmark-results/*"
      ]
    }
  ]
}
```

### Security Notes

- ✅ Credentials stored in browser `localStorage` only
- ✅ Never sent to any server (except AWS)
- ✅ All S3 access happens client-side
- ✅ Use read-only IAM credentials
- ⚠️ Don't commit credentials to git
- ⚠️ Use a dedicated read-only IAM user

## TV/Kiosk Mode

For permanent display on a TV:

### macOS
```bash
open -a "Google Chrome" --args --kiosk ".github/benchmark-dashboard/index.html"
```

### Linux
```bash
chromium-browser --kiosk --app="file://$(pwd)/.github/benchmark-dashboard/index.html"
```

### Auto-start on Boot

Create a simple startup script:

```bash
#!/bin/bash
# /usr/local/bin/start-benchmark-dashboard.sh

cd /path/to/exo
python3 -m http.server 8080 &
sleep 2
chromium-browser --kiosk http://localhost:8080/.github/benchmark-dashboard/
```

## Data Displayed

### Summary Cards
- **Latest Success Rate**: Most recent benchmark success percentage with trend
- **Avg Response Time**: Latest average response time in ms with trend
- **Total Benchmarks**: Count of all benchmarks run
- **Active Configurations**: Number of unique benchmark configs

### Charts
1. **Success Rate Over Time**: Line chart showing reliability trends
2. **Average Response Time**: Performance over time (lower is better)
3. **Throughput**: Tokens/second metric (higher is better)
4. **Request Distribution**: Stacked bar chart of successes/failures

## How It Works

1. **Loads AWS SDK**: Uses AWS SDK for JavaScript (browser version)
2. **Lists S3 Objects**: Fetches all files from `s3://exo-benchmark-results/bench/`
3. **Downloads Results**: Fetches each JSON result file
4. **Parses & Visualizes**: Uses Chart.js to create interactive charts
5. **Auto-Refreshes**: Polls S3 every 60 seconds for new results

## Customization

To modify the dashboard:

1. Edit `index.html` 
2. Adjust `REFRESH_INTERVAL` for different polling frequency
3. Modify chart colors/styles in the Chart.js configuration
4. Add new metrics by extending the results parsing

## Troubleshooting

**"AWS credentials not configured"**
- Click "Configure AWS Credentials" and enter your keys

**"Error loading benchmark data"**
- Check AWS credentials are correct
- Verify S3 bucket name is `exo-benchmark-results`
- Ensure IAM user has read permissions
- Check browser console for detailed errors

**"No benchmark results found"**
- Wait for benchmark workflows to run
- Verify results are being uploaded to S3
- Check S3 bucket has files in `bench/` prefix

**Charts not updating**
- Check browser console for errors
- Verify network connectivity to S3
- Try refreshing the page manually


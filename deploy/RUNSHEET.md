# Pi 5 Deployment Runsheet

This runsheet outlines the step-by-step process for deploying the speech-enhancement model to the Raspberry Pi 5.

## 1. Boot and Prepare Pi
- Boot the Raspberry Pi 5.
- Ensure the Pi is fully powered and disconnected from the internet (offline environment).

## 2. Transfer Deployment Bundle
- Connect the USB drive containing the `deploy/` directory to the Pi.
- Copy the entire `deploy/` directory to the Pi's local storage:
  ```bash
  cp -r /path/to/usb/deploy /home/pi/deploy
  cd /home/pi/
  ```

## 3. Install Offline Wheels
- The bundle includes aarch64-compatible wheels for all necessary packages.
- Install them using the provided offline wheels:
  ```bash
  python3 -m pip install deploy/wheels/*.whl --no-index --find-links deploy/wheels/
  ```

## 4. Run Self-Check
- Run the self-check script to verify the installation, confirm the hardware, run self-tests, and benchmark the latency:
  ```bash
  python3 deploy/selfcheck.py
  ```
- **Check the Output:** Ensure the script completes and prints `GO` at the bottom.
  - If it prints `NO-GO`, check the console output to identify which test failed (e.g. failed imports, self-tests, or missing files).
- The script will also output the median and worst-case execution time across 1, 2, and 4 threads. Note these latencies for the field trial.

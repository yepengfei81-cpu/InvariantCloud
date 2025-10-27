# InvariantCloud: A Globally Invariant, Uniquely Indexed Point Cloud Framework for Robust 6DoF Tactile Pose Tracking

This repository contains the implementation of **InvariantCloud**, a globally invariant, uniquely indexed point cloud framework designed for robust 6DoF tactile pose tracking using vision-based tactile sensors.

## System Requirements

- **Operating System:** Ubuntu 20.04 (tested)
- **Sensor Hardware:** GelSight Mini with marker-patterned gel
- **Python Version:** >= 3.9
- **Acknowledgment:** This work is inspired and supported by the [NormalFlow](https://github.com/joehjhuang/normalflow) project.

## Installation

### 1. Download GelSight SDK

Clone the GelSight SDK repository:

```bash
git clone https://github.com/joehjhuang/gs_sdk.git
```

### 2. Create Conda Environment

Create and activate a new conda environment:

```bash
conda create -n InvariantCloud python=3.9
conda activate InvariantCloud
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure SDK Path

Open `realtime_object_tracking.py` and modify the `gs_sdk_path` variable to point to your downloaded gs_sdk directory:

```bash
gs_sdk_path = "/home/ypf/gs_sdk"  # Change this to your path
```

## Usage

### 1. Connect your GelSight Mini sensor to your computer.

### 2. Run the tracking demo:

```bash
python demos/realtime_object_tracking.py
```

### 3. The system will:
- **Collect background images automatically**
- **Wait for contact with an object** 
- **Start tracking the object's 6DoF pose in real-time**

## Limitations

The current implementation works best with:

✅ Objects with noticeable surface curvature (e.g., eggs, pencils)
✅ Non-circular or irregular shapes

The tracking may be challenging for:

❌ Highly regular spheres
❌ Flat surfaces with minimal curvature




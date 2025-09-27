# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
# Option A: Use the installation script (recommended)
./install.sh

# Option B: Manual installation
pip install -r requirements.txt
```

### 2. Run the System
```bash
# From the project root directory
python src/main_predictor.py
```

### 3. Use the System
- **Start**: The system will automatically open your webcam
- **Monitor**: Watch the real-time display with overlay information
- **Quit**: Press `q` to stop the system

## 📋 What You'll See

The system displays:
- **Gaze Direction**: left/right/center
- **Hand Shape**: gripping/open
- **Trajectory**: movement direction
- **Predicted Action**: what the person is likely doing
- **Alerts**: Red warnings for critical actions

## ⚠️ Important Notes

- **Camera Access**: Make sure your webcam is not being used by other applications
- **Lighting**: Ensure good lighting for better detection accuracy
- **Distance**: Sit 1-2 meters from the camera for optimal performance

## 🔧 Troubleshooting

### Camera Issues
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### Dependencies Issues
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

### Permission Issues (Linux/Mac)
```bash
# Make sure you have camera permissions
sudo chmod 666 /dev/video0
```

## 📊 Understanding the Output

| Display Element | Meaning |
|----------------|---------|
| Green Action Text | Normal activity detected |
| Red Action Text | High-priority action (medication/emergency) |
| ⚠️ ALERT | Critical action requiring attention |
| Yellow Info Text | System status information |

## 🎯 Expected Performance

- **Frame Rate**: ~20 FPS
- **Latency**: <100ms detection to alert
- **Accuracy**: Optimized for elderly movement patterns
- **Memory**: ~200MB RAM usage

## 📝 Logs

All activity is automatically logged to:
- `data/elderly_activity_log.csv` - Detailed activity logs
- `data/elderly_monitoring_output.avi` - Video recording

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review the [examples/](examples/) directory for usage examples
- Run the tests: `python -m pytest tests/`

---

**Ready to monitor? Run `python src/main_predictor.py` and press `q` to quit!**

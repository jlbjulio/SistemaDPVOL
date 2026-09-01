# Intelligent Parking Space Detection System Using Computer Vision

This project is a computer vision application designed to automatically detect and classify parking spaces as free or occupied from camera images. Using machine learning and HOG (Histogram of Oriented Gradients) features, the system analyzes parking lot images and provides real-time occupancy information. The system achieves >95% accuracy and can be integrated into parking management systems, mobile apps, or informational displays.

## Technologies Used

- Python 3.10+
- scikit-learn (LinearSVC, feature extraction)
- OpenCV (image processing)
- Flet (GUI framework)
- Pandas (data handling)
- NumPy (numerical operations)

## Main Features

- **Automatic Dataset Creation**:

  - Builds training dataset from full parking lot images
  - Uses metadata and camera coordinates to label each space
  - Supports automatic or manual dataset organization

- **Machine Learning Model**:

  - LinearSVC classifier with HOG feature extraction
  - Optimized for 64x64 pixel space patches
  - Balanced class weighting for improved accuracy
  - Optional data augmentation (brightness, rotation, flips)

- **Interactive Visualization Interface**:

  - Real-time parking space detection on images
  - Loop playback simulation of sequential frames
  - Color-coded overlay (red=free, blue=occupied)
  - Easy-to-use controls for start, pause, and frame navigation

- **Performance Metrics**:
  - Generates comprehensive classification reports
  - Includes confusion matrix and detailed accuracy metrics
  - Free/Occupied class precision and recall tracking
  - Exportable results in JSON format

## How to Use

1. **Setup Environment**:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:

   - Organize images into `dataset/free` and `dataset/occupied` folders, or
   - Let the system auto-generate patches from full images using metadata

3. **Train the Model**:

   ```powershell
   python SistemaDPVOL.py
   ```

   - System will detect existing dataset and begin training
   - Creates `models/parking_model.pkl` and `report.json`

4. **Launch Visualization Interface**:

   ```powershell
   python gui_app.py
   ```

   - Add sequential frames to `pruebavideo/` folder
   - Use controls to play, pause, and adjust playback speed

5. **View Results**:
   - Check `report.json` for model performance metrics
   - Review `config.json` for augmentation and training settings

## Considerations

- Best performance with consistent lighting conditions; strong shadows or reflections may affect accuracy
- ROIs (Regions of Interest) must be properly defined on a reference image for inference
- Metadata matching uses ±20 minute temporal tolerance for frame labeling
- Augmentation parameters can be adjusted in `config.json` for different weather conditions
- Model training time depends on dataset size and augmentation settings

## Performance Metrics

- Accuracy: ~95.65%
- Free space precision/recall: 0.94/0.96
- Occupied space precision/recall: 0.97/0.95
- Inference speed: ~100 predictions per second on standard hardware

## Possible Improvements

- Integrate with real-time camera feeds
- Implement temporal smoothing to reduce prediction noise
- Migrate to lightweight CNN model (MobileNet) for better robustness
- Add automatic ROI detection and refinement
- Create web-based dashboard for real-time monitoring
- Implement vehicle re-identification across frames
- Add multi-camera space-level tracking

## Data Source

- **CNRPark+EXT Dataset**: Public parking lot dataset with multiple cameras, weather conditions, and occupancy metadata
- 10,000+ labeled parking space images across different times and conditions

---

> **Note**: This system was developed as an educational prototype. For production deployment, ensure proper legal compliance regarding camera usage, data privacy, and fine-tune ROI calibration for your specific parking lot.

---

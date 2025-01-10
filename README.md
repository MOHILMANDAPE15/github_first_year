# Semantic Segmentation Project

This project leverages deep learning to perform semantic segmentation of cellular and microscopic images, identifying cells and nuclei with high precision.
Using a U-Net architecture, the model is trained to distinguish and segment nuclei in biomedical images.
The project is inspired by tasks like automated nucleus detection, which can accelerate research in biology and medicine.

---

## **Features**
- **U-Net Architecture**: Implements a state-of-the-art U-Net model for accurate image segmentation.
- **Deep Learning Frameworks**: Built with TensorFlow/Keras.
- **Binary Classification**: Segments each pixel into nucleus or background.
- **Extensive Preprocessing**: Includes data normalization and augmentation to improve model performance.
- **Customizable Hyperparameters**: Easy-to-adjust model parameters like dropout rates, convolutional layers, and kernel sizes.

---

## **Project Workflow**
1. **Data Preparation**:
   - Input images are resized to 128x128 pixels.
   - Data is split into training, validation, and test sets.
   - Images are normalized to a [0, 1] range.

2. **Model Architecture**:
   - **Encoder**: Progressive down-sampling using convolutional and max-pooling layers.
   - **Bottleneck**: Feature extraction with deeper convolutional layers.
   - **Decoder**: Up-sampling and skip connections to recover spatial details.

3. **Training**:
   - Optimizer: Adam.
   - Loss Function: Binary cross-entropy.
   - Metrics: Accuracy.
   - Early stopping and checkpoint callbacks ensure optimal training.

4. **Evaluation**:
   - Visualizes predictions on training, validation, and test data.
   - Compares segmented output against ground truth.

---

## **Setup Instructions**

### **Prerequisites**
Ensure you have Python installed with the following libraries:
- TensorFlow
- NumPy
- Matplotlib

### **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/MOHILMANDAPE15/SEMANTIC_SEGMENTATION
   ```
2. Navigate to the project directory:
   ```bash
   cd SEMANTIC_SEGMENTATION
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Usage**
1. Prepare your dataset:
   - Organize images into `train`, `validation`, and `test` directories.
   - Update paths in the script if needed.

2. Run the training script:
   ```bash
   python train.py
   ```
3. Evaluate the model:
   ```bash
   python evaluate.py
   ```
4. Visualize results:
   - The script displays input images, ground truth masks, and predicted masks.

---

## **Results**
- **Training Accuracy**: Achieved 96.14% accuracy.
- **Validation Accuracy**: Achieved 95.64% accuracy.
- **Visual Output**:
  - Below is an example of the segmentation performance:

| Input Image | ---> | Predicted Mask |
|--------![input](https://github.com/user-attachments/assets/6bb6cba5-3a2f-4e22-a021-8f4ca3f15e7d)
-----|---![output](https://github.com/user-attachments/assets/504c42bb-61c3-4fa7-9194-b93aec7e0ea2)
-----|----------------|


---

## **Future Improvements**
- Incorporate multi-class segmentation for more complex datasets.
- Experiment with other architectures (e.g., ResNet or EfficientNet encoders).
- Add real-time segmentation capabilities.

---

## **Contributing**
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
- Inspired by biomedical image segmentation challenges such as the Data Science Bowl.
- Built using TensorFlow/Keras with reference to U-Net implementations.

---

Feel free to reach out for collaboration or queries at mohilmandpe33@gmail.com



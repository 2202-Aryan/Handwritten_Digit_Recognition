# Handwritten Digit Recognition

This project allows you to draw a digit on the screen using Pygame and receive real-time predictions for the handwritten digit using a trained deep learning model.

## Prerequisites

- Python 3.x
- Pygame
- OpenCV
- TensorFlow
- Trained model for handwritten digit recognition (e.g., `model.h5`)

## Installation

1. Clone the repository or download the source code.

2. Install the required Python libraries:

3. Place your trained model file (`model.h5`) in the project directory.

## Usage

1. Run the script `app.py`: python app.py

2. A Pygame window will open with a blank canvas.

3. Use the mouse to draw a digit on the canvas.

4. Release the mouse button to trigger the prediction.

5. The predicted digit and its accuracy will be displayed on the canvas.

6. Repeat steps 3-5 to draw and predict more digits.

7. Close the Pygame window to exit the application.

## Customization

- Brush size: You can adjust the brush size by modifying the `BRUSH_SIZE` constant in the code.

- Model file: If your trained model file has a different name, update the `model.h5` filename in the code accordingly.

- Window size: The default window size is set to (280, 280). You can modify the `window_size` variable in the code to change the window dimensions.

## Acknowledgments

- The deep learning model used in this project is trained on the MNIST dataset, which consists of handwritten digit images.

- The model is implemented using the TensorFlow library.

- The graphical user interface for drawing and displaying the digits is built using Pygame.

- Image processing tasks, such as thresholding and contour detection, are performed using OpenCV.





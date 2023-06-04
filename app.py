import pygame
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Initialize Pygame
pygame.init()

# Set up the drawing window
window_size = (280, 280)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Handwritten Digit Recognition")

# Load the trained model
model = keras.models.load_model('model.h5')

# Create a blank canvas
canvas = np.zeros((280, 280), dtype=np.uint8)

# Define colors
BLACK = 0
WHITE = 255

# Define brush size
BRUSH_SIZE = 10

# Function to preprocess and predict the digit
def predict_digit(image):
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    accuracy = prediction[0][predicted_label]
    return predicted_label, accuracy

# Game loop
running = True
drawing = False
digit_predicted = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            digit_predicted = False
            canvas.fill(BLACK)  # Clear the canvas
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            digit_predicted = True
        elif event.type == pygame.MOUSEMOTION and drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, WHITE, (mouse_x, mouse_y), BRUSH_SIZE)

    pygame.display.flip()

    # Get the current screen surface as an image
    image_data = pygame.surfarray.array3d(screen)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

    # Threshold the image
    _, image_data = cv2.threshold(image_data, 127, 255, cv2.THRESH_BINARY_INV)

    # Resize the image to match the model input shape
    image_data = cv2.resize(image_data, (28, 28))

    if digit_predicted:
        # Predict the digit and get accuracy
        prediction, accuracy = predict_digit(image_data)

        # Draw the predicted label and accuracy on the canvas
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"Label: {prediction}", (10, 30), font, 1, WHITE, 2)
        cv2.putText(canvas, f"Accuracy: {accuracy:.2f}", (10, 60), font, 1, WHITE, 2)

    # Display the canvas on the screen
    cv2.imshow("Handwritten Digit Recognition", canvas)
    cv2.waitKey(1)

cv2.destroyAllWindows()
pygame.quit()

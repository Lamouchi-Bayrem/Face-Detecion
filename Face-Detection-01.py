# Import libreries
import cv2
import numpy as np 

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the image data
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If no faces detected, return None
    if len(faces) == 0:
        return None, None
    
    
    # Assuming only one face in the image, extract the region of interest (ROI)
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+w, x:x+h]
    
    # Resize the ROI to a fixed size for model input
    face_roi = cv2.resize(face_roi, (100, 100))
    
    return face_roi, (x, y, w, h)

# Example usage:
image_path = 'path_to_your_image.jpg'
preprocessed_face, face_coords = preprocess_image(image_path)

if preprocessed_face is not None:
    # Display the preprocessed face
    cv2.imshow('Preprocessed Face', preprocessed_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected in the image.")

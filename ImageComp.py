import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt

# Load pre-trained VGG16 model without top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_image(image):
    # Resize the image to the input size expected by the model
    image = cv2.resize(image, (224, 224))
    # Preprocess input for the VGG16 model
    image = preprocess_input(image)
    return image

def extract_features(image, model):
    image = preprocess_image(image)
    features = model.predict(np.expand_dims(image, axis=0))
    return features.flatten()  # Flatten the feature vector

def overlay_images(image1, image2, alpha=0.3):
    # Blend the two images using a weighted average
    blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    return blended_image

def image_similarity(image1_filename, image2_filename):
    try:
        # Load the images using OpenCV
        image1 = cv2.imread(image1_filename)
        image2 = cv2.imread(image2_filename)

        # Extract features from the images using the pre-trained model
        features1 = extract_features(image1, base_model)
        features2 = extract_features(image2, base_model)

        # Calculate cosine similarity between the feature vectors using scikit-learn
        similarity = cosine_similarity([features1], [features2])[0][0]

        # Convert cosine similarity to a percentage (0-100%)
        similarity_percent = (similarity + 1) / 2 * 100

        # Overlay the images with one of them grayed out
        overlaid_image = overlay_images(image1, image2)

        return similarity_percent, overlaid_image

    except Exception as e:
        return str(e), None

if __name__ == "__main__":
    image1_filename = "Apple1.png"
    image2_filename = "Apl1.png"

    similarity_percent, overlaid_image = image_similarity(image1_filename, image2_filename)

    if isinstance(similarity_percent, float):
        print(f"Similarity between the two images: {similarity_percent:.2f}%")

        # Display the overlaid image
        plt.imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
        plt.title('Overlaid Image')
        plt.axis('off')
        plt.show()
    else:
        print(f"Error: {similarity_percent}")

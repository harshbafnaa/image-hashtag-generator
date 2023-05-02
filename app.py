from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import PIL.Image

app = Flask(__name__)

# Load a pre-trained deep learning model for image classification
model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')

# Define a function to preprocess the image
def preprocess(image):
    # Load the image using PIL
    img = PIL.Image.open(image)

    # Resize the image to 299x299 pixels
    img = img.resize((299, 299))

    # Convert the PIL Image object to a NumPy array
    img_array = np.array(img)

    # Normalize the pixel values to be between -1 and 1
    img_array = (img_array / 127.5) - 1.0

    # Add an extra dimension to the array to represent the batch size (1 in this case)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Define a function to generate hashtags for an image
def generate_hashtags(image):
    # Preprocess the image
    img_array = preprocess(image)

    # Use the model to extract features from the image
    features = model.predict(img_array)

    # Convert the features to a list of hashtags
    hashtags = []
    for i in range(10):
        idx = np.argmax(features)
        word = tf.keras.applications.inception_v3.decode_predictions(features, top=10)[0][i][1]
        hashtags.append(word)
        features[0][idx] = -1
        # print(hashtags)

    return hashtags




# Define a route to handle file upload and hashtag generation
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        image = request.files['file']

        # Generate hashtags for the image
        hashtags = generate_hashtags(image)

        # Return the generated hashtags as a JSON response
        return jsonify(hashtags)

    # Render the initial template with the upload form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

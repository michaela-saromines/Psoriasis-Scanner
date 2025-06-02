from flask import Flask, render_template, request, send_from_directory
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from keras import layers
import numpy as np
import os
import uuid # For generating unique filenames

class SpatialPyramidPooling(layers.Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    """
    def __init__(self, pool_list, **kwargs):
        super(SpatialPyramidPooling, self).__init__(**kwargs)
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i * i for i in pool_list])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3] * self.num_outputs_per_channel)

    def call(self, x):
        input_shape = K.shape(x)
        num_rows = input_shape[1]
        num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []
        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for jy in range(num_pool_regions):
                for ix in range(num_pool_regions):
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = jy * row_length[pool_num]
                    y2 = jy * row_length[pool_num] + row_length[pool_num]

                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')
                    y1 = K.cast(K.round(y1), 'int32')
                    y2 = K.cast(K.round(y2), 'int32')

                    # Crop the image region and apply max pooling
                    x_crop = x[:, y1:y2, x1:x2, :]
                    pooled_val = K.max(x_crop, axis=(1, 2))
                    outputs.append(pooled_val)

        return K.concatenate(outputs, axis=1)

get_custom_objects().update({'SpatialPyramidPooling': SpatialPyramidPooling})

model_path = "../Inceptionv3-SPP/r3/best.weights.h5"

model = load_model(model_path, custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})

app = Flask(__name__)

# Base directory for storing images
UPLOAD_FOLDER_BASE = './images/'
NORMAL_FOLDER = os.path.join(UPLOAD_FOLDER_BASE, 'normal')
PSORIASIS_FOLDER = os.path.join(UPLOAD_FOLDER_BASE, 'psoriasis')
TEMP_UPLOAD_FOLDER = './temp_uploads'

# Ensure these directories exist
os.makedirs(NORMAL_FOLDER, exist_ok=True)
os.makedirs(PSORIASIS_FOLDER, exist_ok=True)
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def design():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    original_filename = imagefile.filename

    # Generate a unique filename to prevent overwrites
    file_extension = os.path.splitext(original_filename)[1]
    unique_filename = str(uuid.uuid4()) + file_extension

    temp_image_path = os.path.join(TEMP_UPLOAD_FOLDER, unique_filename)
    imagefile.save(temp_image_path) # Save to temp location first

    image = load_img(temp_image_path, target_size=(224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    yhat = model.predict(img_array)

    labels = ['normal', 'psoriasis']
    predicted_index = int(yhat.argmax())
    predicted_label = labels[predicted_index]
    confidence = float(yhat[0][predicted_index]) * 100

    if predicted_label == 'normal':
        final_save_folder = NORMAL_FOLDER
    else:
        final_save_folder = PSORIASIS_FOLDER

    final_image_path = os.path.join(final_save_folder, unique_filename)

    # Move from temp to final classified folder
    os.rename(temp_image_path, final_image_path)

    classification = f"{predicted_label} ({confidence:.2f}%)"

    return render_template('index.html',
                           prediction=classification,
                           predicted_folder=predicted_label,
                           uploaded_filename=unique_filename)

@app.route('/images/<folder>/<filename>')
def uploaded_file(folder, filename):
    # This route now correctly expects <folder> and <filename>
    return send_from_directory(os.path.join(os.getcwd(), 'images', folder), filename)

# New route for the gallery landing page
@app.route('/gallery_main')
def gallery_main():
    return render_template('gallery_landing.html')


# Existing route to display existing images by category
@app.route('/gallery/<category>')
def gallery(category):
    images_in_folder = []
    if category == 'normal':
        folder_path = NORMAL_FOLDER
    elif category == 'psoriasis':
        folder_path = PSORIASIS_FOLDER
    else:
        return "Invalid category", 404

    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            images_in_folder.append(filename)

    return render_template('gallery.html', category=category, images=images_in_folder)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
import streamlit as st
import numpy as np
import tensorflow as tf
from rembg import remove
from PIL import Image
import cv2
import pandas as pd

# st.title("Leaf Detection App")
st.set_page_config(page_title='Classification', layout='centered')

st.header('Medicinal Plants Identifier', divider=True)


def enhance_image(image):
    # Load the input image
    input_image = Image.fromarray(image)

    # remove background
    output_image = remove(input_image, alpha_matting=True)

    # Create a black background of the same size as the input image
    black_background = Image.new('RGB', input_image.size, (0, 0, 0))

    if black_background.size != output_image.size:
        # Resize one of the images to match the dimensions of the other
        black_background = black_background.resize(output_image.size)

    if black_background.mode != output_image.mode:
        # Convert one of the images to the same mode as the other
        black_background = black_background.convert(output_image.mode)

    # Composite the foreground object onto the black background
    composite_image = Image.alpha_composite(black_background.convert('RGBA'),
                                            output_image)

    # Convert the PIL Image to a NumPy array
    composite_np = np.array(composite_image)

    # Convert to BGR (OpenCV format)
    bgr = cv2.cvtColor(composite_np, cv2.COLOR_RGBA2BGR)

    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(20, 20))
    v = clahe.apply(v)
    hsv_img = np.dstack((h, s, v))
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gamma = 1.5
    # Perform gamma correction
    gamma_corrected = np.power(bgr / 255.0, gamma)
    gamma_corrected = np.uint8(gamma_corrected * 255)
    final_output = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)

    return final_output


img_file_buffer = st.camera_input("Take a picture")
file = st.file_uploader('Choose a file')

if img_file_buffer is not None or file is not None:
    # To read image file buffer as a 3D uint8 tensor with TensorFlow:
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
    else:
        bytes_data = file.getvalue()

    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8),
                           cv2.IMREAD_COLOR)
    cv2_img = enhance_image(cv2_img)
    img_tensor = tf.convert_to_tensor(cv2_img)

    # img_tensor = enhance_image(img_tensor)
    img_tensor = tf.cast(img_tensor, tf.float32)/255.0
    img_tensor = tf.image.resize(img_tensor, [224, 224])
    # img_tensor.set_shape([224, 224, 3])
    # Should output shape: (height, width, channels)
    interpreter = tf.lite.Interpreter(model_path="./Model/model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    input_shape = input_details['shape']
    interpreter.set_tensor(input_details['index'], [img_tensor])

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    classes = ['Moringa', 'Neem', 'Green Chiretta', 'Holy Basil',
               'Malabar Nut', 'Arjun']
    # ["Mint", "Moringa", "Neem", "Green Chiretta","Malabar Nut", "Arjun"]
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # st.write(output_data)
    pred_class = classes[np.argmax(output_data)]
    text = pd.read_csv(f'./Benefits of Medicinal Plants/{pred_class}.csv')
    st.divider()
    st.write("Predicted Class:", pred_class)
    st.divider()
    st.dataframe(text, width=1000)

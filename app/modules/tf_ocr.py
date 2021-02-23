import gzip
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import base64
import cv2

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# PROJECT_DIR = os.path.join(PROJECT_ROOT,'../../')
PROJECT_DIR=''
print(PROJECT_DIR)

# with gzip.open(PROJECT_DIR+'app/modules/TF_OCR/tf_ocr.h5.gz', 'r') as f:
#     OCRModel = pickle.load(f)

def base64_cv2(base64_str):
    """
    base64 to cv2
    Args:
          base64_str: Image base64 encoder string
    Return:
          cv2 format(ndarray) image
    """
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString,np.uint8)
    image = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
    cv2.imwrite(PROJECT_DIR+'app/modules/TF_OCR/upload/'+'output.png', image)
    # return image

def encode_single_sample(img_path):
    img_width = 200
    img_height = 50

    # 1. Read image
    # imgString = base64.b64decode(base64_str)
    # img = tf.io.decode_base64(imgString)
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    # label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return img

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    # Maximum length of any captcha in the dataset
    max_length = 5
    characters = ['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y']
    # Mapping characters to integers
    char_to_num = layers.experimental.preprocessing.StringLookup(
        vocabulary=list(characters), num_oov_indices=0, mask_token=None
    )
    # Mapping integers back to original characters
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def getResult(base64_str):

    prediction_model = load_model(PROJECT_DIR+'app/modules/TF_OCR/tf_ocr.h5', compile=False)
    # prediction_model.summary()
    base64_cv2(base64_str) #save base64 to xxx.png

    img_path  = PROJECT_DIR+'app/modules/TF_OCR/upload/'+'output.png'
    img_fromapi = encode_single_sample(img_path)


    preds = prediction_model.predict(np.array([img_fromapi]))
    pred_texts = decode_batch_predictions(preds)
    return str(pred_texts[0])
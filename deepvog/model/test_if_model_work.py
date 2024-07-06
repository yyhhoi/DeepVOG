from DeepVOG_model import load_DeepVOG
import numpy as np
from PIL import Image

def test_if_model_work():
    model = load_DeepVOG()
    img = np.zeros((1, 240, 320, 3))
    img_raw = Image.open("test_image.png")
    img_array = np.array(img_raw)
    img[:,:,:,:] = (img_array/255).reshape(1, 240, 320, 1)
    prediction = model.predict(img, verbose=0)
    prediction_array = prediction[0,:,:,1]
    prediction_image = Image.fromarray((prediction_array * 255).astype(np.uint8))
    prediction_image.save("test_prediction.png")

if __name__ == "__main__":
    # If model works, the "test_prediction.png" should show the segmented area of pupil from "test_image.png"
    test_if_model_work()


from DeepVOG_model import load_DeepVOG
import skimage.io as ski
import numpy as np
import matplotlib.pyplot as plt

def test_if_model_work():
    model = load_DeepVOG()
    img = np.zeros((1, 240, 320, 3))
    img_raw = ski.imread("test_image.png")    
    img[:,:,:,:] = (img_raw/255).reshape(1, 240, 320, 1)
    prediction = model.predict(img)
    plt.imshow(prediction[0,:,:,1], cmap='gray')
    plt.savefig('test_prediction.png')
    # ski.imsave("test_prediction.png", prediction[0,:,:,1])
    np.save('test_img_raw_array.npy', img_raw)
    np.save('test_model_prediction.npy', prediction)


if __name__ == "__main__":
    # If model works, the "test_prediction.png" should show the segmented area of pupil from "test_image.png"
    test_if_model_work()


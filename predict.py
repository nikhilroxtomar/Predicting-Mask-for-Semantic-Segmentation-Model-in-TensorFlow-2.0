import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from metrics import iou
from tqdm import tqdm

def load_tf_model(path):
    with tf.keras.utils.CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model(path)
        return model

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Load Images """
    test_images = glob("test_images/*")
    print(f"Test Images: {len(test_images)}")

    """ Hyperparamaters """
    size = (256, 256)
    model_path = "model.h5"

    """ Create `results` directory """
    if os.path.exists("results"):
        pass
    else:
        os.makedirs("results")

    """ Load the Model """
    model = load_tf_model(model_path)
    # model.summary()

    time_taken = []
    for path in tqdm(test_images, total=len(test_images)):
        """ Extract the name """
        name = path.split("/")[-1].split(".")[0]

        """ Read image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        H, W, _ = image.shape
        image = cv2.resize(image, size)             ## (256, 256, 3)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)       ## (1, 256, 256, 3)
        image = image.astype(np.float32)

        """ Predict """
        start_time = time.time()
        mask = model.predict(image)[0]
        total_time = time.time() - start_time
        time_taken.append(total_time)

        """ Save Mask """
        mask = cv2.resize(mask, (W, H))
        mask = mask > 0.5
        mask = mask * 255
        mask = mask.astype(np.float32)

        save_path = f"results/{name}.png"
        cv2.imwrite(save_path, mask)

    mean_time = np.mean(time_taken)
    mean_fps = 1/mean_time
    print(f"Mean Time: {mean_time:1.7f} - Mean FPS: {mean_fps:1.7f}")













    ###

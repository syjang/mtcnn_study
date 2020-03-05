import model
import mtcnn
import utils
import numpy as np
import cv2


if __name__ == "__main__":
    net = model.make_pnet()

    img = cv2.imread(
        'dataset/wider_images/0--Parade/0_Parade_marchingband_1_6.jpg')
    scale = 1
    resized = utils.scale_image(img, scale)
    normalized = utils.normalize_image(resized)
    net_input = np.expand_dims(normalized, 0)

    print(net_input.shape)

    outputs = net.predict(net_input)

    pass

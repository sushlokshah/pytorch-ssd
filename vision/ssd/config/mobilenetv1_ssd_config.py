import numpy as np
import cv2
# from vision.utils import box_utils
from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors, center_form_to_corner_form
import numpy as np

image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)
print(priors.shape)

image = np.zeros((300, 300, 3), dtype=np.uint8)
croners = center_form_to_corner_form(priors)
for i in range(100):
    # denormalize corners
    croners[i][0] = croners[i][0] * image_size
    croners[i][1] = croners[i][1] * image_size
    croners[i][2] = croners[i][2] * image_size
    croners[i][3] = croners[i][3] * image_size
    cv2.rectangle(image, (int(croners[i][0]), int(croners[i][1])), (int(
        croners[i][2]), int(croners[i][3])), (255, 0, 0), 1)

cv2.imwrite("/sushlok/img.png", image)

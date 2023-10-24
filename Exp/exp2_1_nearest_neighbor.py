import cv2
import math
import numpy as np
from PIL import Image

x_scale_factor = 2
y_scale_factor = 2

origin = cv2.imread(r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp2\lab2.png')
shape = origin.shape
# print(shape)

y_len = shape[0] * y_scale_factor
x_len = shape[1] * x_scale_factor
out = np.zeros((y_len, x_len, 3))

for i in range(x_len):
    for j in range(y_len):
        out[j, i] = origin[math.floor(j / y_scale_factor), math.floor(i / x_scale_factor)]
        if np.array_equal(out[j, i], [0, 0, 0]):
            print(i)
            print("\t")
            print(j)
            print('\n')
# out[0, 0, 0] = 5
print(out)
# print(origin)
# print(type(origin))
image = Image.fromarray(out.astype('uint8'))

single_insertion_path = r'E:\MyFiles\KnowledgeBase\Year3Fall\DigitalImageProcessing\Exp\Project\resource\exp2\single.png'
image.save(single_insertion_path)

# image.show()

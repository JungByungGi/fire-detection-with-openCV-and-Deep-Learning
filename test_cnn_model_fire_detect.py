import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

model = tf.keras.models.load_model('CNN_Model1.h5')
cnt_fire = 0
avg_fire_percent = 0

print("Fire Image Test\n")

for i in os.listdir('./Test_Image/fire'):
    path = './Test_Image/fire/' + i
    img_data = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img_data)
    x = np.expand_dims(x, axis=0) / 255
    classes = model.predict(x)
    if np.argmax(classes[0]) == 0:
        cnt_fire += 1
        avg_fire_percent += max(classes[0])
    print(np.argmax(classes[0]) == 0, max(classes[0]))


print("50개의 화재 사진 중 화재 이미지 판별 수 : %d" % cnt_fire)
print("화재 분류 평균 정확도 : %f\n" % (avg_fire_percent / cnt_fire))
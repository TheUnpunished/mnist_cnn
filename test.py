from __future__ import print_function
# импортируем keras
import keras
# импортируем json для создания файла с результатом
import json
# импортируем numpy для создания numpy-массива с исходными данными
import numpy as np
# импортируем backend keras'а
from keras import backend as K
# импортируем PIL для работы с изображениями
from PIL import Image

# загружаем существующую натренированную сеть
model = keras.models.load_model("model/model.h5")
# картинки 28*28
img_rows, img_cols = 28, 28
# инициализируем лист с картинками
data = []
for i in range(10):
    # открываем картинки с 0 до 9
    img = Image.open("data/" + str(i) + ".png")
    # достаём "сырые" данные из них
    temp = list(img.getdata())
    # добавляем в лист data
    data.append(temp)
# конвертируем в ndarray
data = np.array(data)
# делаем решейп в зависимости от настройки keras
if K.image_data_format() == 'channels_first':
    data = data.reshape(data.shape[0], 1, img_rows, img_cols)
else:
    data = data.reshape(data.shape[0], img_rows, img_cols, 1)
# нормализируем
data = data.astype('float32')
data /= 255
# выполняем классификацию
pred = model.predict(data)
# извлекаем нужные данные
values = pred.argmax(axis=0).tolist()
pred = pred.tolist()

# создаём лист с данными, которые будем потом записывать в json
json_list = []
for i in range(10):
    # создаём dictionary с нужными данными
    # expected - ожидаемое значение
    # predicted - более наглядное значение, выданное классификатором
    # result - вероятностные данные, выданные классификатором
    x = {"expected": i, "predicted": values[i], "result": pred[i]}
    json_list.append(x)
# записываем в json-файл
with open("result.json", 'w', encoding='utf-8') as f:
    json.dump(json_list, f, ensure_ascii=False, indent=4)
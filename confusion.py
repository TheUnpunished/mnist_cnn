from __future__ import print_function

# для обучения используется keras
import keras
from keras import backend as K
# встроенная API для загрузки данных MNIST с внешнего сервера
from keras.datasets import mnist
# библиотека metrics нужна для подсчитывания confusion-матрицы
from sklearn import metrics

# количество классов, 10 цифр - 10 классов
num_classes = 10
# картинки 28*28
img_rows, img_cols = 28, 28

# загрузка уже существующей модели keras
model = keras.models.load_model("model/model.h5")
# загрузка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# в зависимости от настройки keras, мы делаем решейп данных
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# превращаем матрицу y_test в двоичную матрицу классификации
y_test = keras.utils.to_categorical(y_test, num_classes)

# выполняем тестирование точности
score = model.evaluate(x_test, y_test, verbose=0)
print('Потеря:', score[0])
print('Точность:', score[1])

# с помощью обученной сетки получаем выборку в виде набора классов
y_pred = model.predict(x_test)
# создаём непосредственно confusion матрицу
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

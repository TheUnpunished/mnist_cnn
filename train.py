from __future__ import print_function
# импортируем keras для обучения сети
import keras
# импортируем API базы данных mnist
from keras.datasets import mnist
# импортируем последувательную модель нейронной сети
from keras.models import Sequential
# импортируем различные слои нейронной сети
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# импортируем backend keras'а
from keras import backend as K
# импортируем numpy для решейпов, создания ndarray и удаления из них данных
import numpy as np
# импортируем функцию рандома
import random as rnd

# датасет делится на батчи, которые уже проще прочитать нейронной сети при обучении
batch_size = 128
# количество классов, в нашем случае количество цифр от 0 до 9 - 10
num_classes = 10
# количество эпох обучения, слишком много - переобучение
# слишком мало - недообучение
epochs = 2
# процентное количество тестировочной выборки от всей выборки
part_val = 0.16

# размер входных картинок 28*28
img_rows, img_cols = 28, 28

# загрузка данных mnist
# разделены на тренировочные и валидационные
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# соединяем тренировочные и валидационные данные, чтобы самому выделить необходимое количество
# под каждую операцию
X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0)

# размер общей выборки
size = X.shape[0]
# размер валидационный выборки
val_size = int(size * part_val)
# номера из X, которые будут использоваться для валидационной выборки
val_args = []

# генерируем такие номера
for x in range(0, val_size):
    i = rnd.randrange(0, size - 1)
    while i in val_args:
        i = rnd.randrange(0, size - 1)
    val_args.append(i)

# создаём подмассив из этих номеров
x_test = X[val_args]
y_test = Y[val_args]

# удаляем валидационные данные из общей выборки
X = np.delete(X, val_args, axis=0)
Y = np.delete(Y, val_args, axis=0)

# делаем решейп данных в зависимости от настройки keras
if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# нормализуем данные
X = X.astype('float32')
X /= 255
print('Матрица X обучающих данных имеет размеры: ', X.shape)
print(X.shape[0], 'фрагментов данных для обучения')
print(x_test.shape[0], ' те самые 16% от данных, используются для тестирования')

# превращаем матрицы Y и y_test в двоичные матрицы классификации
Y = keras.utils.to_categorical(Y, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# определяем модель и слои нейронной сети
# последовательная модель
model = Sequential()
# добавляем 2-мерный конволюционный слой с 32 нейронами
# сетка 3*3
# "одна строка" входных данных является матрицей размерностями, указанных в input_shape
# конволюционные слои хороши тем, что собирают данные с пикселей, стоящих только рядом
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# ещё один конволюционный слой с сеткой 3*3 и 64 нейронами
model.add(Conv2D(64, (3, 3), activation='relu'))
# Pooling слой, обычно ставится после конволюционных
# помагает с проблемой переобучения
# размер сетки 2*2
# по сути преобразует 2*2 пикселя в 1,
# используя их, например, максимальное среди них значение яркости
model.add(MaxPooling2D(pool_size=(2, 2)))
# dropout слой
# Если вероятность ниже 0.25
# значение этой вероятности становится 0
model.add(Dropout(0.25))
# Flatten-слой позволяет превратить матрицы в строки
model.add(Flatten())
# dense-слой, в котором каждый нейрон связан со всеми входами
model.add(Dense(128, activation='relu'))
# ещё один dropout слой
model.add(Dropout(0.5))
# ещё один dense слой
model.add(Dense(num_classes, activation='softmax'))
# собираем модель
# выбираем функцию потери, оптимизатор, а также выводимые метрики
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# засовываем данные в сеть
# X, Y - данные для обучения (X - данные изображений, Y - классы)
# epochs - количество эпох
# verbose - режим многословия, какие данные будут выводиться при обучении
# validation data - валидационные данные
model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)
          )
# эвалюируем на данных для валидации
score = model.evaluate(x_test, y_test, verbose=0)
# выводим результат
print('Потеря:', score[0])
print('Точность:', score[1])
# сохраняем модель
model.save("model/model.h5")

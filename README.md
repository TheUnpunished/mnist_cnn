 Программа разбита на 3 скрипта:
 	train.py - загрузка данных из базы данных keras.mnist, создание и обучение нейросети на них. Перевод в файл
 	test.py - загрузка обученной модели из model/model.h5 и проверка на картинках из data, выдача результата в json-файле (result.json)
 	cofusion.py - загрузка обученной модели, загрузка данных из базы данных и создание confusion-матрицы
Для запуска понадобится сделать следующие дейтсвия:
	Открыть терминал
	Перейти по директории, где находится проект
	Выполнить команды:
		source ./env/bin/activate
		python <имя скритпа с расширением>
Примечание. При выполнении команды "source ./env/bin/activate" командная строка перейдёт в виртуальную среду. Чтобы выйти из неё, нужно ввести команду deactivate.
Это сделано для того, чтобы данная программа могла запуститься на любом компьютере.
	
from parking_lot_detection_v2 import process

import cv2 as cv
import numpy as np
import os
import imutils
from math import floor, tan, sqrt
import random
from sklearn.cluster import KMeans
from pandas import DataFrame
import argparse
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import dlib
from matplotlib import pyplot as plt

# Запуск программы с командной строки
# cd C:\CREESTL\Programming\PythonCoding\semestr_3\parking_lot_detection
# python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input parking_lots/video_3.mp4 --output output --skip-frames 5

# парсер аргументов с командной строки
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo",
	help = "path to yolo directory")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", required = True, type=str,
	help="path to input video file")
ap.add_argument("-o", "--output", required = True, type=str,
	help="path to output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
	help="number of frames to skip between detections"
		 "the higher the number the slower the program works")
args = vars(ap.parse_args())


# классы объектов, которые могут быть распознаны алгоритмом
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


# считываем натренированную модель с диска
print("[INFO] loading model...")
net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# путь к исходному видео
print("[INFO] input directory: ", args["input"])

# читаем видео с диска
print("[INFO] opening video file...")
vs = cv.VideoCapture(args["input"])

# объявляем инструмент для записи конечного видео в файл, указываем путь
writer = None
writer_path = args["output"] + "\last_output.avi"
print("[INFO] output directory: ", writer_path)

# инициализируем размеры кадра как пустые значения
# они будут переназначены при анализе первого кадра и только
# это ускорит работу программы
W = None
H = None

# инициализируем алгоритм трекинга
# maxDisappeared = кол-во кадров, на которое объект может исчезнуть с видео и потом опять
# будет распознан
# maxDistance = максимальное расстояние между центрами окружностей, вписанных в боксы машин
# Если расстояние меньше заданного, то происходит переприсваение ID
ct = CentroidTracker(maxDisappeared=30, maxDistance=40)
# сам список трекеров
trackers = []
# список объектов для трекинга
trackableObjects = {}

# полное число кадров в видео
totalFrames = 0

# счетчик машин и временная переменная
total = 0
temp = 0

# статус: распознавание или отслеживание
status = None

# создаем график, показывающий рост количества машин на видео
figure = plt.figure(dpi = 90, figsize = (10, 6))
plt.title ("Amount of cars per frame")
plt.ylabel("Amount of cars", fontsize = 10)
plt.xlabel("Frame number")
totals = []
frames = []

#номер кадра видео
frame_number = 0

# проходим через каждый кадр видео
while True:
	frame_number += 1
	frames.append(frame_number)
	frame = vs.read()
	frame = frame[1]

	# если кадр является пустым значением, значит был достигнут конец видео
	if frame is None:
		print("=============================================")
		print("The end of the video reached")
		print("Total number of cars on the video is ", total)
		print("=============================================")
		break
	else:
		# если кадр первый, то обрабатываем его
		if frame_number == 1:
			# РИСУЮТСЯ ВСЕ ЛИНИИ И АЙДИШНИКИ НА КАДРЕ
			frame, centers_dict, contours = process(frame)

	if frame_number > 1:
		# изменим размер кадра для ускорения работы
		frame = imutils.resize(frame, width=800)



	# после обработки первого кадра у нас остаются все координаты линий и центров с него - их и рисуем
	for id, coords in centers_dict.items():
		cv.circle(frame, (int(coords[0]), int(coords[1])), 5, (255, 255, 0), 3)
		cv.putText(frame, str(id + 1), (int(coords[0]) + 10, int(coords[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv.drawContours(frame, contours[id], -1, (0, 0, 255), 3)  # рисуем контуры на чистой фотке
	# для работы библиотеки dlib необходимо изменить цвета на RGB вместо BGR
	rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

	# размеры кадра
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# задаем путь записи конечного видео
	if  writer is None:
		fourcc = cv.VideoWriter_fourcc(*"MJPG")
		writer = cv.VideoWriter(writer_path,fourcc, 30,
			(W, H), True)

	# этот список боксов может быть заполнен двумя способами:
	# (1) детектором объектов
	# (2) трекером наложений из библиотеки dlib
	rects = []

	# каждые N кадров (указанных в аргументе "skip_frames" производится ДЕТЕКТРОВАНИЕ машин
	# после этого идет ОТСЛЕЖИВАНИЕ их боксов
	# это увеличивает скорость работы программы
	if totalFrames % args["skip_frames"] == 0:
		# создаем пустой список трекеров
		trackers = []

		status = "Detecting..."

		# получаем blob-модель из кадра и пропускаем ее через сеть, чтобы получить боксы распознанных объектов
		blob = cv.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)

		detections = net.forward()

		# анализируем список боксов
		for i in np.arange(0, detections.shape[2]):
			# извлекаем вероятность совпадения
			confidence = detections[0, 0, i, 2]

			# получаем ID наиболее "вероятных" объектов
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])

				# если распознана не машина, то переходим к следующей итерации
				if CLASSES[idx] != "car":
					continue

				# вычисляем координаты бокса объекта
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# теперь необходимо отследить наложения боксов друг на друга
				# для этого используется библиотека dlib
				# в ней бокс представлен в виде прямоугольника (rectangle)
				# создадим такой прямоугольник и начнем отслеживание
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)
				# добавим трекер в список всех трекером для дальнейшего использования
				trackers.append(tracker)

	# если же кадр не явялется N-ым, то необходимо работать с массивом сформированных ранее трекеров, а не боксов
	else:
		for tracker in trackers:

			status = "Tracking..."

			# обновляем список трекеров
			tracker.update(rgb)
			# получаем позицию трекера в списке(это 4 координаты)
			pos = tracker.get_position()

			# из трекера получаем координаты бокса, соответствующие ему
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# и эти координаты помещаем в главный список боксов (по нему и будет производиться рисование)
			rects.append((startX, startY, endX, endY))

	# используем ранее инициализированных трекер центроидов, чтобы сопоставить
	# ранее полученные координаты центроидов боксов с только что полученными
	objects = ct.update(rects) #это словарь (1) ID (2) координаты(?) центроида

	# алгоритм подсчета машин
	length = len(objects.keys())
	if length > total:
		total += length - total
	if (length > temp) and (temp != 0):
		total += length - temp
	if length < total:
		temp = length
	totals.append(total)

	# анализируем массив отслеживаемых объектов
	for (objectID, centroid) in objects.items():
		# проверяем существует ли отслеживаемый объект для данного ID
		to = trackableObjects.get(objectID, None)

		# если его нет, то создаем новый, соответствующий данному центроиду
		if to is None:
			to = TrackableObject(objectID, centroid)

		# в любом случае помещаем объект в словарь
		# (1) ID (2) объект
		trackableObjects[objectID] = to


		# изобразим центроид и ID объекта на кадре
		text = "ID {}".format(objectID + 1)
		cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	info = [
		("Total", total),
		("Status", status)
	]

	# изобразим информаци о количестве машин на краю кадра
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv.putText(frame, text, (10, H - ((i * 20) + 20)),
		cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)

	# записываем конечный кадр в указанную директорию
	if writer is not None:
		writer.write(frame)

	#рисуем график
	plt.plot(frames, totals, c = "red")


	# показываем конечный кадр в отдельном окне
	cv.imshow("Frame", frame)
	key = cv.waitKey(1) & 0xFF

	# для прекращения работы необходимо нажать клавишу "q"
	if key == ord("q"):
		print("[INFO] process finished by user")
		print("Total number of cars on the video is ", total)
		break

	# т.к. все выше-обработка одного кадра, то теперь необходимо увеличить количесвто кадров
	# и обновить счетчик
	totalFrames += 1

# график выводится на экран в конце работы программы
plt.show()

# освобождаем память под переменную
if writer is not None:
	writer.release()


# освобождаем память под переменную
else:
	vs.release()

# закрываем все окна
cv.destroyAllWindows()


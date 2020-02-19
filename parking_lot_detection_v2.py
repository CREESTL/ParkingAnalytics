
import cv2 as cv
import numpy as np
import os
import imutils
from math import floor, tan, sqrt, atan2, pi
import random
from sklearn.cluster import KMeans
from pandas import DataFrame



"""
Класс служит для обозначения факта, что точка уже соединена с какой-то другой или же свободна
"""
class Point():
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.is_start = False  # точка - начало прямой
        self.is_end = False  # точка - конец прямой\
        self.connected = []  # здесь будут хранится координаты центроидов, которые соединены с данным

"""
Класс парковочного места
"""
class Center():
    # id - номер центра парковки, coords - координаты центра парковки
    def __init__(self, id, coords):
        self.id = id
        self.coords = coords



"""
Нужна дла корректной работы следующей функции
"""
def nothing(*arg):
    pass


"""
Пользователь вручную настраивает цветовой фильтр дла распознавания белых линий парковки
"""
def manually_set_filter(img):

    cv.namedWindow("result")  # главное окно
    cv.namedWindow("settings")  # окно настроек

    # создаем 6 бегунков для настроек параметров HLS
    cv.createTrackbar("h1", "settings", 0, 255, nothing)
    cv.createTrackbar("s1", "settings", 0, 255, nothing)
    cv.createTrackbar("v1", "settings", 0, 255, nothing)
    cv.createTrackbar("h2", "settings", 255, 255, nothing)
    cv.createTrackbar("s2", "settings", 255, 255, nothing)
    cv.createTrackbar("v2", "settings", 255, 255, nothing)

    while True:

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # считываем значения бегунков
        h1 = cv.getTrackbarPos("h1", "settings")
        s1 = cv.getTrackbarPos("s1", "settings")
        v1 = cv.getTrackbarPos("v1", "settings")
        h2 = cv.getTrackbarPos("h2", "settings")
        s2 = cv.getTrackbarPos("s2", "settings")
        v2 = cv.getTrackbarPos("v2", "settings")

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((h1, s1, v1), np.uint8)
        h_max = np.array((h2, s2, v2), np.uint8)


        # накладываем фильтр на кадр в модели HSV
        thresh = cv.inRange(hsv, h_min, h_max)

        cv.imshow("result", thresh)

        # для окончания настройки фильтра необходимо нажать на q
        ch = cv.waitKey(5)
        if ch == ord("q"):
            cv.destroyAllWindows()

            return h_min, h_max  #после того как пользователь нажимает на q то возвращаются границы


"""
Эта функция создает "маски" которые фильтруют все пиксели, оставляя только те, цвет которых
указан
"""
def apply_filter(img, black_or_white, green):
    # кадр переводится в формат HSV
    image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    if black_or_white == True:
        # маска для белого цвета в формате HLS
        lower, upper = manually_set_filter(img)
        #ниже - универсальный фильтр для белого цвета
        #lower = np.uint8([0, 0, 211])
        #upper = np.uint8([255, 65, 255])
        white_mask = cv.inRange(image, lower, upper)
        print("Applying white filter")
        mask = white_mask
        return mask
    if green == True:
        # маска для зеленого цвета
        # применяется, чтобы отфильтровать все, кроме линий Хофа
        lower = np.uint8([0, 196, 197])
        upper = np.uint8([255, 255, 255])
        green_mask = cv.inRange(image, lower, upper)
        print("Applying green filter")
        mask = green_mask
        return mask

"""
Функция переводит линии в формат массива
"""
def lines_to_array(lines):
    new = []
    for i, line in enumerate(lines):
        new.append(np.array(line[0]).tolist())
    return new

"""
Функция удаляет все слишком короткие линии
"""
def delete_short_lines(lines):
    max_length = 0
    new = []
    for line in lines:
        line = np.array(line).tolist()
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        length = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if length > max_length:
            max_length = length
    desired_length = max_length / 2  # требуемая длина - половина максимальной
    for line in lines:
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        length = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if length > desired_length: # если длина линии больше требуемой (половины максимальной), то она возвращается
            new.append(line)

    lines = new
    lines = np.array(lines)
    return lines


"""
Функция удаляет из массива контуров одинаковые контуры
"""
def only_different_cnts(contours):
    cnt_box = {}  # словарь (номер контура-бокс)
    for i, c in enumerate(contours):
        rect = cv.minAreaRect(c)  # это типа tuple
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        box = list(box)  # переводим из формала tuple внешний массив
        for i in range(len(box) - 1):
            box[i] = list(box[i])  # переводим из формата tuple внутренние подмассивы

        cnt_box[i] = box  # заносим в словарь

    for i, box in cnt_box.items():
        for j in range(i, len(cnt_box.keys()) - 1):  # от i и до конца массива ключей
            another_box = cnt_box[j]
            if box == another_box:
                print("Found similar contours...Deleting them")
                to_delete = contours[j]
                contours.remove(to_delete)  # если нашли одинаковые контуры, то сразу из массива удаляем все, кроме первого

    return contours  # возвращаем массив без повторений

"""
Функция находит расстояние между двумя точками(центроидами)
"""
def find_distance(p1, p2):
    distance = sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return distance


"""
Функция находит расстояние между двумя точками, представленными через класс
"""
def find_distance_class(p1, p2):
    distance = sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    return distance


"""
Функция удаляет центроиды, находящиеся близко друг к другу
"""
def only_different_centroids(centroids):
    new = []
    for c in centroids:
        new.append(np.array(c).tolist())
    for c in new:
        for a in new:
            if c != a:
                if find_distance(c, a) < 20:  # это расстяние можно варьировать
                    new.remove(a)


    return new

"""
Функция удаляет ложно обнаруженные центры парковок, находящиеся близко друг к другу
"""
def only_different_park_centers(centers):
    for c in centers:
        for a in centers:
            if c != a:
                if find_distance(c, a) < 20:  # это расстяние можно варьировать
                    centers.remove(a)


    return centers


"""
Преобразует массив центроидов в массив экземпляров классов
"""
def to_class_list(centroids):
    class_list = []
    for centroid in centroids:
        class_list.append(Point(centroid[0], centroid[
            1]))  # создаем массив экземпляров класса чтобы потом понимать какие точки соединены, а какие - нет
    centroids = class_list

    return centroids


"""
Функция:
1) Для каждого центроида находятся две ближайшие точки, одна дальше другой. 
2) Проверятся, не был ли центроид соединен с каким-то из найденных
2) Центроид соединяется с теми, с которыми еще не соединён
"""
def connect_two_closest(centroids, image, show_steps):
    # находится первый ближайший центроид
    for i, centroid in enumerate(centroids):
        min_dist = 1000
        for j, another in enumerate(centroids):
            if i != j:
                dist = find_distance_class(centroid, another)
                if dist < min_dist:
                    min_dist = dist # это минимальное расстояние
                    min_index = j # это номер ближайшего центроида
        # теперь находится центроид, который дальше только что найденного, но ближе всех остальных
        # то есть второй по дальности
        min_diff = 1000
        for k, another in enumerate(centroids):
            if (k != i) and (k != min_index):
                dist = find_distance_class(centroid, another)
                diff = dist - min_dist  # находится разность расстояний до ближайше точки и до данной
                if diff < min_diff:  # если это расстояние меньше заданного, то ищем минимальное
                    min_diff = diff
                    second_min_index = k
        if centroids[min_index] not in centroids[i].connected: # если точки еще не были соединены
            cv.line(image, (int(centroid.x), int(centroid.y)),(int(centroids[min_index].x), int(centroids[min_index].y)), (0, 255, 0), 3)
            # если рисовать центроиды, то будет наглядно видно как происодит соединение, но последующий код будет работать некорректно
            #cv.circle(image, (centroids[min_index].x, centroids[min_index].y), 1, (0, 0, 255),5)  # самую близку точку рисуем зеленым
            centroids[i].connected.append(centroids[min_index]) # фиксируется соединение двуз центроидов

        if centroids[second_min_index] not in centroids[i].connected: # если точки еще не были соединены
            cv.line(image, (int(centroid.x), int(centroid.y)), (int(centroids[second_min_index].x), int(centroids[second_min_index].y)),(0,255, 0), 3)
            #cv.circle(image, (centroids[second_min_index].x, centroids[second_min_index].y), 1, (0, 0, 0),5)  # вторую по удаленности рисуем красны
            centroids[i].connected.append(centroids[second_min_index])
        #cv.circle(image, (centroids[i].x, centroids[i].y), 1, (255, 255, 255), 5)  # начальную точку рисуем белым
        #if show_steps:
            #cv.imshow("closest", image)
            #cv.waitKey()


"""
Функция переводит approx в координаты x и y
"""
def approx_to_x_and_y(approx):
    x = []
    y = []
    for el in approx:
        el = np.array(el).tolist()
        el = el[0]
        x.append(el[0])
        y.append(el[1])
    return x, y

#========================================================================================================================

# главная функция программы, которая вызывает все, описанные выше
# на вход подается кадр с камеры
# на выходе получаем тот же кадр с размеченными на нем парковочными местами с номерами, расставленными на них
def process(original_img):

    if original_img is None:
        print("You wrote a wrong path to the image! Try once again")
        exit()

    # пользователь выбирает, хочет ли он видеть все шаги обработки кадра
    print("Do you want to see all the steps of processing?")
    print("1) Yes\n2) No")
    choice = int(input())
    if choice == 1:
        show_steps = True
    else:
        show_steps = False

    # размер кадра изменяется для удобства
    original_img = imutils.resize(original_img, width=800)
    (H, W) = original_img.shape[:2]
    if show_steps:
        cv.imshow("original_img", original_img)
        cv.waitKey()

    #убираются шумы с фото
    no_noize = cv.bilateralFilter(original_img, 11, 17, 17)

    #накладывается белый фильтр, чтобы потом применить линии Хофа
    white_img = apply_filter(no_noize, True, False)
    for_lines = white_img
    with_lines = original_img.copy()
    if show_steps:
        cv.imshow("white_filter", white_img)
        cv.waitKey()

    # минимальная длина линии и минимальный разрыва между линиями
    min_length = W / 100
    max_gap = W / 80


    # находятся все линии на кадре
    lines = cv.HoughLinesP(for_lines, rho=1, theta=np.pi/180, threshold=50, minLineLength=min_length, maxLineGap=max_gap)

    # переводим линии в формат массива
    lines = lines_to_array(lines)

    # иногда, если находятся линии во весь экран, то слишком много подходящих из-за этого удаляется
    lines = delete_short_lines(lines)

    # все линии рисуются на кадре
    for line in lines:
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        cv.line(with_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # все линии рисуются
    if show_steps:
        cv.imshow("Hough lines", with_lines)
        cv.waitKey()

    # на кадр с нарисованными зелеными линиями применяется фильтр зеленого цвета
    with_lines = apply_filter(with_lines, False, True)
    if show_steps:
        cv.imshow("green_filter", with_lines)
        cv.waitKey()


    # находятся все грани
    edges = cv.Canny(with_lines, 100, 200)
    if show_steps:
        cv.imshow('edges', edges)
        cv.waitKey()

    # создаем копии для рисования на них
    copy_1 = original_img.copy()
    copy_2 = original_img.copy()
    approx_img = original_img.copy()
    for_centroids = original_img.copy()
    for_centroids_2 = for_centroids.copy()
    for_final_contours = original_img.copy()
    for_pure_rects = original_img.copy()

    # находятся все конутры
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print("Found contours: ", len(contours))
    # удаляются все одинаковые контуры, чтобы не рисовать их много раз
    contours = only_different_cnts(contours)
    print("Final number of contours: ", len(contours))
    # рисуются все полученные контуры
    cv.drawContours(copy_2, contours, -1, (0, 255, 0), 2)
    if show_steps:
        cv.imshow("all_contours", copy_2)
        cv.waitKey()

    # словарь координат точек approx
    data = {
            "x":[],
            "y":[]
            }

    approx_lenghts = []
    for i,c in enumerate(contours):
        perimeter = cv.arcLength(c, True)
        # эпсилон - максимальное расстояние от настоящего угла на картинке и его "предсказания"
        # True отвечает за замыкание первой и последней точек
        epsilon = 0.02 * perimeter

        # мест четное -> approx = мест * 5
        # мест нечетное -> approx = мест * 4 + 1

        approx = cv.approxPolyDP(c, epsilon, False) #находим незамкнутые контуры

        # надо ограничить длину - не надо добавлять контуры из 200 точек
        if len(approx) // 4 < 2 :
            approx_lenghts.append(len(approx))

        # находим все x и y координаты точек
        x, y = approx_to_x_and_y(approx)
        for each_x in x:
            data["x"].append(each_x)
        for each_y in y:
            data["y"].append(each_y)

        # рисуются точки
        cv.drawContours(approx_img, approx, -1, (0, 255, 0), 3)
        if show_steps:
            cv.imshow("approx", approx_img)


    dataframe = DataFrame(data, columns=["x", "y"])

    # это - ожидаемое число кластеров центроидов
    num_of_clusters = sum(approx_lenghts) // 2

    # применятеся алгоритм kmeans чтобы распределить все центроиды на кластеры
    kmeans = KMeans(n_clusters=num_of_clusters).fit(dataframe)
    centroids = kmeans.cluster_centers_

    centroids = only_different_centroids(centroids)

    centroids = to_class_list(centroids) # делаем массив классов вместо просто массива

    # все центриды рисуются в виде синих кругов
    if show_steps:
        for c in centroids:
            cv.circle(for_centroids_2, (c.x, c.y), 5, (255,0,0), 3)
    if show_steps:
        cv.imshow("all_centroids", for_centroids_2)
        cv.waitKey()


    # каждый центроид соединятеся с двумя ближайшими
    connect_two_closest(centroids, for_centroids, show_steps)

    #на фотке с соединенными центроидами рисуем все линии
    for line in lines:
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        cv.line(for_centroids, (x1, y1), (x2, y2), (0, 255, 0), 3)
    if show_steps:
        cv.imshow("everything", for_centroids)
        cv.waitKey()

    # теперь еще раз применятся фильтр зеленого цвета
    green = apply_filter(for_centroids, False, True)
    if show_steps:
        cv.imshow("green_rects", green)
        cv.waitKey()

    # еще раз находятся все контуры
    contours, _ = cv.findContours(green, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # все контуры рисуются
    cv.drawContours(for_final_contours, contours, -1, (0, 255, 0), 3)
    if show_steps:
        cv.imshow("final_contours", for_final_contours)
        cv.waitKey()

    centers = []  # массив координтак центров парковок (не центроидов!)
    good_contours = []  # массив подходящих нам контуров
    for i, c in enumerate(contours):
        rect = cv.minAreaRect(c)  # находится прямоугольник, вписанный в контур
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        box = np.array(box).tolist()
        perimeter = cv.arcLength(c, True)
        epsilon = 0.02 * perimeter
        approx = cv.approxPolyDP(c, epsilon, False)


        min_area = (W * H) * 0.002  # это минимальная площадь контура, которая будте обрабатываться
        # ограничение в 0.002 сильно влияет на корректность работы - его можно варьировать
        if area > min_area:
            # отсеиваются контуры, у которых слишком много изгибов
            if (len(approx)) in range(5, 7):
                cv.drawContours(for_pure_rects, [c], -1, (0, 0, 255), 3) # рисуются контуры на чистом кадре
                good_contours.append(c)  # координаты контуры добавляются в массив, из которого потом будут извлечены
                x1, y1 = box[0][0], box[0][1]
                x2, y2 = box[2][0], box[2][1]
                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                centers.append(center)  # добавляет координаты центра парковки в массив, чтобы потом их нарисовать
                if show_steps:
                    cv.imshow("rect", for_pure_rects)

    # создаем массив классов центров парковок
    centers_classes = []
    # удаляем центры, которые накладываются друг на друга
    centers = only_different_park_centers(centers)
    # уже после того, как нарисовали прямоугольники - рисуем номера, чтобы они были поверх них
    id = 0
    for i,center in enumerate(centers):
        # в массив добавляем новый центр
        centers_classes.append(Center(id, center))
        cv.circle(for_pure_rects, (int(center[0]), int(center[1])), 5, (255, 255, 0), 3)
        # ставим id над центром
        cv.putText(for_pure_rects, str(id + 1), (int(center[0]) + 10, int(center[1]) - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 0, 0), 2)
        id += 1
    if show_steps:
        cv.imshow("FINAL", for_pure_rects)

    # финальная версия картинки выводится на экран
    cv.imshow("FINAL", for_pure_rects)


    # после всех преобразований возвращает:
    # финальную картинку
    # массив классов центра парковок (id, координаты центра)
    # массив контуров, подходящих нам (парковок)
    return for_pure_rects, centers_classes, good_contours


##############################################################################################################################

# путь к фотографии, которую хотим обработать
original_img = cv.imread(r'C:\CREESTL\Programming\PythonCoding\semestr_3\parking_lot_detection\parking_lots\ideal.jpg')

if __name__ == "__main__":
    process(original_img)


k = cv.waitKey(0)

print("\nGOODBYE!")
cv.destroyAllWindows()
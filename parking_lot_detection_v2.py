"""
Здесь все то же самое, что и в v2, только по фильтрованной фотке будут рисоваться белые линии Хофа,
а уже по ним будут искаться контуры, чтобы не было прервания линий при детекте
"""


import cv2 as cv
import numpy as np
import os
import imutils
from math import floor, tan, sqrt
import random
from sklearn.cluster import KMeans
from pandas import DataFrame


"""
ПОЭТОМУ НАДО СДЕЛАТЬ ЧТОБЫ ПОЛЬЩОВАТЕЛЬ САМ НА ПЕРВМО КАДРЕ КРУТИЛ БЕГУНКИ ИЗ SET_FILTER, А ПОТОМ ВСЕ РАБОТАЛО
"""


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
    crange = [0, 0, 0, 0, 0, 0]

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
        # h_min = np.uint8([h1, s1, v1])
        h_max = np.array((h2, s2, v2), np.uint8)
        # h_max = np.uint8([h2, s2, v2])
        # накладываем фильтр на кадр в модели HSV

        thresh = cv.inRange(hsv, h_min, h_max)

        cv.imshow("result", thresh)

        ch = cv.waitKey(5)
        if ch == ord("q"):
            cv.destroyAllWindows()
            print("h_min = ", h_min, " h_max = ", h_max)
            return h_min, h_max  #после того как пользователь нажимает на q то возвращаются границы


"""
Эта функция создает "маски" которые фильтруют все пиксели, оставляя только те, цвет которых
указан

"""
def apply_filter(img, black_or_white, green):
    #если black_or_white == true , то накладываем черный фильтр
    #нужно перевести картинку в формат HLS
    image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    """
    Теперь с помощью операции дизъюнкции мы объеденяем все маски в одну
    В итоге, при наложении этой маски на фотографии, отсеятся все пиксели, кроме
    белых пока что (смотри return)
    Именно этими цветами обычно рисуются полосы парковки и госномера
    """
    if black_or_white == True:
        # маска для белого цвета в формате HLS!!!
        # эти числа получаю через прогу set_filter
        #                H   S    V
        lower, upper = manually_set_filter(img)
        #ниже - универсальный фильтр для белого цвета, а сверху - кастомный
        #lower = np.uint8([0, 0, 211])
        #upper = np.uint8([255, 65, 255])
        white_mask = cv.inRange(image, lower, upper)
        print("Applying white filter")
        mask = white_mask
        return mask
    if green == True:
        # маска для зеленого цвета
        # применяется, чтобы отфильтровать все, кроме линий Хофа
        lower = np.uint8([0, 196, 197])  # новый попробовать 56 0 189
        upper = np.uint8([255, 255, 255])  # 80 255 217
        green_mask = cv.inRange(image, lower, upper)
        print("Applying green filter")
        mask = green_mask
        return mask


"""
Эта функция все серые пиксели раскидывает на абсолютно черные и абсолютно белые
"""
def stabilize_image(img,mode):

    #убираем шумы с фото
    blur = cv.GaussianBlur(img, ksize=(33, 33), sigmaX=0)

    if(mode == 0):

        """
        разобратсья что делают эти функции, но стоит их оставить
        но суть в том, что они замыкают формы, которые чуть чуть не дотягивают до круга
        """
        closed = cv.erode(blur, None, iterations=15)
        closed = cv.dilate(closed, None, iterations=12)
        """
        На вход функции подается серая картинка
        Функция проверяет значение цвета каждого пискеля
        Если значение больше 40, то оно переприсваивается и становится 255 (то есть белым)
        Если значение меньше 40, то оно переприсваивается и становится 0 (то есть черным)
        THRESH_BINARY_INV - это режим фильтрации, при нем на выходном результате
        цвет пикселей в массиве идет по убыванию - от белого к черноому
        белый слева, черный справа
        """
        _, thresh = cv.threshold(img, 40, 255, cv.THRESH_BINARY_INV)

        return thresh
    else:
        #если нет необходимости в обработке кадра, то просто возвращается его серая версия
        return blur
    # иначе позвращается обработанная фотка


"""
Функция удаляет из массива контуров одинаковые контуры
"""
def only_different_cnts(contours):
    cnt_box = {}  # словарь (номер контура-бокс)
    for i, c in enumerate(contours):
        rect = cv.minAreaRect(c)  # это типа tuple
        # print("rect = ", rect)
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        # print("box = ", box)
        box = np.int0(box)  # округление координат
        #print("tuple box = ", box, "*******")
        box = list(box)  # переводим из формала tuple внешний массив
        #print("array box = ", box, "*******")
        for i in range(len(box) - 1):
            box[i] = list(box[i])  # переводим из формата tuple внутренние подмассивы

        cnt_box[i] = box  # заносим в словарь

    for i, box in cnt_box.items():
        for j in range(i, len(cnt_box.keys()) - 1):  # от i и до конца массива ключей
            another_box = cnt_box[j]
            if box == another_box:
                print("Found similar contours...Deleting them")
                to_delete = contours[j]
                print("to_delete = ", to_delete)
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
        print("\nc = ", c)
        for a in new:
            if c != a:
                if find_distance(c, a) < 20:  # это расстяние можно варьировать
                    print(c, " is too close to ", a, " remove ", a)
                    new.remove(a)


    return new

"""
Функция удаляет ложно обнаруженные центры парковок, находящиеся близко друг к другу
"""
def only_different_park_centers(centers):
    for c in centers:
        print("\nc = ", c)
        for a in centers:
            if c != a:
                if find_distance(c, a) < 20:  # это расстяние можно варьировать
                    print(c, " is too close to ", a, " remove ", a)
                    centers.remove(a)


    return centers


"""
Функция рисует все контуры случайными цветами
"""
def colored_contours(contours, img, min_dots = 1, max_dots = 10):
    for c in contours:
        perimeter = cv.arcLength(c, True)
        # эпсилон - максимальное расстояние от настоящего угла на картинке и его "предсказания"
        # True отвечает за замыкание первой и последней точек
        # approxPolyDP находит не совсем углы, например для буквы П он выдает число 6
        # по два на каждый верхний угол и по 1 на низ каждой из палок
        epsilon = 0.02 * perimeter
        approx = cv.approxPolyDP(c, epsilon, False)  # находим незамкнутые контуры
        if len(approx) in range(min_dots, max_dots):  # задаем количество изгибов через аргументы
            red = random.randint(0,255)
            green = random.randint(0,255)
            blue = random.randint(0,255)
            cv.drawContours(img, c, -1, (red, green, blue), 3)


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
1) Для каждого центроида находятся две ближайшие точки, одна дальше другой. Он с ними соединятеся
2) Можно рисовать центроиды по желанию, но тогда плохо будет работать следующий код
3) Производится проверка, чтобы не рисовать линию между двумя центроидами, которые уже были соединеиы
"""
def connect_two_closest(centroids, image, show_steps):
    for i, centroid in enumerate(centroids):
        min_dist = 1000
        for j, another in enumerate(centroids):
            if i != j:
                dist = find_distance_class(centroid, another)
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
        print("now min_dist = ", min_dist)
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
            # ЕСЛИ РИСОВАТЬ ЦЕНТРОИДЫ, ТО ОН КОРЯВО ПОТОМ НАХОДИТ КОНТУРЫ
            #cv.circle(image, (centroids[min_index].x, centroids[min_index].y), 1, (0, 0, 255),3)  # самую близку точку рисуем зеленым
            centroids[i].connected.append(centroids[min_index])
        if centroids[second_min_index] not in centroids[i].connected:
            cv.line(image, (int(centroid.x), int(centroid.y)), (int(centroids[second_min_index].x), int(centroids[second_min_index].y)),(0,255, 0), 3)
            #cv.circle(image, (centroids[second_min_index].x, centroids[second_min_index].y), 1, (0, 0, 0),3)  # вторую по удаленности рисуем
            centroids[i].connected.append(centroids[second_min_index])
        #cv.circle(image, (centroids[i].x, centroids[i].y), 1, (255, 255, 255), 3)  # начальную точку рисуем белым
        if show_steps:
            cv.imshow("closest", image)
            cv.waitKey()


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


def process(original_img):

    if original_img is None:
        print("You wrote a wrong path to the image! Try once again")
        exit()


    print("Do you want to see all the steps of processing?")
    print("1) Yes\n2) No")
    choice = int(input())
    if choice == 1:
        show_steps = True
    else:
        show_steps = False

    original_img = imutils.resize(original_img, width=800)
    (H, W) = original_img.shape[:2]
    if show_steps:
        cv.imshow("original_img", original_img)
        cv.waitKey()


    #убираем шумы с фото
    no_noize = cv.bilateralFilter(original_img, 11, 17, 17)
    if show_steps:
        cv.imshow("no_noize", no_noize)
        cv.waitKey()

    #накладываем белый фильтр, чтобы потом применить линии хофа
    white_img = apply_filter(no_noize, True, False)
    for_lines = white_img # здесь искать линии
    with_lines = original_img.copy() # сюда рисовать линии
    if show_steps:
        cv.imshow("white_filter", white_img)
        cv.waitKey()

    """
    ТЕПЕРЬ НАДО НАРИСОВАТЬ ЛИНИИ НА ОРИГИНАЛЬНЙО ФОТКЕ, ПОТОМ ЕЕ СНОВА ПЕРЕКРАСИТЬ ФИЛЬТРОМ, НО УЖЕ 
    ЗЕЛЕНЫМ, А ПОТОМ ИСКАТЬ КОНТУРЫ!!!!
    """


    min_length = W / 100
    max_gap = W / 80
    #находим линии на копии БЕЛОЙ ФОТКИ
    lines = cv.HoughLinesP(for_lines, rho=1, theta=np.pi/180, threshold=50, minLineLength=min_length, maxLineGap=max_gap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(with_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #рисуем все линии на копии оригиниальной фотки
    if show_steps:
        cv.imshow("Hough lines", with_lines)
        cv.waitKey()
    cv.imwrite("green.jpg", with_lines)
    cv.waitKey()

    with_lines = apply_filter(with_lines, False, True)
    if show_steps:
        cv.imshow("green_filter", with_lines)
        cv.waitKey()


    #находим все грани на черном фильтре (именно на нем надо)
    edges = cv.Canny(with_lines, 100, 200)
    if show_steps:
        cv.imshow('edges', edges)
        cv.waitKey()

    #создаем копии для рисования на них
    copy_1 = original_img.copy()
    copy_2 = original_img.copy()
    approx_img = original_img.copy() # для точек аппрокса
    colored_cnts = original_img.copy()
    for_centroids = original_img.copy() # для рисования сех центроидов
    for_final_contours = original_img.copy() # для конутров уже самих прямоугольников парковок
    for_pure_rects = original_img.copy()  # здесь будут только красные контуры и айдишники

    # находим ВСЕ конутры
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print("Found contours: ", len(contours))
    #удаляем все одинаковые контуры, чтобы не рисовать их много раз
    contours = only_different_cnts(contours)
    print("Final number of contours: ", len(contours))
    #рисуем все полученные контуры на СТАБИЛИЗИРОВАННОЙ ФОТКЕ
    cv.drawContours(copy_2, contours, -1, (0, 255, 0), 2)
    if show_steps:
        cv.imshow("all_contours", copy_2)
        cv.waitKey()

    # рисуем все контуры разными цветами
    #colored_contours(contours, colored_cnts, 5, 7)
    #cv.imshow("colored_cnts" ,colored_cnts)


    data = {
            "x":[],
            "y":[]
            } # словарь координат точек approx

    approx_lenghts = []  # сюда записываем длину каждого аппрокса, а потом посчитаем общее количество точек

    for i,c in enumerate(contours):
        perimeter = cv.arcLength(c, True)
        # эпсилон - максимальное расстояние от настоящего угла на картинке и его "предсказания"
        # True отвечает за замыкание первой и последней точек
        # approxPolyDP находит не совсем углы, например для буквы П он выдает число 6
        # по два на каждый верхний угол и по 1 на низ каждой из палок
        epsilon = 0.02 * perimeter

        # мест четное -> approx = мест * 5

        # мест нечетное -> approx = мест * 4 + 1

        approx = cv.approxPolyDP(c, epsilon, False) #находим незамкнутые контуры
        print("len approx = ", len(approx))

        if len(approx) // 4 < 2 :# надо ограничить длину - не надо добавлять контуры из 200 точек
            approx_lenghts.append(len(approx))  #чтобы потом посчитать сколько всего точек
            # длина аппрокса - количество точек изгиба контура


        x, y = approx_to_x_and_y(approx)# находим все x и y координаты точек
        print("x = ", x)
        print("y = ", y)
        for each_x in x:
            data["x"].append(each_x)
        for each_y in y:
            data["y"].append(each_y)


        cv.drawContours(approx_img, approx, -1, (0, 255, 0), 3)
        if show_steps:
            cv.imshow("approx", approx_img)


    print("\n\nDOING KMEANS!!!")

    dataframe = DataFrame(data, columns=["x", "y"])
    print("dataframe")
    print(dataframe)

    print("approx lengths = ", approx_lenghts)
    print("Number of curves: ", sum(approx_lenghts))

    num_of_clusters = sum(approx_lenghts) // 2             # лучше всего подходит 2 для двух рядов парково (напротив друг друга)

    print("\nNUMBER OF CLUSTERS: ", num_of_clusters)

    kmeans = KMeans(n_clusters=num_of_clusters).fit(dataframe)
    centroids = kmeans.cluster_centers_
    centroids = only_different_centroids(centroids)
    print("cleared centroids are: ", centroids)
    '''
    Это я убрал, чтобы время не тратить
    for c in centroids:
        c = np.array(c).tolist()
        c = list(map(lambda x: floor(x), c))
        cv.circle(for_centroids, (c[0],c[1]), 5, (255,0,0), 3)
        if show_steps:
            cv.imshow("centroids", for_centroids)
            cv.waitKey()
    '''
    centroids = to_class_list(centroids) # делаем массив классов вместо просто массива

    #находим просто ближайшие центроиды и соединяем
    connect_two_closest(centroids, for_centroids, False)

    #на фотке с соединенными центридами рисуем все линии
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(for_centroids, (x1, y1), (x2, y2), (0, 255, 0), 3)
    if show_steps:
        cv.imshow("everything", for_centroids)
        cv.waitKey()


    green = apply_filter(for_centroids, False, True)
    if show_steps:
        cv.imshow("green_rects", green)
        cv.waitKey()
    #edges = cv.Canny(green, 100, 200)
    contours, _ = cv.findContours(green, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #contours = only_different_cnts(contours)
    cv.drawContours(for_final_contours, contours, -1, (0, 255, 0), 2)
    if show_steps:
        cv.imshow("final_contours", for_final_contours)
        cv.waitKey()
    # рисую круг в центре четырехугольника
    centers = []  # массив координтак центров парковок
    good_contours = []  # массив подходящих нам четырехугольников
    for i, c in enumerate(contours):
        rect = cv.minAreaRect(c)  # это типа tuple
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        # print("rect = ", rect)
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        # print("box = ", box)
        box = np.int0(box)  # округление координат
        box = np.array(box).tolist()
        perimeter = cv.arcLength(c, True)
        epsilon = 0.02 * perimeter
        approx = cv.approxPolyDP(c, epsilon, True)
        # среди контуров ищем контуры с 4 углами

        min_area = (W * H) * 0.02  # это минимальная площадь контура, которая будте обрабатываться
        if area > min_area:
            if (len(approx)) in range(4, 6):
                cv.drawContours(for_pure_rects, [c], -1, (0, 0, 255), 3)  # рисуем контуры на чистой фотке
                good_contours.append(c)  # добавляем координаты контура, чтобы потом его рисовать на кадрах видео
                print("box = ", box)
                x1, y1 = box[0][0], box[0][1]
                x2, y2 = box[2][0], box[2][1]
                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                centers.append(center)  # добавляет координаты центра парковки в массив, чтобы потом их нарисовать
                if show_steps:
                    cv.imshow("rect", for_pure_rects)
    # создаю словарь ID - координаты для центров
    centers_dict = {}
    # удаляем центры, которые накладываются друг на друга
    centers = only_different_park_centers(centers)
    # уже после того, как нарисовали прямоугольники - рисуем номера, чтобы они были поверх них
    id = 0
    for i,center in enumerate(centers):
        centers_dict[id] = center # добавляю центр в словарь, чтобы потом менять его цвет
        cv.circle(for_pure_rects, (int(center[0]), int(center[1])), 5, (255, 255, 0), 3)
        cv.putText(for_pure_rects, str(id + 1), (int(center[0]) + 10, int(center[1]) - 10), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 0, 0), 2)
        id += 1
    if show_steps:
        cv.imshow("FINAL", for_pure_rects)




    # после всех преобразований возвращает:
    # финальную картинку
    # словарь ЦЕНТРОВ(ID центра - координаты четырехугольника)
    # массив контуров, подходящих нам (парковок)
    return for_pure_rects, centers_dict, good_contours


##############################################################################################################################

original_img = cv.imread(r'C:\CREESTL\Programming\PythonCoding\semestr_3\parking_lot_detection\parking_lots\sample.jpg')

if __name__ == "__main__":
    process(original_img)


k = cv.waitKey(0)
cv.destroyAllWindows()
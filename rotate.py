import cv2
import numpy as np
import imutils
from math import sqrt, atan2, pi


"""
На фотке находится угол, под которым наклонено больше всего прямых линий 
и затем фотка поворачивается так, чтобы все эти линии были строго вертикальными
"""

"""
Функция находит угол наклона самой большой полосы на кадре

"""
def find_angle(img, lines):
    # массив частот встречающися углов линий
    angles = []
    freqs = {} # словарь количество раз-угол
    max_length = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        angle = atan2(y2 - y1, x2 - x1)
        angle = angle * 180 / pi
        angles.append(angle)
        if length > max_length:
            max_length = length

    for angle in angles:
        times = angles.count(angle)
        if times not in freqs.values():
            freqs[times] = angle # угол - количество раз
    for times, angle in freqs.items():
        print("angle ", angle, " times ", times)

    times = freqs.keys()
    most_times = max(times)
    most_angle = freqs[most_times] # это - самый часто встречающийся угол


    return most_angle

"""
Функция поворачивает картинку на заданные угол
Поворот происходит без обрезания краев фотографии

БЕСПОЛЕЗНА
"""
def rotate(image, angle, colored):
    desired_angle = 90
    if angle < 0:
        dif = angle - abs(desired_angle)
        dif = -1 * dif
    else:
        dif = desired_angle - angle
    # если фотография цветная, то ее надо перевести в другой формат
    if colored == True:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    elif colored == False:
        gray = cv2.GaussianBlur(image, (3, 3), 0)

    edged = cv2.Canny(gray, 20, 100)

    # находим контуры на гранях
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # А НУЖНО ЛИ ЭТО ВООБЩЕ?
    # если контуры были найдены
    if len(cnts) > 0:
        # находим наибольший контур и рисуем его маску
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # находим бокс контура
        (x, y, w, h) = cv2.boundingRect(c)
        # находим ROI (region of interest)
        imageROI = image[y:y + h, x:x + w]
        # накладываем маску
        maskROI = mask[y:y + h, x:x + w]
        imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)

        # если сюда вместо image написать imageROI, то он обрежет фотку по определенному объекту на ней, а не
        # целиком её покажет
    rotated = imutils.rotate_bound(image, dif)

    return rotated


############################################################################################


if __name__ == "__main__":
    print(1)
    #img = cv2.imread(r"C:\CREESTL\Programming\PythonCoding\semestr_3\parking_lot_detection\parking_lots\ro.png", 1)

    img = imutils.resize(img, width=800)
    rows,cols, height = img.shape
    cv2.imshow("original", img)
    cv2.waitKey()


    copy_1 = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 170, 200)
    cv2.imshow("edges", edges)
    cv2.waitKey()

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=10, maxLineGap=10)
    print("Всего линий ", len(lines))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(copy_1, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imshow("lines", copy_1)
    cv2.waitKey()

    angle = find_angle(edges, lines)

    result = rotate(img, angle, True)

    cv2.imshow("result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()

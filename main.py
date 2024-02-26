# Импорт необходимых пакетов орорлор
import numpy as np
import argparse
import imutils
import cv2
import os

# настройка анализатора аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--puzzle_folder", required=True, help="Path to the folder containing puzzle images")
ap.add_argument("-w", "--waldo", required=True, help="Path to the waldo image")
args = vars(ap.parse_args())

# загрузка изображения "Waldo"
waldo = cv2.imread(args["waldo"])
print(waldo)
(waldoHeight, waldoWidth) = waldo.shape[:2]

# проход по файлам в папке головоломок
puzzle_folder = args["puzzle_folder"]
puzzle_files = os.listdir(puzzle_folder)

for puzzle_file in puzzle_files:
    # создаем полный путь к файлу головоломки
    puzzle_path = os.path.join(puzzle_folder, puzzle_file)

    # загружаем изображение головоломки
    puzzle = cv2.imread(puzzle_path)

    # поиск шаблона в головоломке
    result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)
    (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

    # извлекаем координаты области совпадения
    topLeft = maxLoc
    botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
    roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

    # создаем затемняющий слой
    mask = np.zeros(puzzle.shape, dtype="uint8")
    puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

    # делаем ярче изображение, которое ищем
    puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi

    # представляем результат
    cv2.imshow("Puzzle", imutils.resize(puzzle, height=650))
    cv2.imshow("Waldo", waldo)
    cv2.waitKey(0)

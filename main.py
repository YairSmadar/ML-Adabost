import datetime
import sys
from random import shuffle
import itertools
import math
import numpy as np
from typing import List

blue = -1
red = 1
all_left_is_red = 1
all_left_is_blue = -1


class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color  # -1: blue, 1: red
        self.weight = -1


class Line:
    def __init__(self, low: Point, high: Point, side=all_left_is_blue):
        self.low = low
        self.high = high

        if low.y > high.y:
            self.low = high
            self.high = low

        self.rate = 0
        self.side = side  # 1: all left is red, -1: all left is blue

    def is_right(self, p: Point):
        ans = True

        # check if it wrong
        if isLeft(self.low, self.high, p) and p.color == blue and self.side == all_left_is_red:
            ans = False
        elif isLeft(self.low, self.high, p) and p.color == red and self.side == all_left_is_blue:
            ans = False
        elif (not isLeft(self.low, self.high, p)) and p.color == red and self.side == all_left_is_red:
            ans = False
        elif (not isLeft(self.low, self.high, p)) and p.color == blue and self.side == all_left_is_blue:
            ans = False

        return ans


class H:
    def __init__(self, ht: List[Line], at: List[int], r: int):
        self.ht = ht
        self.at = at
        self.r = r

    def sing(self, p, numOfRolls: int):
        sum = 0

        for i in range(numOfRolls):
            is_left = isLeft(self.ht[i].low, self.ht[i].high, p)
            if (self.ht[i].side == all_left_is_red and is_left) \
                    or (self.ht[i].side == all_left_is_blue and not is_left):
                sum += self.at[i]
            else:
                sum -= self.at[i]

        if sum >= 0:
            return red
        else:
            return blue

    def is_right(self, p: Point, numOfRolls: int):
        if self.sing(p, numOfRolls) == p.color:
            return True
        else:
            return False


def isLeft(a: Point, b: Point, c: Point):
    return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) > 0


def line_rate_error(line, points: List[Point]):
    rate = 0
    for p in points:
        flag = False
        # if the rule is mistake on p
        if isLeft(line.low, line.high, p) and p.color == blue and line.side == all_left_is_red:
            flag = True
        elif isLeft(line.low, line.high, p) and p.color == red and line.side == all_left_is_blue:
            flag = True
        elif (not isLeft(line.low, line.high, p)) and p.color == red and line.side == all_left_is_red:
            flag = True
        elif (not isLeft(line.low, line.high, p)) and p.color == blue and line.side == all_left_is_blue:
            flag = True
        if flag:
            rate += p.weight
    if rate == 0:
        return 0.00000000001  # In order to be division by zero
    else:
        return rate


def best_line(points):
    best = None
    best_rate = sys.maxsize

    #  for all_left_is_blue
    for p in itertools.combinations(points, 2):
        temp = Line(p[0], p[1])
        temp_rate = line_rate_error(temp, points)
        if temp_rate < best_rate:
            best = temp
            best_rate = temp_rate

    #  for all_left_is_red
    for p in itertools.combinations(points, 2):
        temp = Line(p[0], p[1], all_left_is_red)
        temp_rate = line_rate_error(temp, points)
        if temp_rate < best_rate:
            best = temp
            best_rate = temp_rate

    return best, best_rate


def adaboost(learn, numOfRules):
    # initialized point weights
    n = len(learn)
    w = 1 / n
    for p in learn:
        p.weight = w

    Hs = []
    As = []
    for i in range(numOfRules):
        sum = 0

        # compute weighed error for each h in H
        ht, et = best_line(learn)
        Hs.append(ht)

        # set classifier weight alfa_t based in its error
        num = (1 - et) / et
        at = 0.5 * (np.log(num))
        As.append(at)

        # update point weights
        for p in learn:
            if not ht.is_right(p):
                p.weight = p.weight * (math.e ** at)
            else:
                p.weight = p.weight * (math.e ** (-at))
            sum += p.weight
        for p in learn:
            p.weight = p.weight / sum

    ans = H(Hs, As, numOfRules)
    for p in learn:
        p.weight = 1
    return ans


def run(points: list, numOfLines: int, times: int):
    """
    :param points: list of Points
    :param numOfLines: int
    :param times: int
    :return: run adaboost for each i from(1,r) (times) times with the shape
    """
    start = datetime.datetime.now()
    multi_sum_test_arr = [0]*numOfLines
    multi_sum_train_arr = [0]*numOfLines

    for j in range(1, times + 1):  # 1 - 100
        start_iter_1_100 = datetime.datetime.now()
        shuffle(points)  # mix all the point
        train = points[:75]  # 50% train
        test = points[75:]  # 50% test

        ans = adaboost(train, numOfLines)

        # in order co compute AVG
        for i in range(len(ans.ht)):
            rate_test = 0
            rate_train = 0
            for p in test:
                if ans.is_right(p, i + 1):
                    rate_test += 1
            multi_sum_test_arr[i] += (rate_test / len(test) * 100)

            for p in train:
                if ans.is_right(p, i + 1):
                    rate_train += 1
            multi_sum_train_arr[i] += (rate_train / len(train) * 100)

        end_iter_1_100 = datetime.datetime.now()
        print("Total time for {} points is: {}".format(j, end_iter_1_100 - start_iter_1_100))
        tempTest = multi_sum_test_arr.copy()
        tempTrain = multi_sum_train_arr.copy()
        for i in range(numOfLines):
            tempTest[i] /= j
            tempTrain[i] /= j
        print("multi_sum_test_arr: {}".format(tempTest))
        print("multi_sum_train_arr: {}".format(tempTrain))

    for i in range(len(multi_sum_train_arr)):
        multi_sum_test_arr[i] /= times
        multi_sum_train_arr[i] /= times

    end = datetime.datetime.now()
    for i in range(1, len(multi_sum_train_arr) + 1):
        print("the train rate of success for {} is {} percent ".format(i, multi_sum_train_arr[i]))
    print("---------------------------------------------------")
    for i in range(1, len(multi_sum_test_arr) + 1):
        print("the test rate of success for {} is {} percent ".format(i, multi_sum_test_arr[i]))

    print("Total time for {} points and {} times, from 1 to {} is :{}".format(len(points), times, numOfLines,
                                                                              end - start))


def getPointList():
    file = open("rectangle.txt", "r")
    pointList = []

    for i in range(150):
        line = file.readline()
        line = line.split(' ')
        if i != 149:
            line[2] = line[2][:-1]
        line = [float(i) for i in line]
        pointList.append(Point(line[0], line[1], line[2]))

    return pointList


def main():
    pointList = getPointList()

    run(pointList, 8, 100)


if __name__ == '__main__':
    main()

import csv
from image_parsing import generate_img_statistic
from car import Car 
from gameboard import Gameboard
from PIL import Image

def run_test_gameboard_1():
    """
    IMGS 13-18
    """
    goal_car = Car(1, 2, True, 2)
    yellow = Car(4, 0, False, 2)
    green = Car(4, 1, False, 2)
    pink = Car(3, 2, False, 3)
    blue = Car(4, 2, False, 3)
    orange = Car(5, 2, False, 3)

    expected_board = Gameboard(goal_car=goal_car, cars=[yellow, green, pink, blue, orange])

    files = []
    imgs = []
    for i in range(13, 19):
        fn = f"IMG_{i}.png"
        files.append(fn)
        img = Image.open(fn)
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data


def run_test_gameboard_2():
    """
    IMGS 20-25
    completely full
    needs to be rotated
    """
    # im_rotate = im.rotate(90)
    goal_car = Car(1, 2, True, 2)

    c_1 = Car(0, 0, True, 3)
    c_2 = Car(0, 3, True, 3)
    c_3 = Car(1, 0, True, 3)
    c_4 = Car(1, 3, True, 3)
    
    c_5 = Car(2, 0, True, 2)
    c_6 = Car(2, 2, True, 2)
    c_7 = Car(2, 4, True, 2)

    c_8 = Car(3, 0, True, 2)
    c_9 = Car(3, 2, True, 2)
    c_10 = Car(3, 4, True, 2)

    c_11 = Car(4, 0, True, 2)
    c_12 = Car(4, 2, True, 2)
    c_13 = Car(4, 4, True, 2)

    c_14 = Car(5, 0, True, 2)
    c_15 = Car(5, 2, True, 2)
    c_16 = Car(5, 4, True, 2)

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16])

    files = []
    imgs = []
    for i in range(20, 26):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data

def run_test_gameboard_3():
    """
    IMGS 26-(31_2)
    """

    goal_car = Car(1, 2, True, 2)
    c_1 = Car(0, 1, True, 2)
    c_2 = Car(0, 0, False, 2)
    c_3 = Car(1, 2, False, 2)
    c_4 = Car(3, 1, True, 3)
    c_5 = Car(0, 3, False, 3)

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5])

    files = [f"IMG_31_2.jpg"]
    imgs = [Image.open(f"IMG_31_2.jpg")]
    for i in range(26, 31):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data

def run_test_gameboard_4():
    """
    IMGS 31-34
    same gameboard as above but different colors
    """

    goal_car = Car(1, 2, True, 2)
    c_1 = Car(0, 1, True, 2)
    c_2 = Car(0, 0, False, 2)
    c_3 = Car(1, 2, False, 2)
    c_4 = Car(3, 1, True, 3)
    c_5 = Car(0, 3, False, 3)

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5])

    files = []
    imgs = []
    for i in range(31, 35):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data

def run_test_gameboard_5():
    """
    IMGS 35-40
    """

    goal_car = Car(1, 2, True, 2)
    c_1 = Car(0, 0, True, 3)
    c_2 = Car(2, 2, False, 2)
    c_3 = Car(4, 2, False, 2)
    c_4 = Car(5, 3, True, 3)
    c_5 = Car(3, 4, True, 2)
    c_6 = Car(0, 5, False, 3)
    c_7 = Car(3, 3, False, 2)

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5, c_6, c_7])

    files = []
    imgs = []
    for i in range(35, 41):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data

def run_test_gameboard_6():
    """
    IMGS 42-46
    """

    goal_car = Car(1, 2, True, 2)
    c_1 = Car(0, 0, True, 3) # blue
    c_2 = Car(1, 2, False, 2) # dark green
    c_3 = Car(0, 4, False, 3) # light green
    c_4 = Car(5, 3, True, 2) # orange bright
    c_5 = Car(4, 5, False, 2) # coral
    c_6 = Car(4, 0, False, 3) # orange
    c_7 = Car(4, 1, True, 2) # pink

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5, c_6, c_7])


    files = []
    imgs = []
    for i in range(42, 47):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data

def run_test_gameboard_7():
    """
    IMGS 47-50 (need to be rotated) (+img 90)
    """

    goal_car = Car(1, 2, True, 2)
    c_1 = Car(1, 0, True, 3) # orange
    c_2 = Car(0, 3, False, 2) # teal
    c_3 = Car(2, 2, False, 2) # green
    c_4 = Car(5, 1, True, 2) # pink
    c_5 = Car(5, 4, False, 2) # coral
    c_6 = Car(4, 4, True, 2) # yellow
    c_7 = Car(3, 5, False, 2) # purple
    c_8 = Car(5, 4, True, 2) # blue
    c_9 = Car(3, 0, False, 2) # dark orange

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9])

    files = []
    imgs = []
    for i in range(47, 51):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data


def run_test_gameboard_8():
    """
    IMGS 51-55 (need to be rotated)
    """

    goal_car = Car(1, 2, True, 2)
    c_1 = Car(0, 3, True, 3) # orange
    c_2 = Car(3, 2, False, 3) # blue
    c_3 = Car(5, 3, True, 3) # pink
    c_4 = Car(3, 3, True, 2) # coral
    c_5 = Car(1, 3, False, 2) # teal
    c_6 = Car(1, 4, False, 2) # dark orange
    c_7 = Car(3, 5, False, 3) # light green
    c_8 = Car(4, 3, True, 2) # green

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8])

    files = []
    imgs = []
    for i in range(51, 56):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data

def run_test_gameboard_9():
    """
    IMGS 57-61 (need to be rotated)
    """
    goal_car = Car(1, 2, True, 2)
    c_1 = Car(0, 2, True, 3) # blue
    c_2 = Car(0, 5, False, 3) # orange
    c_3 = Car(5, 1, True, 3) # pink
    c_4 = Car(3, 0, False, 3) # light green

    c_5 = Car(3, 4, True, 2) # green
    c_6 = Car(2, 2, False, 2) # teal
    c_7 = Car(2, 3, False, 2) # dark orange

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5, c_6, c_7])

    files = []
    imgs = []
    for i in range(57, 62):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data

def run_test_gameboard_10():
    """
    IMGS 62-64 (need to be rotated)
    """

    goal_car = Car(1, 2, True, 2)
    c_1 = Car(0, 3, True, 2) # green
    c_2 = Car(0, 5, False, 3) # orange
    c_3 = Car(3, 3, True, 3) # blue
    c_4 = Car(3, 0, False, 3) # pink
    c_5 = Car(2, 2, False, 3) # light green
    c_6 = Car(1, 3, False, 2) # teal
    c_7 = Car(5, 1, True, 2) # dark orange

    expected_board = Gameboard(goal_car=goal_car, cars=[c_1, c_2, c_3, c_4, c_5, c_6, c_7])

    files = []
    imgs = []
    for i in range(62, 65):
        fn = f"IMG_{i}.jpg"
        files.append(fn)
        img = Image.open(fn)
        img.rotate(90) # counter clockwise
        imgs.append(img)

    data = []
    for fn, img in zip(files, imgs):
        d = {}
        try:
            eq, time = generate_img_statistic(img, expected_board)
            d["img"] = fn
            d["board_matched"] = eq
            d["time"] = time
            data.append(d)
        except:
            d["img"] = fn
            d["board_matched"] = False
            d["time"] = -1
            data.append(d)

    return data

def run_test_noisy():
    """
    IMGS 65-68 (need to be rotated)
    """
    pass

def run_tests():

    data = []
    data.extend(run_test_gameboard_1())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 1 done")

    data.extend(run_test_gameboard_2())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 2 done")

    data.extend(run_test_gameboard_3())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 3 done")

    data.extend(run_test_gameboard_4())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 4 done")

    data.extend(run_test_gameboard_5())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 5 done")

    data.extend(run_test_gameboard_6())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 6 done")

    data.extend(run_test_gameboard_7())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 7 done")

    data.extend(run_test_gameboard_8())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 8 done")

    data.extend(run_test_gameboard_9())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 9 done")

    data.extend(run_test_gameboard_10())
    dict_list_to_csv(data, "statistics.csv")
    print("gameboard 10 done")

def dict_list_to_csv(dict_list, filename):
    if not dict_list:
        print("Empty dictionary list. Nothing to write.")
        return
    
    # Extract fieldnames from the keys of the first dictionary
    fieldnames = dict_list[0].keys()
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write the data rows
        for data in dict_list:
            writer.writerow(data)
    
    print(f"CSV file '{filename}' created successfully.")
        

run_tests()
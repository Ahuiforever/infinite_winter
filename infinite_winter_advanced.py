# import requirements
# local
import time
import tkinter as tk

# opencv
import cv2
# numpy
import numpy as np
# psutil
import psutil
# pyautogui
import pyautogui
import win32api
# pywin23
import win32gui
# pywinauto
from pywinauto import Application

BIAS = (100, 100)


def print_memory_info():
    # 获取虚拟内存信息
    memory_info = psutil.virtual_memory()

    # 获取交换空间信息
    swap_info = psutil.swap_memory()

    print("Memory Information:")
    print(f"Total memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used memory: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"Memory usage: {memory_info.percent}%")

    print("\nSwap Information:")
    print(f"Total swap space: {swap_info.total / (1024 ** 3):.2f} GB")
    print(f"Used swap space: {swap_info.used / (1024 ** 3):.2f} GB")
    print(f"Free swap space: {swap_info.free / (1024 ** 3):.2f} GB")
    print(f"Swap usage: {swap_info.percent}%")


class Coordinates:
    def __init__(self, coord_dict: dict):
        super().__init__()
        for key, value in coord_dict.items():
            # key: str, value: list[tuple, tuple] | tuple[int, int]
            if 'check' in key:
                coord_dict[key][0]: tuple[int, int] = self.to_relative(value[0])

        self.coord_dict = coord_dict

    @staticmethod
    def to_relative(xy: tuple):
        return xy[0] - BIAS[0], xy[1] - BIAS[1]

    @staticmethod
    def to_absolute(xy: tuple):
        return xy[0] + BIAS[0], xy[1] + BIAS[1]

    def __getitem__(self, item: str) -> tuple[int, int] | list[tuple[int, int], tuple[int, int, int]]:
        return self.coord_dict[item]

    def __setitem__(self, key, value):
        # Handle item assignment
        self.coord_dict[key] = value  # Store the value in the dictionary


class Matcher:
    def __init__(self, win):
        super().__init__()
        self.win = win

    # @staticmethod
    def color(self, xy: tuple, target_color: tuple, threshold: float = 0.9) -> bool:
        # try:
        #     print_memory_info()
        screen_shot = self.win.capture_as_image()
        # except Exception as e:
        #     print(e)
        #     print(datetime.now())
        # screen_shot = pyautogui.screenshot(region=win32gui.GetWindowRect(self.hwnd))
        image = np.array(screen_shot)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        pixel_color = image[xy[1], xy[0]]  # height=x, width=y
        color_dist = np.linalg.norm(np.array(pixel_color) - np.array(target_color))
        max_dist = 441.67
        similarity = 1 - (color_dist / max_dist)
        if similarity >= threshold:
            return True
        else:
            return False

    def image(self, template_path, threshold=0.9, scale_factor=0.9):
        if template_path is None:
            template_path = 'donate_tip_4kx150.png'
        template = cv2.imread(template_path)
        target = self.win.capture_as_image()
        target = np.array(target)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        w, h = template.shape[:2][::-1]
        wh = (w, h)
        scale = 1.0
        matches = []

        while w > 15 and h > 15:
            # match
            result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)

            # filter
            max_loc = np.unravel_index(np.argmax(result), result.shape)
            loc = np.where(np.atleast_1d(result[max_loc]) >= threshold)

            # result
            if loc == ([0],):
                matches.extend(max_loc[::-1])
                break

            # resize the template image
            scale *= scale_factor
            w = int(w * scale)
            h = int(h * scale)
            template = cv2.resize(template, (w, h))

        if len(matches) > 0:
            # for pt in zip(*loc[::-1]):  # loc[::-1] 转换成 (x, y) 格式
            #     cv2.rectangle(target, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            # cv2.imshow('Matches', target)
            # cv2.waitKey(0)
            return matches[0] + wh[0], matches[1] + wh[1]
        else:
            result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
            max_loc = np.unravel_index(np.argmax(result), result.shape)
            loc = np.where(np.atleast_1d(result[max_loc]) >= threshold)
            # for pt in zip(*loc[::-1]):  # loc[::-1] 转换成 (x, y) 格式
            #     cv2.rectangle(target, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            # cv2.imshow('Matches', target)
            # cv2.waitKey(0)

            if loc == ([0],):
                # return [(pt[0] + w // 2, pt[1] + h // 2) for pt in zip(*loc[::-1])]
                matches.extend(max_loc[::-1])
                return matches[0] + wh[0], matches[1] + wh[1]
            else:
                return


class AutoTab:
    def __init__(self, win, coord: Coordinates):
        super().__init__()
        self.win = win
        self.coord = coord
        self.match = Matcher(win)

    def triangle_check(self):
        if not self.match.color(self.coord['tri_check'][0], self.coord['tri_check'][1]):
            while True:
                pyautogui.click(self.coord['tri'][0], self.coord['tri'][1])
                time.sleep(1)
                if self.match.color(self.coord['tri_check'][0], self.coord['tri_check'][1]):
                    if not self.match.color(self.coord['town_check'][0], self.coord['town_check'][1]):
                        pyautogui.click(self.coord['town'][0], self.coord['town'][1])
                        time.sleep(1)
                    break
                self.back_check()
        else:
            if not self.match.color(self.coord['town_check'][0], self.coord['town_check'][1]):
                pyautogui.click(self.coord['town'][0], self.coord['town'][1])
                time.sleep(1)

    def world_check(self):
        if self.match.color(self.coord['town_logo_check'][0], self.coord['town_logo_check'][1]):
            pyautogui.click(self.coord['town_logo'][0], self.coord['town_logo'][1])
            time.sleep(4)

    def back_check(self):
        if self.match.color(self.coord['back_check'][0], self.coord['back_check'][1]):
            pyautogui.click(self.coord['back'][0], self.coord['back'][1])
            time.sleep(1)

    def train_status_check(self, mode: str):
        mode_check = mode + '_check'
        if self.match.color(self.coord[mode_check][0], self.coord[mode_check][1]):
            pyautogui.click(self.coord[mode][0], self.coord[mode][1])
            time.sleep(1)
            pyautogui.click(self.coord['point'][0], self.coord['point'][1])
            time.sleep(1)
            pyautogui.click(self.coord['tri'][0], self.coord['tri'][1])
            time.sleep(1)
            pyautogui.click(self.coord[mode][0], self.coord[mode][1])
            time.sleep(1)
            pyautogui.click(self.coord['train'][0], self.coord['train'][1])
            time.sleep(2)
            pyautogui.click(self.coord['long_train'][0], self.coord['long_train'][1])
            time.sleep(1)
            pyautogui.click(self.coord['back'][0], self.coord['back'][1])
            time.sleep(2)

    def train(self):
        for mode in ('dun', 'mao', 'she'):
            self.train_status_check(mode)
            self.triangle_check()

    def help(self):
        if self.match.color(self.coord['help_check'][0], self.coord['help_check'][1]):
            pyautogui.click(self.coord['help'][0], self.coord['help'][1])

    @staticmethod
    def _drag_up(s):
        pyautogui.moveTo(s[0], s[1])
        pyautogui.mouseDown()
        pyautogui.drag(0, - s[1], duration=0.5)
        pyautogui.mouseUp()
        time.sleep(2)

    @staticmethod
    def _drag_down(s):
        pyautogui.moveTo(s[0], s[1])
        pyautogui.mouseDown()
        pyautogui.drag(0, s[1], duration=0.5)
        pyautogui.mouseUp()
        time.sleep(2)

    def warehouse(self):
        s: tuple[int, int] = self.coord['drag_point']
        self._drag_up(s)

        # donate
        if self.donate():
            self.triangle_check()
            self._drag_up(s)

        # collect
        if self.match.color(self.coord['warehouse_check'][0], self.coord['warehouse_check'][1]):
            pyautogui.click(self.coord['warehouse'][0], self.coord['warehouse'][1])
            time.sleep(1)
            pyautogui.click(self.coord['point'][0], self.coord['point'][1])
            time.sleep(2)
            pyautogui.click(self.coord['blank'][0], self.coord['blank'][1])
            time.sleep(2)
        else:
            self._drag_down(s)

    def assemble(self):
        if self.match.color(self.coord['assemble_check'][0], self.coord['assemble_check'][1]):
            pyautogui.click(self.coord['assemble'][0], self.coord['assemble'][1])
            time.sleep(1.5)
            if self.match.color(self.coord['join_check'][0], self.coord['join_check'][1]):
                pyautogui.click(self.coord['join'][0], self.coord['join'][1])
                time.sleep(1.5)
                if self.match.color(self.coord['need_more_queue_check'][0], self.coord['need_more_queue_check'][1]):
                    pyautogui.click(self.coord['close_need_more_queue'][0], self.coord['close_need_more_queue'][1])
                    time.sleep(1)
                    pyautogui.click(self.coord['back'][0], self.coord['back'][1])
                elif self.match.color(self.coord['march_out_check'][0], self.coord['march_out_check'][1]):
                    pyautogui.click(self.coord['average'][0], self.coord['average'][1])
                    time.sleep(1)
                    pyautogui.click(self.coord['march_out'][0], self.coord['march_out'][1])
                    time.sleep(1)
                    if self.match.color(self.coord['surpass_capacity_check'][0],
                                        self.coord['surpass_capacity_check'][1]):
                        pyautogui.click(self.coord['close_surpass_capacity'][0],
                                        self.coord['close_surpass_capacity'][0])
                        time.sleep(1)
                    # elif self.match.color(self.coord['accelerate_check'][0],
                    #                       self.coord['accelerate_check'][1]):
                    #     pyautogui.click(self.coord['close_accelerate'][0],
                    #                     self.coord['close_accelerate'][0])
                    #     time.sleep(1)
                    pyautogui.click(self.coord['back'][0], self.coord['back'][1])
            else:
                pyautogui.click(self.coord['back'][0], self.coord['back'][1])

    def donate(self):
        if (
                (self.match.color(self.coord['donate1_check'][0], self.coord['donate1_check'][1]) and
                 not self.match.color(self.coord['recruit_check'][0], self.coord['recruit_check'][1])) or
                (self.match.color(self.coord['donate2_check'][0], self.coord['donate2_check'][1]) and
                 not self.match.color(self.coord['tech_research_check'][0], self.coord['tech_research_check'][1]))
        ):
            pyautogui.click(self.coord['alliance'][0], self.coord['alliance'][1])
            time.sleep(1)
            pyautogui.click(self.coord['tech'][0], self.coord['tech'][1])
            time.sleep(1)
            donate_tip = self.match.image(None, threshold=0.9)
            if donate_tip:  # if there exits donate recommendation
                donate_tip = self.coord.to_absolute((donate_tip[0], donate_tip[1]))
                pyautogui.click(donate_tip[0], donate_tip[1])
                time.sleep(1)
                # pyautogui.position(self.coord['donating'][0], self.coord['donating'][0])
                # pyautogui.click(self.coord['donating'][0], self.coord['donating'][0], duration=10)
                pyautogui.mouseDown(self.coord['donating'][0], self.coord['donating'][1])
                time.sleep(10)
                pyautogui.mouseUp()
                time.sleep(1)
                pyautogui.click(self.coord['close_donation'][0], self.coord['close_donation'][1])
                time.sleep(1)
                pyautogui.click(self.coord['back'][0], self.coord['back'][1])
                time.sleep(1)
                pyautogui.click(self.coord['back'][0], self.coord['back'][1])
                time.sleep(1)
            else:
                pyautogui.click(self.coord['back'][0], self.coord['back'][1])
                time.sleep(1)
                pyautogui.click(self.coord['back'][0], self.coord['back'][1])
                time.sleep(1)
            return True
        else:
            return False

    def explore(self):
        if self.match.color(self.coord['explore_check'][0], self.coord['explore_check'][1]):
            pyautogui.click(self.coord['explore'][0], self.coord['explore'][1])
            time.sleep(1)
            pyautogui.click(self.coord['collect'][0], self.coord['collect'][1])
            time.sleep(1)
            pyautogui.click(self.coord['long_collect'][0], self.coord['long_collect'][1])
            time.sleep(1)
            pyautogui.click(self.coord['blank'][0], self.coord['blank'][1])
            time.sleep(1)
            pyautogui.click(self.coord['back'][0], self.coord['back'][1])
            time.sleep(1)

    def filter_for_snow_monster(self):
        ...

    def __call__(self):
        self.world_check()
        self.triangle_check()
        self.train()
        self.help()
        self.warehouse()  # donate
        self.assemble()
        self.explore()
        time.sleep(10)


def get_dpi_scaling():
    root = tk.Tk()
    dpi_ = root.winfo_fpixels('1i')
    w, h = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
    return (w, h), dpi_


def main(coord: Coordinates):
    # get handle
    hwnd = win32gui.FindWindow(None, '无尽冬日')
    l, t, r, b = win32gui.GetWindowRect(hwnd)

    win32gui.MoveWindow(hwnd, BIAS[0], BIAS[1], r - l, b - t, True)

    # connect window
    app = Application().connect(handle=hwnd)
    window = app.window(handle=hwnd)
    window.set_focus()

    # auto tap
    auto = AutoTab(window, coord)

    while True:
        auto()


if __name__ == '__main__':
    # initialize coordinate
    # 4k, 150%
    c4kx150 = Coordinates({
        # point_name: (x, y)
        # point_check: [(x, y), (R, G, B)]
        'tri': (125, 855),
        'tri_check': [(690, 850), (255, 255, 255)],
        'town': (180, 500),
        'town_check': [(180, 500), (96, 171, 230)],
        'town_logo': (915, 1645),
        'town_logo_check': [(915, 1645), (255, 254, 192)],
        'dun': (632, 842),
        'dun_check': [(632, 842), (255, 30, 31)],
        'mao': (632, 934),
        'mao_check': [(632, 934), (255, 30, 31)],
        'she': (632, 1025),
        'she_check': [(632, 1025), (255, 30, 31)],
        'point': (560, 890),
        'train': (710, 1230),
        # 'train_check': [(1195, 1505), (96, 143, 216)],
        'long_train': (770, 1670),
        # 'long_train_check': [(1413, 2033), (79, 165, 252)],
        # 'training_check': [(968, 1667), (39, 89, 141)],
        'back': (160, 210),
        'back_check': [(160, 210), (203, 255, 255)],
        'help': (780, 1550),
        'help_check': [(780, 1550), (22, 169, 50)],
        'drag_point': (400, 900),
        'warehouse': (615, 1180),
        'warehouse_check': [(615, 1180), (31, 255, 108)],
        'blank': (570, 1400),
        'assemble': (940, 840),
        'assemble_check': [(940, 840), (255, 255, 255)],
        'join': (910, 650),
        'join_check': [(910, 650), (255, 255, 255)],
        'march_out': (800, 1700),
        'march_out_check': [(890, 1680), (79, 165, 252)],
        'close_need_more_queue': (910, 590),
        'need_more_queue_check': [(910, 590), (197, 220, 255)],
        'close_surpass_capacity': (905, 725),
        'surpass_capacity_check': [(905, 725), (197, 220, 255)],
        'average': (360, 1640),
        'donate1': (620, 825),
        'donate2': (620, 684),
        'donate1_check': [(632, 804), (255, 30, 31)],
        'donate2_check': [(632, 663), (255, 30, 31)],
        'recruit_check': [(162, 832), (177, 114, 52)],
        'tech_research_check': [(162, 832), (255, 255, 255)],
        'alliance': (780, 1685),
        'tech': (800, 1335),
        'donating': (750, 1450),
        'close_donation': (910, 415),
        'explore': (200, 1690),
        'explore_check': [(235, 1650), (255, 30, 31)],
        'collect': (882, 1317),
        'long_collect': (560, 1320),
        'close_accelerate': (985, 725),
        'accelerate_check': [(985, 725), (197, 220, 255)],
        'snow_monster_check': [(235, 620), (126, 49, 21)]
    })
    # 1k, 100%
    c1kx100 = Coordinates({
        # point_name: (x, y)
        # point_check: [(x, y), (R, G, B)]
        'tri': (115, 485),
        'tri_check': [(398, 487), (255, 255, 255)],
        'town': (150, 310),
        'town_check': [(150, 310), (95, 171, 230)],
        'town_logo': (510, 900),
        'town_logo_check': [(510, 880), (255, 253, 185)],
        'dun': (368, 482),
        'dun_check': [(368, 482), (255, 30, 31)],
        'mao': (368, 528),
        'mao_check': [(368, 528), (255, 30, 31)],
        'she': (368, 573),
        'she_check': [(368, 573), (255, 30, 31)],
        'point': (335, 500),
        'train': (435, 650),
        # 'train_check': [(1195, 1505), (96, 143, 216)],
        'long_train': (440, 900),
        # 'long_train_check': [(1413, 2033), (79, 165, 252)],
        # 'training_check': [(968, 1667), (39, 89, 141)],
        'back': (130, 170),
        'back_check': [(130, 170), (203, 255, 255)],
        'help': (440, 835),
        'help_check': [(440, 835), (22, 169, 50)],
        'drag_point': (200, 500),
        'warehouse': (358, 650),
        'warehouse_check': [(358, 650), (31, 255, 108)],
        'blank': (335, 770),
        'assemble': (525, 480),
        'assemble_check': [(525, 480), (255, 255, 255)],
        'join': (500, 380),
        'join_check': [(500, 380), (37, 183, 86)],
        'march_out': (450, 900),
        'march_out_check': [(485, 900), (79, 165, 252)],
        'close_need_more_queue': (508, 355),
        'need_more_queue_check': [(508, 355), (197, 220, 255)],
        'close_surpass_capacity': (448, 401),
        'surpass_capacity_check': [(448, 401), (187, 209, 242)],
        'average': (230, 880),
        'donate1': (368, 462),
        'donate2': (368, 392),
        'donate1_check': [(368, 462), (255, 30, 31)],  # lower donate
        'donate2_check': [(368, 392), (255, 30, 31)],
        'recruit_check': [(132, 476), (113, 79, 62)],  # in case of false detect of hero_recruit
        'tech_research_check': [(162, 832), (255, 255, 255)],  # todo: 没改
        'alliance': (440, 900),
        'tech': (440, 730),
        'donating': (420, 785),
        'close_donation': (506, 266),
        'explore': (150, 900),
        'explore_check': [(169, 885), (255, 30, 31)],
        'collect': (490, 720),
        'long_collect': (330, 720),

        # 没改
        'close_accelerate': (985, 725),
        'accelerate_check': [(985, 725), (197, 220, 255)],
        'snow_monster_check': [(235, 620), (126, 49, 21)]
    })
    resolution, dpi = get_dpi_scaling()
    # dpi, 96: 100%, 144: 150%, 192: 200%
    if resolution == (3840, 2160) and int(dpi) == 144:
        c = c4kx150
    elif resolution == (1920, 1080) and int(dpi) == 96:
        c = c1kx100
    else:
        raise EnvironmentError(f'Expected to be used on 4kx150 or 1kx100 screen, '
                               f'got resolution of {resolution} with {dpi} dpi scaling.')
    main(c)

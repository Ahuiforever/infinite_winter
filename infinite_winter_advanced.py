# import requirements
# local
import time

# pywin23
import win32gui
import win32api
import pyautogui
from pywinauto import Application

# numpy
import numpy as np

# opencv
import cv2

BIAS = (100, 100)


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
    def color(self, xy: tuple, target_color: tuple) -> bool:
        screen_shot = self.win.capture_as_image()
        # screen_shot = pyautogui.screenshot()
        image = np.array(screen_shot)
        pixel_color = image[xy[1], xy[0]]  # height=x, width=y
        if tuple(pixel_color) == target_color:
            return True
        return False

    def image(self, template_path, threshold=0.8, scale_factor=0.9):
        if template_path is None:
            template_path = 'donate_tip_2.png'
        template = cv2.imread(template_path)
        target = self.win.capture_as_image()
        target = np.array(target)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        w, h = template.shape[:2][::-1]
        scale = 1.0
        matches = []

        # while w > 30 and h > 30:  # 如果模板图像尺寸过小，则停止缩放
        #     # 在目标图像中进行模板匹配
        #     result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        #
        #     # 找到匹配度大于 threshold 的位置
        #     loc = np.where(result >= threshold)
        #
        #     # 记录匹配结果
        #     if len(loc[0]) > 0:
        #         matches.extend(list(zip(*loc[::-1])))
        #
        #     # 缩小模板图像
        #     scale *= scale_factor
        #     w = int(w * scale)
        #     h = int(h * scale)
        #     template = cv2.resize(template, (w, h))

        if len(matches) > 0:
            return matches
        else:
            result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)
            # for pt in zip(*loc[::-1]):  # loc[::-1] 转换成 (x, y) 格式
            #     cv2.rectangle(target, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            # cv2.imshow('Matches', target)
            # cv2.waitKey(0)

            if len(loc[0]) > 0:
                # return [(pt[0] + w // 2, pt[1] + h // 2) for pt in zip(*loc[::-1])]
                return list(zip(*loc[::-1]))
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
                    elif self.match.color(self.coord['accelerate_check'][0],
                                          self.coord['accelerate_check'][1]):
                        pyautogui.click(self.coord['close_accelerate'][0],
                                        self.coord['close_accelerate'][0])
                        time.sleep(1)
                    pyautogui.click(self.coord['back'][0], self.coord['back'][1])
            else:
                pyautogui.click(self.coord['back'][0], self.coord['back'][1])

    def donate(self):
        if (
                (self.match.color(self.coord['donate1_check'][0], self.coord['donate1_check'][1]) and
                 not self.match.color(self.coord['recruit_check'][0], self.coord['recruit_check'][1])) or
                self.match.color(self.coord['donate2_check'][0], self.coord['donate2_check'][1])
        ):
            pyautogui.click(self.coord['alliance'][0], self.coord['alliance'][1])
            time.sleep(1)
            pyautogui.click(self.coord['tech'][0], self.coord['tech'][1])
            time.sleep(1)
            donate_tip = self.match.image(None, threshold=0.9)
            if donate_tip:  # if there exits donate recommendation
                donate_tip = self.coord.to_absolute((donate_tip[0][0], donate_tip[0][1]))
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
        # self.back_check()
        time.sleep(10)


def main(coord: Coordinates):
    # get screen resolution
    w, h = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(0)

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
    c = Coordinates({
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
    })
    main(c)

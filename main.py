import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tiny_functions import *
SCREEN_SIZE = (1920, 1080)
NEWSIE = (350, 350)
ICON_SIZE = (100, 100)
RADIUS = 10

class Object:
    def __init__(self, im: Image, icon: Image, center: list, front_and_back: list, color_point: str = 'white',
                 x_m: float = 0, y_m: float = 0, vx: float = 0, vy: float = 0, name: str = ""):
        self.im = im
        self.icon = icon
        self.center = center
        self.front_and_back = front_and_back
        self.color_point = color_point
        self.name = name

        # Коэффициенты ПД-регулятора
        self.k = None
        self.v_1_past = None
        self.x_safe = 2.5

        self.y_list = [y_m]
        self.front_list = [y_m + front_and_back[0]]
        self.back_list = [y_m + front_and_back[1]]
        self.x_m = x_m
        self.y_m = y_m
        self.vx_kmh = vx
        self.vy_kmh = vy

    def meter2pixel(self, h: int, x_vec, y_vec, y_past: float = 0, point: str = 'center'):
        x_m, y_m = (self.x_m, self.y_m)
        if point == 'front':
            y_m += self.front_and_back[0]
        if point == 'back':
            y_m += self.front_and_back[1]
        tmp = x_m * h * x_vec + (y_m * h + y_past) * y_vec
        return int(round(tmp[0])), int(round(tmp[1]))

    def kmh2pxs(self, h: int):
        return np.array([self.vx_kmh, self.vy_kmh]) / 3.6 * h

    def process(self, dt: float, a: any = (0., 0.)):
        self.vx_kmh += a[0] * dt
        self.vy_kmh += a[1] * dt
        self.x_m += self.vx_kmh / 3.6 * dt
        self.y_m += self.vy_kmh / 3.6 * dt
        self.y_list += [self.y_m]
        self.front_list += [self.y_m + self.front_and_back[0]]
        self.back_list += [self.y_m + self.front_and_back[1]]

    def add_to_draw(self, im: Image, draw: ImageDraw, x_vec, y_vec, y_past: float, d: int):
        w, h = SCREEN_SIZE
        x, y = self.meter2pixel(d, x_vec, y_vec, y_past)
        x_screen = int(w//2) - self.center[0]
        y_screen = int(h//2) - self.center[1]
        im.paste(self.im, (x_screen - x, y_screen - y), self.im)
        im.paste(self.icon, (x_screen - x + self.center[0] + 20, y_screen - y + 20), self.icon)
        for xy in [self.meter2pixel(d, x_vec, y_vec, y_past, point='front'),
                   self.meter2pixel(d, x_vec, y_vec, y_past, point='back'), (x, y)]:
            draw.ellipse((int(w//2) - xy[0] - RADIUS, int(h//2) - xy[1] - RADIUS,
                         int(w//2) - xy[0] + RADIUS, int(h//2) - xy[1] + RADIUS),
                         fill=self.color_point, outline=(0, 0, 0))

class FrontCar(Object):
    def new_process(self, dt: float, t: float, v0):
        self.vx_kmh = 0.
        self.vy_kmh = v0 + 1 * np.sin(t / 5.)
        self.x_m += self.vx_kmh / 3.6 * dt
        self.y_m += self.vy_kmh / 3.6 * dt
        self.y_list += [self.y_m]
        self.front_list += [self.y_m + self.front_and_back[0]]
        self.back_list += [self.y_m + self.front_and_back[1]]

class MyCar(Object):
    def get_acceleration(self, obj: any, a_max: float = 1e20, dt: float = 0.5):
        v_diff = (self.vy_kmh - obj.vy_kmh)  # + 0.5 * abs(obj.vy_kmh)
        safe_distance = self.x_safe + 1e1 * abs(v_diff)
        x_diff = - ((obj.y_m + obj.front_and_back[1]) - (self.y_m + self.front_and_back[0]) - safe_distance)
        if self.k is None:
            print(x_diff, v_diff)
            if self.v_1_past is None:
                self.v_1_past = obj.vy_kmh
            C_1 = x_diff
            C_2 = v_diff
            print(f"2 * C_1 {2 * C_1}")
            self.k = 2 * (C_2 / (2 * C_1 + self.x_safe)) ** 2
            print(self.k)

        k_x = self.k
        k_v = 2 * np.sqrt(self.k)
        a_front = (obj.vy_kmh - self.v_1_past) / dt
        a = - k_x * x_diff - k_v * v_diff + a_front  # + clip(a_front, -1e1, 1e1)
        self.v_1_past = obj.vy_kmh
        return [0, clip(a, -a_max, a_max)]

class Display:
    def __init__(self, len_road_meter: int):
        # Изображения
        self.images = {'background': Image.open("img/back.png"),
                       'cars': [Image.open("img/car1.png").resize(NEWSIE), Image.open("img/car2.png").resize(NEWSIE),
                                Image.open("img/car_heavy.png"), Image.open("img/car_police.png").resize(NEWSIE)],
                       'icons': [Image.open("img/icon_pc.png").resize(ICON_SIZE),
                                 Image.open("img/icon_human.png").resize(ICON_SIZE)]}

        self.h_pixels = 150
        self.squares_in_screen = 8
        self.len_road_pix = len_road_meter * self.h_pixels
        self.extra_h_y = len_road_meter
        self.directions = {'x': np.array([1/np.sqrt(2), -1/2]),
                           'y': np.array([-1/np.sqrt(2), -1/2]),
                           'z': np.array([0., 1.])}

class Process:
    def __init__(self, dt: float = 1., vy: float = 10, frames: int = 10, saving: bool = False):
        # Общие параметры
        self.saving = saving
        self.dt = dt
        self.vy = vy
        self.frames = frames
        self.counter = 0
        self.t = 0
        self.y = 0

        # Классы
        self.d = Display(len_road_meter=int(round(dt * vy * frames)))

        # Объекты
        self.front_car = FrontCar(self.d.images['cars'][0], self.d.images['icons'][1],
                                  [int(NEWSIE[0]//2), int(NEWSIE[1]//2) + 20], x_m=0, y_m=2, vy=self.vy,
                                  front_and_back=[1.1, -1.1], color_point='cyan', name="Машина спереди")
        self.my_car = MyCar(self.d.images['cars'][1], self.d.images['icons'][0],
                            [int(NEWSIE[0]//2), int(NEWSIE[1]//2) + 40], x_m=0, y_m=-1, vy=self.vy - 0.2,
                            front_and_back=[1., -1.], color_point='magenta', name="Грузовик")
        self.my_car_2 = MyCar(self.d.images['cars'][2], self.d.images['icons'][0], [256, 256 + 50],
                              x_m=0, y_m=-5, front_and_back=[1.8, -1.8], vy=self.vy - 0.4,
                              color_point='navy', name="Фура")
        self.objects = [self.front_car, self.my_car, self.my_car_2]

    def add_lines(self, img):
        color = "#505050"
        d = self.d.h_pixels
        w, h = SCREEN_SIZE
        x = self.d.directions['x']
        y = self.d.directions['y']
        n_x = self.d.squares_in_screen
        n_y = n_x + self.d.extra_h_y
        for i_y in arrange_both(n_y) + list(range(n_y)[n_x:n_y]):
            a = [i_y*d*y + n_x*d*x - y*self.y, i_y*d*y - n_x*d*x - y*self.y]
            img.line([a[0][0] + w//2, a[0][1] + h//2, a[1][0] + w//2, a[1][1] + h//2], fill=color, width=3)
        for i_x in arrange_both(n_x):
            c = ('#6E6E6E' if i_x == 0 else '#C8C8C8') if abs(i_x) < 2 else color
            width = (240 if i_x == 0 else 10) if abs(i_x) < 2 else 3
            a = [i_x*d*x + n_x*d*y - y*self.y, i_x*d*x - n_y*d*y - y*self.y]
            img.line([a[0][0] + w//2, a[0][1] + h//2, a[1][0] + w//2, a[1][1] + h//2], fill=c, width=width)

    def show(self):
        screen = self.d.images['background'].copy()
        draw = ImageDraw.Draw(screen)
        self.add_lines(draw)
        self.my_car_2.add_to_draw(screen, draw, self.d.directions['x'], self.d.directions['y'], self.y, self.d.h_pixels)
        self.my_car.add_to_draw(screen, draw, self.d.directions['x'], self.d.directions['y'], self.y, self.d.h_pixels)
        self.front_car.add_to_draw(screen, draw, self.d.directions['x'], self.d.directions['y'], self.y,
                                   self.d.h_pixels)
        return screen

    def process(self):
        for i in range(self.frames):
            self.counter += 1
            self.t = self.counter * self.dt
            v = self.vy / 3.6 * self.d.h_pixels
            self.y = - i * v * self.dt

            # Процессы объектов
            self.front_car.new_process(self.dt, self.t, self.vy)
            self.my_car.process(dt=self.dt, a=self.my_car.get_acceleration(self.front_car))
            self.my_car_2.process(dt=self.dt, a=self.my_car_2.get_acceleration(self.my_car))
            # self.my_car.process(dt=self.dt)
            # self.my_car_2.process(dt=self.dt)

            # Отображение
            if self.saving:
                self.show().save(f"res/result_{self.counter:03}.jpg")

    def plot_process(self):
        colors = ['navy', 'teal', 'aqua']
        for j in range(len(self.objects)):
            t = [i * self.dt for i in range(len(self.objects[j].y_list))]
            y = [self.objects[j].front_list[i] * (i % 2) + self.objects[j].back_list[i] * ((i+1) % 2)
                 for i in range(len(self.objects[j].y_list))]
            plt.plot(t, y, label=self.objects[j].name)
            # plt.plot(t, self.objects[j].y_list, label=self.objects[j].name, c=colors[j])
            # plt.plot(t, self.objects[j].front_list, label=self.objects[j].name, c=colors[j])
            # plt.plot(t, self.objects[j].back_list, c=colors[j])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    p = Process(frames=30000, dt=0.1, vy=1e0, saving=False)
    # p.show().show()
    p.process()
    p.plot_process()

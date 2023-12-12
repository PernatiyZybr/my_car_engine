import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tiny_functions import *
SCREEN_SIZE = (1920, 1080)
ICON_SIZE = (100, 100)
NEWSIE = (350, 350)
RADIUS = 10

class Object:
    """Подвижный объект в численном интегрировании"""
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
        self.x_1_past = None
        self.x_2_past = None
        self.a_1_past = None
        self.x_safe = 1.5

        self.y_list = [y_m]
        self.a_list = []
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
        """Интегрирование движения схемой 'Уголок'"""
        self.vx_kmh += a[0] * dt
        self.vy_kmh += a[1] * dt
        self.vy_kmh = clip(self.vy_kmh, 0, 1e10)
        self.x_m += self.vx_kmh / 3.6 * dt
        self.y_m += self.vy_kmh / 3.6 * dt
        self.y_list += [self.y_m]
        self.front_list += [self.y_m + self.front_and_back[0]]
        self.back_list += [self.y_m + self.front_and_back[1]]

    def add_to_draw(self, im: Image, draw: ImageDraw, x_vec, y_vec, y_past: float, d: int):
        """Добавление объекта на картинку"""
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
    """Единичный объект с заданным управлением"""
    def new_process(self, dt: float, t: float, v0):
        self.vx_kmh = 0.
        self.vy_kmh = v0 + v0 * np.sin(t / 1.)
        self.x_m += self.vx_kmh / 3.6 * dt
        self.y_m += self.vy_kmh / 3.6 * dt
        self.y_list += [self.y_m]
        self.front_list += [self.y_m + self.front_and_back[0]]
        self.back_list += [self.y_m + self.front_and_back[1]]

class MyCar(Object):
    """Преследующий объект с управлением с обратной связью"""
    def get_acceleration(self, obj: any, dt: float, a_max: float = 1e3, da_max: float = 1e20, noise: float = 0.01):
        """Функция задания управляющего ускорения"""
        x_1 = get_noise_distance(distance=obj.y_m + obj.front_and_back[1], sigma=noise)
        if self.k is None:  # Определение параметров на первом шаге
            self.x_1_past = x_1
            self.x_2_past = self.x_1_past
            self.a_1_past = 0.
        v_1 = (x_1 - self.x_1_past) / dt
        a_1 = (x_1 - 2*self.x_1_past + self.x_2_past) / dt
        v_diff = v_1 - self.vy_kmh
        safe_distance = self.x_safe  # + 0.2 * abs(v_1)  # Можно добавить динамическое расстояние преследования
        x_diff = x_1 - (self.y_m + self.front_and_back[0]) - safe_distance

        if self.k is None:  # Определение параметров на первом шаге
            self.k = v_diff**2 / x_diff**2
            print(f"{self.name}: k={self.k}")
        k_x = self.k
        k_v = 2 * np.sqrt(self.k)
        a = self.a_1_past + clip(k_x * x_diff + k_v * v_diff - self.a_1_past, -da_max*dt, da_max*dt)

        a = clip(a, -a_max, a_max)
        self.a_1_past = a
        self.a_list += [a]
        self.x_2_past = self.x_1_past
        self.x_1_past = x_1
        return [0, a]

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
    def __init__(self, dt: float = 1., vy: float = 10, frames: int = 10, saving: bool = False, many_cars: bool = False):
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
                                  [int(NEWSIE[0]//2), int(NEWSIE[1]//2) + 20], x_m=0, y_m=5, vy=self.vy,
                                  front_and_back=[1.1, -1.1], color_point='cyan', name="Машина спереди")
        self.my_car = MyCar(self.d.images['cars'][1], self.d.images['icons'][0],
                            [int(NEWSIE[0]//2), int(NEWSIE[1]//2) + 40], x_m=0, y_m=1, vy=self.vy - 0.2,
                            front_and_back=[1., -1.], color_point='magenta', name="Грузовик")
        self.my_car_2 = MyCar(self.d.images['cars'][2], self.d.images['icons'][0], [256, 256 + 50],
                              x_m=0, y_m=-5, front_and_back=[1.8, -1.8], vy=self.vy - 0.4,
                              color_point='navy', name="Фура")
        self.objects = [self.front_car, self.my_car, self.my_car_2]
        if many_cars:
            self.many_cars = [MyCar(self.d.images['cars'][2], self.d.images['icons'][0], [256, 256 + 50],
                              x_m=0, y_m=-9 - 5*i, front_and_back=[1.8, -1.8], vy=self.vy - 0.4,
                              color_point='navy', name=f"Фура {i+2}") for i in range(10)]
            self.objects += self.many_cars

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
            for j in range(len(self.objects)):
                if j == 0:
                    self.objects[j].new_process(self.dt, self.t, self.vy)
                else:
                    self.objects[j].process(dt=self.dt,
                                            a=self.objects[j].get_acceleration(self.objects[j-1], dt=self.dt))

            # Отображение
            if self.saving:
                self.show().save(f"res/result_{self.counter:03}.jpg")

    def plot_process(self):
        for j in range(len(self.objects)):
            t = flatten([[i * self.dt] * 2 for i in range(len(self.objects[j].y_list))])
            y = flatten([[self.objects[j].front_list[i], self.objects[j].back_list[i]]
                         for i in range(len(self.objects[j].y_list))])
            plt.plot(t, y, label=self.objects[j].name)
        plt.ylabel(f"Расстояние, м")
        plt.xlabel(f"Время, с")
        plt.legend()
        plt.show()

    def plot_acceleration(self):
        for j in range(len(self.objects) - 1):
            t = [i * self.dt for i in range(len(self.objects[j+1].a_list))]
            plt.plot(t, self.objects[j+1].a_list)
            plt.ylabel(f"Ускорение, м/c²")
            plt.xlabel(f"Время, с")
            plt.title(self.objects[j+1].name)
            plt.show()


if __name__ == '__main__':
    p = Process(frames=1000, dt=0.025, vy=10, saving=True, many_cars=False)
    p.process()
    p.plot_process()
    # p.plot_acceleration()

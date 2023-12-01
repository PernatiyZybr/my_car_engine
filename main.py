import numpy as np
from PIL import Image, ImageDraw
from tiny_functions import *
SCREEN_SIZE = (1920, 1080)
NEWSIE = (350, 350)
RADIUS = 10

class Object:
    def __init__(self, im: Image, center: list, front_and_back: list, color_point: str = 'white',
                 x_m: float = 0, y_m: float = 0, vx: float = 0, vy: float = 0):
        self.im = im
        self.center = center
        self.front_and_back = front_and_back
        self.color_point = color_point

        self.x_m = x_m
        self.y_m = y_m
        self.vx_kmh = vx
        self.vy_kmh = vy
        self.integral = 0.

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

    def add_to_draw(self, im: Image, draw: ImageDraw, x_vec, y_vec, y_past: float, d: int):
        w, h = SCREEN_SIZE
        x, y = self.meter2pixel(d, x_vec, y_vec, y_past)
        x_screen = int(w//2) - self.center[0]
        y_screen = int(h//2) - self.center[1]
        im.paste(self.im, (x_screen - x, y_screen - y), self.im)
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

class MyCar(Object):
    def get_acceleration(self, obj: any, a_max: float = 1e10, dt: float = 0.5):
        x_diff = ((self.y_m + self.front_and_back[0]) - (obj.y_m + obj.front_and_back[1]) + 1.5)
        v_diff = (self.vy_kmh - obj.vy_kmh)  # + 0.5 * abs(obj.vy_kmh)
        self.integral += x_diff * dt
        a = - 0.01 * x_diff - 0.05 * v_diff  # - 0.01 * self.integral
        a -= 0.15 / x_diff**2
        return [0, clip(a, -a_max, a_max)]

class Display:
    def __init__(self, len_road_meter: int):
        # Изображения
        self.images = {'background': Image.open("img/back.png"),
                       'cars': [Image.open("img/car1.png").resize(NEWSIE), Image.open("img/car2.png").resize(NEWSIE),
                                Image.open("img/car_heavy.png"), Image.open("img/car_police.png").resize(NEWSIE)]}

        self.h_pixels = 150
        self.squares_in_screen = 8
        self.len_road_pix = len_road_meter * self.h_pixels
        self.extra_h_y = len_road_meter
        self.directions = {'x': np.array([1/np.sqrt(2), -1/2]),
                           'y': np.array([-1/np.sqrt(2), -1/2]),
                           'z': np.array([0., 1.])}

class Process:
    def __init__(self, dt: float = 1., vy: float = 10, frames: int = 10):
        # Общие параметры
        self.dt = dt
        self.vy = vy
        self.frames = frames
        self.counter = 0
        self.t = 0
        self.y = 0

        # Классы
        self.d = Display(len_road_meter=int(round(dt * vy * frames)))

        # Объекты
        self.front_car = FrontCar(self.d.images['cars'][0], [int(NEWSIE[0]//2), int(NEWSIE[1]//2) + 20], x_m=0, y_m=2,
                                  front_and_back=[-1.1, 1.1], color_point='cyan')
        self.my_car = MyCar(self.d.images['cars'][1], [int(NEWSIE[0]//2), int(NEWSIE[1]//2) + 40], x_m=0, y_m=-1,
                            front_and_back=[-1, 1], color_point='magenta')
        self.my_car_2 = MyCar(self.d.images['cars'][2], [256, 256 + 50], x_m=0, y_m=-5, front_and_back=[-1.8, 1.8],
                              color_point='navy')

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

            # Сохранение
            self.show().save(f"res/result_{self.counter:03}.jpg")


if __name__ == '__main__':
    p = Process(frames=300, dt=0.5, vy=1e0)
    # p.show().show()
    p.process()

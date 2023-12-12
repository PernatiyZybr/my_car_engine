def arrange_both(n):
    return [-i-1 for i in range(n)] + [0] + [i+1 for i in range(n)]

def flatten(lst):
    """Функция берёт 2D массив, делает 1D"""
    return [item for sublist in lst for item in sublist]

def get_noise_distance(distance: float, sigma: float):
    from numpy.random import normal
    return distance + normal(loc=0, scale=sigma)

def clip(a, bot, top):
    """Функция ограничения переменной в заданных пределах"""
    if a < bot:
        return float(bot)
    if a > top:
        return float(top)
    return float(a)

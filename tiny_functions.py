def arrange_both(n):
    return [-i-1 for i in range(n)] + [0] + [i+1 for i in range(n)]

def clip(a, bot, top):
    if a < bot:
        return bot
    if a > top:
        return top
    return a

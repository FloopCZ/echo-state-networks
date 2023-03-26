import math

# List reservoirs in various shapes while keeping the size of approx. 500 neurons.

size=500
i = 1
while i <= math.ceil(math.sqrt(size)):
    w=i
    h=size/i
    size_diff_floor = math.fabs(size - math.floor(h) * w)
    size_diff_ceil = math.fabs(size - math.ceil(h) * w)
    if size_diff_floor < size_diff_ceil:
        size_diff = size_diff_floor
        h = math.floor(h)
    else:
        size_diff = size_diff_ceil
        h = math.ceil(h)

    print(f"{h}x{w} (size {h*w}, size diff {size_diff})")
    i = i + 2



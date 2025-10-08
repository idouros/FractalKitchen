import cv2
import numpy as np

img_rows = 500
img_cols = 500
img_channels = 3

x_start = -2.5
x_end = 1.5
y_start = -2.0
y_end = 2.0

def kernel_mandelbrot(c, max_iter):
    z = complex(0, 0)
    i = 0
    keep_going = True
    while(keep_going):
        z = (z*z) + c
        i += 1
        if i >= max_iter or abs(z) > 2:
            keep_going = False
    if abs(z) > 2:
        return 255
    else:
        return 0

def fractal():
    fractal_canvas = np.full((img_rows, img_cols, img_channels), fill_value = 0, dtype = np.uint8)
    x_step = (x_end - x_start) / float(img_cols)
    y_step = (y_end - y_start) / float(img_rows)
    i = 0
    while i < img_rows:
        y = y_start + float(i) * y_step
        j = 0
        while j < img_cols:
            x = x_start + float(j) * x_step
            val = kernel_mandelbrot(complex(x, y), 10000)
            fractal_canvas[i, j, 0] = val
            fractal_canvas[i, j, 1] = val
            fractal_canvas[i, j, 2] = val
            j += 1
        i += 1
    print(fractal_canvas)
    cv2.imshow('RGB', fractal_canvas)
    cv2.waitKey(0)
    

def main():
    fractal()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
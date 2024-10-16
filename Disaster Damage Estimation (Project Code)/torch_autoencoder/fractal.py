import numpy as np
import matplotlib.pyplot as plt

def generate_fractal(width, height, max_iterations):
    img = np.zeros((height, width))

    # Define the region of interest in the complex plane
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5

    for x in range(width):
        for y in range(height):
            zx, zy = x * (x_max - x_min) / (width - 1) + x_min, y * (y_max - y_min) / (height - 1) + y_min
            c = zx + zy * 1j
            z = c
            for i in range(max_iterations):
                if abs(z) > 2.0:
                    img[y, x] = i
                    break 
                z = z * z + c

    return img

# Generate a fractal image
width, height, max_iterations = 800, 600, 1000
fractal_image = generate_fractal(width, height, max_iterations)

# Display the fractal
plt.imshow(fractal_image, cmap='hot', extent=[-2, 1, -1.5, 1.5])
plt.colorbar()
plt.title("Mandelbrot Fractal")
plt.show()
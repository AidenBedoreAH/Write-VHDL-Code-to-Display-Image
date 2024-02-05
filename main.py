import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import resize

BLACK_THRESHOLD = 55 #Threshold for determining if a square is black
DIFF_COLOR_THRESHOLD = 30 #Threshold for determining if two squares are different color
AVERAGE_COLOR_MATRIX_DIMENSION = 75 #Matrix dimensions for the average color function to output
VHDL_IMG_PIXEL_HEIGHT_AND_WIDTH = 550 #The height/width for the VHDL code to display for
img_path = 'abby_rd_680.jpg'

def similar_rgb_colors(color1, color2):
    """
    Return true if color1 is close to color2
    :param color1: list [r, g, b] r,g,b values should be between 0 and 255
    :param color2: list [r, g, b] r,g,b values should be between 0 and 255
    :return: True if colors are similar, False Otherwise
    """
    color1 = np.array(color1, dtype=np.float64)
    color2 = np.array(color2, dtype=np.float64)
    dist = np.linalg.norm(color1 - color2)
    if dist < DIFF_COLOR_THRESHOLD:
        return True
    else:
        return False

def rgb_is_black(color):
    """
    Return true if color is close to black
    :param color: list [r, g, b] r,g,b values should be between 0 and 255
    :return: True if color is almost black, False Otherwise
    """
    white = [1, 1, 1]
    color1 = np.array(color, dtype=np.float64)
    color2 = np.array(white, dtype=np.float64)
    dist = np.linalg.norm(color1 - color2)
    if dist < BLACK_THRESHOLD:
        return True
    else:
        return False

def rgb_to_hex(rgb):
    """Convert a list of [r, g, b] to a hex value."""
    hex_value = '"{:02x}{:02x}{:02x}"'.format(*rgb)
    return 'x' + hex_value.upper()


def average_color_grid(image, n):
    """Returns an n by n numpy matrix
    each value of the matrix will be the average
    color of image in that region"""
    # Ensure the image is square
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image is not square.")

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the size of each grid square
    grid_size = height // n

    result_array = np.zeros((n, n, 3), dtype=np.uint8)
    # Iterate over each grid
    for i in range(n):
        for j in range(n):
            # Calculate the indices for the current grid
            start_row, end_row = i * grid_size, (i + 1) * grid_size
            start_col, end_col = j * grid_size, (j + 1) * grid_size

            # Extract the current grid from the image
            grid = image[start_row:end_row, start_col:end_col, :]

            # Calculate the average color for the current grid
            average_color = np.mean(grid, axis=(0, 1)).astype(np.uint8)

            # Store the average color in the result array
            result_array[i, j] = average_color

    return result_array


def write_vhdl(max_pixels, color_arr, fileName):
    """

    :param max_pixels: max pixels for the image to be displayed in VHDL
    :param color_arr: a square np array of an image
    :param fileName: name of file to write text to
    :return: None
    """
    w, h, _ = color_arr.shape
    dimensions = []

    for r in range(w):
        draw_start_idx = 0
        for c in range(1, h):
            prev_color = color_arr[r][c-1]
            cur_color = color_arr[r][c]

            if not similar_rgb_colors(cur_color, prev_color): #Change of color
                #If previous is black... update start index
                if rgb_is_black(prev_color):
                    draw_start_idx = c
                #If previous not white
                else:
                    # code for previous color
                    dimension_tuple = (draw_start_idx, c, r, prev_color) #start, end, color
                    dimensions.append(dimension_tuple)
                    #Then update index to current column
                    draw_start_idx = c

            # at the end and last color isnt black
            if c == w-1 and not rgb_is_black(cur_color):
                dimension_tuple = (draw_start_idx, c+1, r, prev_color)  # start, end, color
                dimensions.append(dimension_tuple)

    px_per_box = max_pixels // w
    start_x = 100#(1280 - max_pixels) // 2
    start_y = 100#(720 - max_pixels) // 2
    print('writing text file')
    with open(fileName, "w") as file:
        file.write(f"--This is for {fileName}\n")
        for i in range(len(dimensions)):
            dimension = dimensions[i]
            start_idx, end_idx, row, color = dimension
            hex_color = rgb_to_hex(color)
            lower_x = start_x + start_idx * px_per_box
            upper_x = start_x + end_idx * px_per_box
            lower_y = start_y + row * px_per_box
            upper_y = start_y + (row+1) * px_per_box
            file.write(f'--row:{row} | startIdx: {start_idx} | endIdx: {end_idx}\n')

            # first run through
            if i == 0:
                file.write(f'if hcount > {lower_x} and hcount < {upper_x} and vcount > {lower_y} and vcount < {upper_y} then\n')
                file.write(f'\trgb_data <= {hex_color};\n\n')

            # last run though
            elif i == len(dimensions)-1:
                file.write(f'else\n')
                file.write("\trgb_data <= (others => '0');\n")
                file.write("end if;\n")

            #All middle terms
            else:
                file.write(f'elsif hcount > {lower_x} and hcount < {upper_x} and vcount > {lower_y} and vcount < {upper_y} then\n')
                file.write(f'\trgb_data <= {hex_color};\n\n')


img = cv.imread(img_path)
# dimensions = (680, 680, 3)
# resized_image = resize(img, dimensions, mode='constant', anti_aliasing=True)

colorArr = average_color_grid(img, AVERAGE_COLOR_MATRIX_DIMENSION)
write_vhdl(550, colorArr, img_path.split(".")[0])
plt.imshow(colorArr)
plt.title('Average Color Grid')
plt.show()

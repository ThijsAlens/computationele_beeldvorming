import numpy as np
import matplotlib.pyplot as plt

def read_image(timestamp: int, n_images: int = 1) -> tuple[list[float], list[int], list[int], list[int], list[int]]:
    """
    Read an image from a file.
    
    Args:
        timestamp (int): The beginning timestamp of the image.
        n_images (int): The amount of images to read. Default is 1.
    
    Returns:
        tuple (tuple[list[int], list[int], list[int], list[int], list[int]]): 
        - The ground truth image
        - The timestamps of the events
        - The x-coordinates of the events
        - The y-coordinates of the events
        - The polarities of the events
    """
    with open("data/images.txt", "r") as file:
        lines = file.readlines()
        try:
            line = lines[timestamp]
        except IndexError:
            raise IndexError(f"The timestamp is out of bounds. Asked for {timestamp}, max = {len(lines)}.")
        t0, image0 = line.split()
        t0 = float(t0.strip())
        try:
            t1, image1 = lines[timestamp + n_images].split()
            t1 = float(t1.strip())
        except IndexError:
            raise IndexError(f"The timestamp + n_images is out of bounds. Asked for {timestamp + n_images}, max = {len(lines)}.")
        
    ground_truth: np.ndarray = plt.imread("data/" + image1.rsplit("\n")[0])
    t_list: list[float] = []
    x_list: list[int] = []
    y_list: list[int] = []
    p_list: list[int] = []

    with open("data/events.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            t, x, y, p = line.split()
            t = float(t.strip())
            x = int(x.strip())
            y = int(y.strip())
            p = int(p.strip())
            if t0 <= t < t1:
                t_list.append(t)
                x_list.append(x)
                y_list.append(y)
                p_list.append(p)
            if t >= t1:
                break
    return ground_truth, t_list, x_list, y_list, p_list

def accumulate_events(t: list[float], x: list[int], y: list[int], p: list[int], ground_truth: np.ndarray = None, image_size: tuple[int, int] = (180, 240)) -> np.ndarray:
    """
    Generate an image based on the events.
    
    Args:
        t (list[float]): The timestamps of the events.
        x (list[int]): The x-coordinates of the events.
        y (list[int]): The y-coordinates of the events.
        p (list[int]): The polarities of the events.
        ground_truth (np.ndarray): The ground truth image.
        image_size (tuple[int, int]): The size of the image.

    Returns:
        np.ndarray: The generated image.
    """

    # Create the image
    image = np.zeros(image_size)

    # Accumulate the events
    print(len(t), len(x), len(y), len(p))
    for i in range(len(t)):
        if p[i] == 1:
            image[y[i]-1, x[i]-1] += 1
        else:
            image[y[i]-1, x[i]-1] -= 1

    # Plot the image
    plt.figure(figsize=(6, 4))
    plt.imshow(image, cmap='gray', vmin=-4, vmax=4)
    plt.show()

    return image

if __name__ == "__main__":
    # get an image from the dataset
    image, t, x, y, p = read_image(0, 1)
    # get the events from the dataset between the 0th and 1st image
    accumulate_events(t, x, y, p)
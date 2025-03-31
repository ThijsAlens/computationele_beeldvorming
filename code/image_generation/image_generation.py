import numpy as np
import matplotlib.pyplot as plt

def read_image(timestamp: int, n_images: int = 1) -> tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Read an image from a file.
    
    Args:
        timestamp (int): The beginning timestamp of the image.
        n_images (int): The amount of images to read. Default is 1.
    
    Returns:
        tuple (tuple[np.array, np.array, np.array, np.array, np.array]): 
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
            print("The timestamp is out of bounds.")
            return None
        t0, image0 = line.split(" ")
        t1, image1 = lines[timestamp + n_images].split(" ")

    ground_truth = plt.imread("data/" + image1.rsplit("\n")[0])
    t_list: float = []
    x_list: int = []
    y_list: int = []
    p_list: int = []

    with open("data/events.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            t, x, y, p = line.split(" ")
            if t0 <= t < t1:
                t_list.append(float(t))
                x_list.append(int(x))
                y_list.append(int(y))
                p_list.append(int(p))
            if t >= t1:
                break
    return ground_truth, np.array(t_list), np.array(x_list), np.array(y_list), np.array(p_list)

def accumulate_events(t: np.array, x: np.array, y: np.array, p: np.array, ground_truth: np.array = None, image_size: tuple[int, int] = (240, 180)) -> np.array:
    """
    Generate an image based on the events.
    
    Args:
        t (np.array): The timestamps of the events.
        x (np.array): The x-coordinates of the events.
        y (np.array): The y-coordinates of the events.
        p (np.array): The polarities of the events.
        ground_truth (np.array): The ground truth image.
        image_size (tuple[int, int]): The size of the image.

    Returns:
        np.array: The generated image.
    """

    # Create the image
    image = np.zeros(image_size)

    # Accumulate the events
    for i in range(len(t)):
        image[x[i], y[i]] += p[i]

    # Plot the image
    plt.imshow(image, cmap='gray')
    plt.show()

    return image

if __name__ == "__main__":
    # get an image from the dataset
    image, t, x, y, p = read_image(0, 100)
    # get the events from the dataset between the 0th and 1st image
    accumulate_events(t, x, y, p)
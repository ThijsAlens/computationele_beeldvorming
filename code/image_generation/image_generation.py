import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def show_image(reconstructed_image: np.ndarray, ground_truth_image: np.ndarray) -> None:
    """
    Show the reconstructed image and the ground truth image side by side.

    Args:
        reconstructed_image (np.ndarray): The reconstructed image.
        ground_truth_image (np.ndarray): The ground truth image.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))  # Make the images larger

    # Show reconstructed image
    plt.subplot(1, 2, 1)
    plt.imshow(reconstructed_image, cmap="gray", vmin=-4, vmax=4)
    plt.title("Reconstructed Image")
    plt.axis("off")

    # Show ground truth image
    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth_image, cmap="gray", vmin=-4, vmax=4)
    plt.title("Ground Truth Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    return


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

def time_surface(t: list[float], x: list[int], y: list[int], p: list[int], decay_function: callable = lambda dt: np.exp(-dt/0.1), image_size: tuple[int, int] = (180, 240)) -> np.ndarray:
    """
    Generate an image based on the events using a time surface approach with a decay function.
    
    Args:
        t (list[float]): The timestamps of the events.
        x (list[int]): The x-coordinates of the events.
        y (list[int]): The y-coordinates of the events.
        p (list[int]): The polarities of the events.
        decay_function (callable): The decay function to use. Default is an exponential decay.
        image_size (tuple[int, int]): The size of the image.

    Returns:
        dp.ndarray: The generated image.
    """
    # Initialize the time surface
    time_surface = np.zeros(image_size, dtype=np.float32)
    current_time = max(t) if t else 0  # Use latest timestamp as a start

    # Create an array to store the last event timestamps
    last_event_time = np.full(image_size, -np.inf)  # Initialize with very low values

    # Update the last event timestamp for each pixel
    for i in range(len(t)):
        last_event_time[y[i], x[i]] = t[i]

    # Compute time difference and apply decay
    delta_t = current_time - last_event_time
    delta_t[delta_t < 0] = np.inf  # Ignore pixels with no events
    time_surface = decay_function(delta_t)

    return time_surface

def accumulate_events(t: list[float], x: list[int], y: list[int], p: list[int], consider_polarity: bool = True, image_size: tuple[int, int] = (180, 240)) -> np.ndarray:
    """
    Generate an image based on the events using a simple accumulation.
    
    Args:
        t (list[float]): The timestamps of the events.
        x (list[int]): The x-coordinates of the events.
        y (list[int]): The y-coordinates of the events.
        p (list[int]): The polarities of the events.
        consider_polarity (bool): Whether to consider the polarity of the events. Default is True.
        - If False, all events are treated as positive.
        - If True, the polarity is considered.
        image_size (tuple[int, int]): The size of the image.

    Returns:
        np.ndarray: The generated image.
    """

    # Create an empty image
    image = np.zeros(image_size)

    # Accumulate the events
    for i in range(len(t)):
        if consider_polarity:
            if p[i] == 1:
                image[y[i]-1, x[i]-1] += 1
            else:
                image[y[i]-1, x[i]-1] -= 1
        else:
            image[y[i]-1, x[i]-1] += 1

    return image

if __name__ == "__main__":
    # get an image from the dataset
    image, t, x, y, p = read_image(0, 1)
    # get the events from the dataset between the 0th and 1st image
    generated_image = accumulate_events(t, x, y, p, consider_polarity=False)
    # generated_image = time_surface(t, x, y, p)
    show_image(generated_image, image)
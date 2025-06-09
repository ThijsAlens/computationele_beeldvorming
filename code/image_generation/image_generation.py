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

def save_image(image: np.ndarray, filename: str) -> None:
    """
    Save the image to a file with proper grayscale normalization.
    
    Args:
        image (np.ndarray): The image to save.
        filename (str): The name of the file to save the image to.
    
    Returns:
        None
    """
    vmin, vmax = 0, 1
    normalized_image = np.clip((image - vmin) / (vmax - vmin), 0, 1)

    plt.imsave(filename, normalized_image, cmap="gray")


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
    time_surface = np.zeros(image_size, dtype=np.int8)
    current_time = max(t) if t else 0

    last_event_time = np.full(image_size, -np.inf)

    # Update the last event timestamp for each pixel
    for i in range(len(t)):
        last_event_time[y[i], x[i]] = t[i]

    # Compute time difference and apply decay
    delta_t = current_time - last_event_time
    delta_t[delta_t < 0] = np.inf  # Ignore pixels with no events
    time_surface = decay_function(delta_t)

    return time_surface

def generate_image(orig_image: list[float], t: list[float], x: list[int], y: list[int], p: list[int], consider_polarity: bool = True, image_size: tuple[int, int] = (180, 240)) -> np.ndarray:
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

    # Accumulate the events
    for i in range(len(t)):
        if consider_polarity:
            if p[i] == 1:
                orig_image[y[i]-1, x[i]-1] += 0.1
            else:
                orig_image[y[i]-1, x[i]-1] -= 0.1
        else:
            orig_image[y[i]-1, x[i]-1] += 1

    return orig_image

def generate_video(start: int, end: int, step: int = 1) -> None:
    """
    Generate a video from the images.
    
    Args:
        start (int): The starting timestamp.
        end (int): The ending timestamp.
        step (int): The step size. Default is 1.
    
    Returns:
        None
    """
    for i in range(start, end, step):
        image, t, x, y, p = read_image(i, 1)
        final_image = generate_image(image, t, x, y, p)
        save_image(final_image, f"code/video/frame_{i}.png")
    
    import os
    import cv2
    image_dir = "code/video"
    output_file = "code/video/output.mp4"
    fps = 30
    # Get all PNG files in sorted order
    files = sorted(f for f in os.listdir(image_dir) if f.startswith("frame_") and f.endswith(".png"))
    if not files:
        print("No PNG files found.")
        return

    # Read first image to get frame size
    first_frame = cv2.imread(os.path.join(image_dir, files[0]))
    height, width, _ = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' or 'avc1' also work
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for file in files:
        frame = cv2.imread(os.path.join(image_dir, file))
        video.write(frame)

    video.release()

    return

if __name__ == "__main__":
    # get an image from the dataset
    image, t, x, y, p = read_image(20, 1)
    # get the events from the dataset between the 0th and 1st image
    generated_image = generate_image(image, t, x, y, p, consider_polarity=True)
    # generated_image = time_surface(t, x, y, p)
    save_image(generated_image, "code/generated_image.png")
    save_image(image, "code/ground_truth.png")
    show_image(generated_image, image)
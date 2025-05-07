from image_generation.image_generation import *
from inference.inference import *

if __name__ == "__main__":
    # get an image from the dataset
    image, t, x, y, p = read_image(200, 1)
    # get the events from the dataset between the 0th and 1st image
    generated_image = accumulate_events(t, x, y, p, consider_polarity=False)
    # generated_image = time_surface(t, x, y, p)
    save_image(generated_image, "code/generated_image.png")
    show_image(generated_image, image)
    # infer the generated image
    run_inference("code/generated_image.png", "code/inference_output.png")
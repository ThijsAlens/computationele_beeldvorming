from ultralytics import YOLO

def run_inference(input_path: str, output_path: str, model_path: str = "yolov8n.pt") -> None:
    """
    Run inference on an image using YOLOv8.

    Args:
        path (str): Path to the image file.
        model_path (str): Path to the YOLOv8 model file. Default is "yolov8n.pt".

    Returns:
        None: Displays the image with detected objects.
    """
    model = YOLO("yolov8n.pt")
    # Run inference on an image
    results = model(input_path)

    # Show and save the results
    results[0].show()
    results[0].save(filename=output_path) 

    # Access detected objects
    for result in results:
        boxes = result.boxes
        for box in boxes:
            print("Class:", model.names[int(box.cls)])
            print("Confidence:", float(box.conf))
            print("Box coordinates:", box.xyxy.tolist())
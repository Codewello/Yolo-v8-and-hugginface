import gradio as gr
import torch
from ultralyticsplus import YOLO, render_result


torch.hub.download_url_to_file(
    'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftexashafts.com%2Fwp-content%2Fuploads%2F2016%2F04%2Fconstruction-worker.jpg', 'one.jpg')
torch.hub.download_url_to_file(
    'https://www.pearsonkoutcherlaw.com/wp-content/uploads/2020/06/Construction-Workers.jpg', 'two.jpg')
torch.hub.download_url_to_file(
    'https://nssgroup.com/wp-content/uploads/2019/02/Building-maintenance-blog.jpg', 'three.jpg')


def yoloV8_func(image: gr.inputs.Image = None,
                image_size: gr.inputs.Slider = 640,
                conf_threshold: gr.inputs.Slider = 0.4,
                iou_threshold: gr.inputs.Slider = 0.50):
    """This function performs YOLOv8 object detection on the given image.

    Args:
        image (gr.inputs.Image, optional): Input image to detect objects on. Defaults to None.
        image_size (gr.inputs.Slider, optional): Desired image size for the model. Defaults to 640.
        conf_threshold (gr.inputs.Slider, optional): Confidence threshold for object detection. Defaults to 0.4.
        iou_threshold (gr.inputs.Slider, optional): Intersection over Union threshold for object detection. Defaults to 0.50.
    """
    # Load the YOLOv8 model from the 'best.pt' checkpoint
    model_path = "best.pt"
    model = YOLO(model_path)

    # Perform object detection on the input image using the YOLOv8 model
    results = model.predict(image,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            imgsz=image_size)

    # Print the detected objects' information (class, coordinates, and probability)
    box = results[0].boxes
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)

    # Render the output image with bounding boxes around detected objects
    render = render_result(model=model, image=image, result=results[0])
    return render


inputs = [
    gr.inputs.Image(type="filepath", label="Input Image"),
    gr.inputs.Slider(minimum=320, maximum=1280, default=640,
                     step=32, label="Image Size"),
    gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.25,
                     step=0.05, label="Confidence Threshold"),
    gr.inputs.Slider(minimum=0.0, maximum=1.0, default=0.45,
                     step=0.05, label="IOU Threshold"),
]


outputs = gr.outputs.Image(type="filepath", label="Output Image")

title = "YOLOv8 101: Custom Object Detection on Construction Workers"


examples = [['one.jpg', 640, 0.5, 0.7],
            ['two.jpg', 800, 0.5, 0.6],
            ['three.jpg', 900, 0.5, 0.8]]

yolo_app = gr.Interface(
    fn=yoloV8_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=True,
)

# Launch the Gradio interface in debug mode with queue enabled
yolo_app.launch(debug=True, enable_queue=True)

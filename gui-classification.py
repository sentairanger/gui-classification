from guizero import App, PushButton, Drawing
from picamera2 import Picamera2
from time import sleep
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
import cv2

matplotlib.use('GTK3Agg')
timestamp = datetime.now().isoformat()
ie = Core()
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)

def capture_image():
    picam2.start()
    sleep(3)
    picam2.capture_file("intel_%s.jpg" % timestamp)
    viewer.image(20, 10, "intel_%s.jpg" % timestamp)
    viewer.text(20, 20, "intel_%s.jpg" % timestamp, color="aqua")

def classify():
    model = ie.read_model(model="model/v3-small_224_1.0_float.xml")
    compiled_model = ie.compile_model(model=model, device_name="MYRIAD")
    output_layer = compiled_model.output(0)
    image = cv2.cvtColor(cv2.imread('intel_%s.jpg' % timestamp), code=cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(src=image, dsize=(224,224))
    input_image = np.expand_dims(input_image, 0)
    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)
    imagenet_classes = open("model/imagenet_2012.txt").read().splitlines()
    imagenet_classes = ['background'] + imagenet_classes
    text = imagenet_classes[result_index]
    viewer.text(20, 320, text, color="aqua")


app = App(title="Sample Update")
button = PushButton(app, text="Take picture", command=capture_image)
button2 = PushButton(app, text="Show Result", command=classify)
viewer = Drawing(app, width='fill', height='fill')

app.display()
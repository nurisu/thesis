from flask import Flask,render_template,Response
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from flask_sse import sse
import time
import datetime
import os
import json

from collections import deque

from centroidtracker import *

app=Flask(__name__)

current_dir = os.path.dirname(os.path.realpath(__file__))

data = json.load(open(fr"{current_dir}\mapsettings.json"))

net = cv2.dnn.readNet(fr'{current_dir}\model\rjom.weights', fr'{current_dir}\model\rjom.cfg')

classes = []
with open(fr"{current_dir}\model\rjom.names", "r") as f:
    classes = f.read().splitlines()


cap = cv2.VideoCapture(fr'{current_dir}\videos\traffic.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))


labels1 =['Bus', 'Car', 'Motorbike', 'Truck']
data1 = [0, 0, 0, 0]

start = datetime.datetime.now()

for item in data:
    cap.open(fr'{current_dir}\videos\{item["filename"]}.mp4')
    _, img = cap.read()
    cv2.imwrite(fr'{current_dir}\preview\{item["filename"]}.jpg', img)


lnames = net.getLayerNames()
out_lays = [lnames[i - 1] for i in net.getUnconnectedOutLayers()]

W, H = (None, None)

source = fr"{current_dir}\cam5.mp4"
cap = cv2.VideoCapture(source)

points = deque(maxlen=124)

ct = CentroidTracker(maxDisappeared=8)


def generate_frames():
    while True:
        ret, img = cap.read()
        ret, img = cap.read()
        ret, img = cap.read()
        if ret:

            H, W = (img.shape[0], img.shape[1])
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(out_lays)

            boxes = []
            confidences = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype('int')

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        box_centers = [centerX, centerY]

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_ids)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)
            fin_boxes = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    print(class_ids[i][0])
                    label = str(class_ids[i])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 0), 2)
                    #cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 255), 1)
                    fin_boxes.append([x, y, x + w, y + h])
            objects = ct.update(fin_boxes)
            for (objectID, centroid) in objects.items():
                text = str(objectID + 1)
                cv2.putText(img, text, (centroid[0], centroid[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret,buffer=cv2.imencode('.jpg',img)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_graph():
    while True:
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])

        ax.bar(labels1,data1)
        plt.savefig(fr'{current_dir}\graph.jpg', bbox_inches='tight')
        plt.close()
        img = cv2.imread(fr'{current_dir}\graph.jpg')

        img_encode = cv2.imencode('.jpg', img)[1]
          

        data_encode = np.array(img_encode)
          

        byte_encode = data_encode.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + byte_encode + b'\r\n')
        time.sleep(1)


@app.route('/select')
def select():
    config = {"title": "Выбор камеры", "data": data}
    return render_template('base.html', config=config)


@app.route('/map')
def mappage():
    config = {"title": "Карта", "data": data}
    return render_template('base.html', config=config)


@app.route('/camera/<name>')
def hello_name(name):
    lnames = net.getLayerNames()
    out_lays = [lnames[i - 1] for i in net.getUnconnectedOutLayers()]

    (W, H) = (None, None)

    points = deque(maxlen=124)

    ct = CentroidTracker(maxDisappeared=8)
    title = ""
    for item in data:
        if item["filename"] == name:
            title = item["title"]
    cap.open(fr'{current_dir}\videos\{name}.mp4')
    config = {"title": "Камера", "data": {"title": title}}
    return render_template('base.html', config=config)


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/graph')
def graph():
    return Response(generate_graph(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/perminute')
def perminute():
    delta = (datetime.datetime.now() - start).total_seconds()
    d = round(sum(data1) / delta * 60)
    return str(d)


@app.route('/perhour')
def perhour():
    delta = (datetime.datetime.now() - start).total_seconds()
    d = round(sum(data1 ) / delta * 60 * 60)
    return str(d)

 
if __name__=="__main__":
    app.run(debug=True, use_reloader=False)
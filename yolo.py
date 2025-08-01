import numpy as np
import argparse
import cv2 as cv
import subprocess
import os
from yolo_utils import infer_image, show_image

FLAGS = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model-path", type=str, default="./yolov3-coco/",
                        help="Directory of YOLOv3 model files.")
    parser.add_argument("-w", "--weights", type=str, default="./yolov3-coco/yolov3.weights",
                        help="Path to YOLOv3 weights file.")
    parser.add_argument("-cfg", "--config", type=str, default="./yolov3-coco/yolov3.cfg",
                        help="Path to YOLOv3 config file.")
    parser.add_argument("-i", "--image-path", type=str, help="Path to input image file.")
    parser.add_argument("-v", "--video-path", type=str, help="Path to input video file.")
    parser.add_argument("-vo", "--video-output-path", type=str, default="./output.avi",
                        help="Path to save output video.")
    parser.add_argument("-l", "--labels", type=str, default="./yolov3-coco/coco-labels",
                        help="Path to labels file (newline separated).")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="Minimum confidence threshold. Default: 0.5")
    parser.add_argument("-th", "--threshold", type=float, default=0.3,
                        help="NMS threshold. Default: 0.3")
    parser.add_argument("--download-model", type=bool, default=False,
                        help="Set True to download YOLO model.")
    parser.add_argument("-t", "--show-time", type=bool, default=False,
                        help="Print inference time.")

    FLAGS, _ = parser.parse_known_args()

    # Download YOLO model files if requested
    if FLAGS.download_model:
        subprocess.call(["./yolov3-coco/get_model.sh"])

    # Load class labels
    with open(FLAGS.labels, "r") as f:
        labels = f.read().strip().split("\n")

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # Load model
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
    layer_names = net.getLayerNames()

    # Compatible output layer extraction
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    if FLAGS.image_path is None and FLAGS.video_path is None:
        print("Neither image nor video path provided.")
        print("Starting webcam inference...")

    # 1. Image Inference
    if FLAGS.image_path:
        img = cv.imread(FLAGS.image_path)
        if img is None:
            raise ValueError("Image cannot be loaded. Please check the path.")

        height, width = img.shape[:2]
        img, _, _, _, _ = infer_image(net, output_layers, height, width, img, colors, labels, FLAGS)
        show_image(img)

    # 2. Video File Inference
    elif FLAGS.video_path:
        vid = cv.VideoCapture(FLAGS.video_path)
        if not vid.isOpened():
            raise ValueError("Video file could not be opened. Check path or file format.")

        height, width = None, None
        writer = None

        while True:
            grabbed, frame = vid.read()
            if not grabbed:
                break

            if width is None or height is None:
                height, width = frame.shape[:2]

            frame, _, _, _, _ = infer_image(net, output_layers, height, width, frame, colors, labels, FLAGS)

            if writer is None:
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, (width, height), True)

            writer.write(frame)

        print("[INFO] Cleaning up...")
        writer.release()
        vid.release()

    # 3. Webcam Inference
    else:
        vid = cv.VideoCapture(0)
        if not vid.isOpened():
            print("[ERROR] Could not open webcam.")
            exit()

        count = 0
        boxes = confidences = classids = idxs = None

        while True:
            ret, frame = vid.read()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame from webcam.")
                break

            height, width = frame.shape[:2]

            if count == 0:
                frame, boxes, confidences, classids, idxs = infer_image(
                    net, output_layers, height, width, frame, colors, labels, FLAGS
                )
            else:
                frame, boxes, confidences, classids, idxs = infer_image(
                    net, output_layers, height, width, frame, colors, labels, FLAGS,
                    boxes, confidences, classids, idxs, infer=False
                )
            count = (count + 1) % 6

            cv.imshow("webcam", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        vid.release()
        cv.destroyAllWindows()

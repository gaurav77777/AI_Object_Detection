import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # Import this to display images in Colab

# Load YOLO pre-trained model and config files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Check if classes are loaded correctly
print("Classes loaded from coco.names:", classes[:10])  # Print the first 10 class names

# Load the image
image_path = "Lion.jpg"  # Path to your image file
image = cv2.imread(image_path)
height, width, channels = image.shape

# Convert image to blob
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Process detections
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w // 2
            y = center_y - h // 2
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Debugging: Print class IDs of detected objects
print("Class IDs of detected objects:", class_ids)

# Apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw the detected bounding boxes on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        label = classes[class_id]  # Get the label from coco.names
        print(f"Detected object: {label} (Class ID: {class_id})")  # Debugging: print detected label and class ID

        color = (0, 255, 0)  # Green color for bounding boxes
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw the label above the bounding box
        label_text = f"{label}: {confidences[i]:.2f}"  # Including confidence score
        cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the image with labels using cv2_imshow for Colab
cv2_imshow(image)


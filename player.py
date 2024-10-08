import cv2
import torch
import numpy as np
import argparse
from ultralytics import YOLO

# Load the YOLOv5 model
def load_yolo_model(model_path):
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA version: {torch.version.cuda}")

    # Load the entire model
    model = YOLO(model_path)
    
    # Move model to the appropriate device
    model.to(device)
    
    return model

# Process the video
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame)
        
        # Process results
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label
                label = f'Class: {cls}, Conf: {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main(model_path, video_path):
    #  prompt the user to playing the which video and model
    print(f"Playing video: {video_path} with model: {model_path}")
    # Load YOLOv5 model
    model = load_yolo_model(model_path)

    # Process the video
    process_video(video_path, model)


# main
if __name__ == "__main__":
    # when start the python player.py, it will need two parameters, model_path and video_path
    # if not, it will use the default model and video
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best.pt')
    parser.add_argument('--video_path', type=str, default='video/v_d3.mp4')
    args = parser.parse_args()
    main(args.model_path, args.video_path)


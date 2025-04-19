"""
Penalty Kick Prediction System v2.0
Combining YOLOv7 Pose Estimation with LSTM Temporal Analysis
"""
import cv2
import torch
import numpy as np
from collections import deque
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# Configuration
CONFIG = {
    "yolo_weights": "yolov7-w6-pose.pt",
    "lstm_weights": "lstm_model.pth",
    "sequence_length": 15,  # 0.5s at 30fps
    "input_size": 54,       # 18 keypoints * 3 (x,y,conf)
    "hidden_size": 128,
    "num_classes": 4,      # Left, Middle, Right, Out
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}



class PoseLSTMPredictor:
    def __init__(self):
        # Initialize YOLOv7 Pose 
        self.yolo_model = attempt_load(CONFIG["yolo_weights"], map_location=CONFIG["device"])
        self.yolo_model.eval()
        
        # Initialize LSTM 
        self.lstm = torch.nn.LSTM(
            input_size=CONFIG["input_size"],
            hidden_size=CONFIG["hidden_size"],
            batch_first=True
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(CONFIG["hidden_size"], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, CONFIG["num_classes"])
        )
        self.load_lstm_weights()
        
        # Temporal buffer 
        self.keypoint_buffer = deque(maxlen=CONFIG["sequence_length"])

    def load_lstm_weights(self):
        """Load pre-trained LSTM weights"""
        state_dict = torch.load(CONFIG["lstm_weights"], map_location=CONFIG["device"])
        self.lstm.load_state_dict(state_dict["lstm"])
        self.classifier.load_state_dict(state_dict["classifier"])

    def preprocess_frame(self, frame):
        """YOLOv7 Pose preprocessing """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1).float().to(CONFIG["device"])
        img = img / 255.0  # 0 - 1.0
        return img.unsqueeze(0)

    def extract_keypoints(self, frame):
        """YOLOv7 Pose estimation pipeline """
        img = self.preprocess_frame(frame)
        with torch.no_grad():
            output, _ = self.yolo_model(img)
        
        # Process output 
        output = non_max_suppression_kpt(
            output, 
            0.25,   # Confidence threshold
            0.65,    # IoU threshold
            nc=self.yolo_model.yaml["nc"],
            nkpt=self.yolo_model.yaml["nkpt"],
            kpt_label=True
        )
        
        # Extract kicker's keypoints 
        keypoints = output_to_keypoint(output)[0][:, 7:].cpu().numpy()
        if len(keypoints) > 0:
            return keypoints[0]  # Assume first detection is kicker
        return None

    def predict_kick_direction(self):
        """LSTM-based prediction """
        if len(self.keypoint_buffer) < CONFIG["sequence_length"]:
            return None
            
        # Convert sequence to tensor
        sequence = torch.FloatTensor(np.array(self.keypoint_buffer)).to(CONFIG["device"])
        sequence = sequence.unsqueeze(0)  # Add batch dim
        
        # LSTM forward pass
        with torch.no_grad():
            lstm_out, _ = self.lstm(sequence)
            logits = self.classifier(lstm_out[:, -1, :])
        
        return torch.argmax(logits).item()

    def process_video(self, video_path):
        """Main processing pipeline """
        cap = cv2.VideoCapture(video_path)
        predictions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
        
            # Extract keypoints 
            keypoints = self.extract_keypoints(frame)
            if keypoints is not None:
                self.keypoint_buffer.append(keypoints)
                
                # Make prediction when buffer full
                if len(self.keypoint_buffer) == CONFIG["sequence_length"]:
                    pred = self.predict_kick_direction()
                    predictions.append(pred)
                    
        cap.release()
        return self._postprocess(predictions)

    def _postprocess(self, predictions):
        """Majority voting for final prediction """
        counts = np.bincount(predictions)
        return np.argmax(counts)

# Usage Example
if __name__ == "__main__":
    predictor = PoseLSTMPredictor()
    result = predictor.process_video("penalty_kick.mp4")
    directions = ["Left", "Middle", "Right", "Out"]
    print(f"Predicted Direction: {directions[result]}")
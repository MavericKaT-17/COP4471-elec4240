"""
Penalty Kick Prediction System v2.1
Combining YOLOv7-Pose with Hybrid CNN-LSTM Architecture
"""
import cv2
import torch
import numpy as np
import logging
from collections import deque
from torchvision.models import resnet50
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.FileHandler('penalty_analysis.log'), logging.StreamHandler()]
)

class EnhancedPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_models()
        self._init_buffers()
        
    def _init_models(self):
        """Initialize hybrid model components"""
        # YOLOv7-Pose 
        self.yolo = attempt_load('yolov7-w6-pose.pt', map_location=self.device)
        self.yolo.eval()
        
        # ResNet feature extractor 
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = torch.nn.Identity()  # Remove final layer
        self.resnet.eval()
        
        # Enhanced LSTM with dropout 
        self.lstm = torch.nn.LSTM(
            input_size=2560,  # 18*3 keypoints + 2048 ResNet features
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        
        # Classifier with regularization 
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4)
        )

    def _init_buffers(self):
        """Initialize temporal buffers"""
        self.keypoint_buffer = deque(maxlen=15)  # 0.5s @30fps
        self.feature_buffer = deque(maxlen=15)
        self.prediction_buffer = deque(maxlen=5)  # For majority voting

    def _extract_hybrid_features(self, frame):
        """Combine YOLO keypoints with ResNet visual features """
        try:
            # YOLO keypoints 
            img_tensor = torch.from_numpy(frame).permute(2,0,1).float().to(self.device)/255.0
            with torch.no_grad():
                output = self.yolo(img_tensor.unsqueeze(0))[0]
            
            # Improved kicker detection 
            detections = non_max_suppression_kpt(output, 0.7, 0.65, nc=self.yolo.yaml['nc'], nkpt=18)
            if len(detections) == 0:
                return None, None
                
            # Select most central player 
            img_center = np.array([frame.shape[1]/2, frame.shape[0]/2])
            distances = [np.linalg.norm(d[:2] - img_center) for d in detections]
            kicker = detections[np.argmin(distances)]
            
            # ResNet features 
            resnet_input = transforms.Resize(224)(torch.tensor(frame).permute(2,0,1).float()/255.0)
            with torch.no_grad():
                cnn_features = self.resnet(resnet_input.unsqueeze(0).to(self.device))
            
            return kicker[7:].cpu().numpy(), cnn_features.squeeze().cpu().numpy()
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {str(e)}")
            return None, None

    def _dynamic_sequence_processing(self):
        """Sliding window prediction with majority voting """
        if len(self.feature_buffer) < 15:
            return None
            
        # Create overlapping sequences
        seq_window = np.array(self.feature_buffer)[-12:]
        hybrid_features = torch.FloatTensor(seq_window).unsqueeze(0).to(self.device)
        
        # LSTM prediction 
        with torch.no_grad():
            lstm_out, _ = self.lstm(hybrid_features)
            logits = self.classifier(lstm_out[:, -1, :])
            pred = torch.argmax(logits).item()
            
        self.prediction_buffer.append(pred)
        return max(set(self.prediction_buffer), key=self.prediction_buffer.count)

    def process_video(self, video_path):
        """Enhanced processing pipeline with visualization"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return None
            
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret: break
                
                # Feature extraction
                keypoints, features = self._extract_hybrid_features(frame)
                if keypoints is not None:
                    self.keypoint_buffer.append(keypoints)
                    self.feature_buffer.append(features)
                    
                    # Dynamic prediction 
                    pred = self._dynamic_sequence_processing()
                    if pred is not None:
                        self._visualize_prediction(frame, pred)
                        
            except Exception as e:
                logging.error(f"Frame processing error: {str(e)}")
                continue
                
        cap.release()
        return self._final_prediction()

    def _visualize_prediction(self, frame, pred):
        """Visualization overlay """
        directions = ["Left", "Middle", "Right", "Out"]
        cv2.putText(frame, f"Pred: {directions[pred]}", (20,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Analysis", frame)
        if cv2.waitKey(1) == 27:  # ESC退出
            exit()

    def _final_prediction(self):
        return max(set(self.prediction_buffer), key=self.prediction_buffer.count)
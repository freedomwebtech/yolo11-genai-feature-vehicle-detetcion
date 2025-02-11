import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class VehicleDetectionProcessor:
    def __init__(self, video_file, api_key, yolo_model_path="yolo11s.pt"):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.yolo_model = YOLO(yolo_model_path)
        self.names = self.yolo_model.names
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file.")

        # Define Region of Interest (ROI) for vehicle detection
        self.area = np.array([(128, 565), (35, 733), (538, 744), (527, 579)], np.int32)

        # Set to track processed track_ids
        self.processed_track_ids = set()
        self.current_date = time.strftime("%Y-%m-%d")
        self.output_filename = f"vehicle_data_{self.current_date}.txt"

        # Initialize output file if empty
        if not os.path.exists(self.output_filename) or os.path.getsize(self.output_filename) == 0:
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Numberplate | Vehicle Type | Vehicle Color | Vehicle Company\n")
                file.write("-" * 80 + "\n")

    def encode_image_to_base64(self, image):
        _, img_buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(img_buffer).decode('utf-8')

    def analyze_image_with_gemini(self, current_image):
        if current_image is None:
            return "No image available for analysis."

        current_image_data = self.encode_image_to_base64(current_image)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """
                    Analyze this image and extract only the following details:

                    | Numberplate | Vehicle Type | Vehicle Color | Vehicle Company |
                    |--------------|--------------|---------------|-----------------|
                    |              |              |               |                 |
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{current_image_data}"},
                    "description": "Detected vehicle"
                }
            ])

        try:
            response = self.gemini_model.invoke([message])
            return response.content
        except Exception as e:
            print(f"Error invoking Gemini model: {e}")
            return "Error processing image."

    def process_crop_image(self, current_image):
        response_content = self.analyze_image_with_gemini(current_image)
        extracted_data = response_content.strip().split("\n")[2:]

        if extracted_data:
            current_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.output_filename, "a", encoding="utf-8") as file:
                for row in extracted_data:
                    if "--------------" in row or not row.strip():
                        continue
                    values = [col.strip() for col in row.split("|")[1:-1]]
                    if len(values) == 4:
                        plate, vehicle_type, vehicle_color, vehicle_company = values
                        file.write(f"{current_datetime} | {plate} | {vehicle_type} | {vehicle_color} | {vehicle_company}\n")
            print("✅ Data saved to file.")

    def crop_and_process(self, frame, box, track_id):
        if track_id in self.processed_track_ids:
            print(f"Track ID {track_id} already processed. Skipping.")
            return  

        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        self.processed_track_ids.add(track_id)

        threading.Thread(target=self.process_crop_image, args=(crop,), daemon=True).start()

    def process_video_frame(self, frame):
        frame = cv2.resize(frame, (1090, 800))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                c = self.names[class_id]
                x1, y1, x2, y2 = box
                cy = (y1 + y2) // 2

                result = cv2.pointPolygonTest(self.area, (x2, y2), False)
                if result >= 0:  # Vehicle is inside the defined ROI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                    self.crop_and_process(frame, box, track_id)
 #               else:
 #                   cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def RGB(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Coordinates: [{x}, {y}]")

    def start_processing(self):
        window_name = "Vehicle Detection"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.RGB)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_video_frame(frame)
            cv2.polylines(frame, [self.area], True, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Ensures real-time processing
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Data saved to {self.output_filename}")


# Example usage:
if __name__ == "__main__":
    video_file = "ind1.mp4"
    api_key = "AIzaSyDVe39cm6E9jrUkUGxNPq_p542nv-gbqwE"  # Replace with your actual API key
    processor = VehicleDetectionProcessor(video_file, api_key)
    processor.start_processing()

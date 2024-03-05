import cv2
from ultralytics import YOLO
import time
from multiprocessing import Process, Queue
import numpy as np
from collections import defaultdict
import os
import json
from imutils.video import VideoStream
import datetime
import threading
import pathlib
import logging
import torch
from mongoFunc import send_requests_interval, increaseIN, increaseOUT

new_width, new_height = 640, 480  # dọc, ngang
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)

output_folder_original = "output_videos_original"
os.makedirs(output_folder_original, exist_ok=True)

def load_config():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    return config


config = load_config()
lines1 = config["camera_config2"]["192.168.0.64"]["lines"]
lines2=  config["camera_config2"]["192.168.0.65"]["lines"]

camera_ip_top = "192.168.0.64"
camera_ip_bottom = "192.168.0.65"

# Cài đặt lưu video
config_video_output = {
    "fps": 20
}
middle_line_1=200
middle_line_2=200

# #setting log
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filemode="a", format=log_format, level=logging.DEBUG)
# Create a FileHandler with a dynamic file name based on the current date
log_file_name = datetime.datetime.now().strftime("log_%Y-%m-%d.log")
file_handler = logging.FileHandler(log_file_name)

# Set the formatter for the FileHandler
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger = logging.getLogger()
logger.addHandler(file_handler)
#end setting log
def count_people(source, rtsp, count_mode, camera_id):
    vs = VideoStream(src=rtsp).start()
    # new_frame_time=0
    # prev_frame_time=0
    # device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    model = YOLO("yolov8m.pt")
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True)
    # model.to(device)
    object_data = {}
    y_center_threshold = 50
    ra = 0
    vao = 0
    stop_time = None  # Thời điểm bắt đầu không thâý đối tượng
    save_timeout = 10  # Timeout để dừng lưu video nếu không có đối tượng trong 2s
    result = None
    show_imshow=True
    current_camera = camera_id
    result_original = None
    size = (new_width, new_height)
    middle_line=0
    if camera_id=="camera1":
        middle_line = middle_line_1
    if camera_id=="camera2":
        middle_line = middle_line_2
    while True:
        try:
            if not vs.stream.isOpened():
                logger.error(f"{camera_id}:disconnected")
                time.sleep(2)
                vs = VideoStream(src=rtsp).start()
            else:
                frame = vs.read()
                if frame is None:
                    break
                frame = cv2.resize(frame, (new_width, new_height))
                # frame = queue.get()
                current_date = datetime.datetime.now().strftime("%d-%m-%Y")
                current_time = time.time()
                if frame is None:
                    break
                if result_original is not None:
                    result_original.write(frame)

                mask = np.zeros_like(frame)
                roi_corners = np.array([lines1], dtype=np.int32)
                cv2.fillPoly(mask, roi_corners, (255, 255, 255))
                results = model.track(frame, persist=True, classes=0, conf=0.5, tracker="bytetrack.yaml",verbose=False)
                annotated_frame = results[0].plot()
                if camera_id =="camera1":
                    for i in range(len(lines1) - 1):
                        cv2.line(annotated_frame, lines1[i], lines1[i + 1], (0, 255, 0), 2)
                    cv2.line(annotated_frame, lines1[-1], lines1[0], (0, 255, 0), 2)
                if camera_id == "camera2":
                    for i in range(len(lines2) - 1):
                        cv2.line(annotated_frame, lines2[i], lines2[i + 1], (0, 255, 0), 2)
                    cv2.line(annotated_frame, lines2[-1], lines2[0], (0, 255, 0), 2)
                    
                output_folder_date = os.path.join(output_folder, current_date, current_camera)
                if not os.path.exists(output_folder_date):
                    os.makedirs(output_folder_date, exist_ok=True)
                # Đường dẫn tới video
                    
                video_folder_date = os.path.join(output_folder_date)
                if not os.path.exists(video_folder_date):
                    os.makedirs(video_folder_date, exist_ok=True)

                output_folder_date_original = os.path.join(output_folder_original, current_date, current_camera)
                if not os.path.exists(output_folder_date_original):
                    os.makedirs(output_folder_date_original, exist_ok=True)
                # Tạo thư mục cho video theo ngày
                    
                video_folder_date_original = os.path.join(output_folder_date_original)
                if not os.path.exists(video_folder_date_original):
                    os.makedirs(video_folder_date_original, exist_ok=True)
                    
                if results[0].boxes.id != None:
                    if result is None and result_original is None:
                        now = datetime.datetime.now()
                        # video_start_time = now
                        dt_string = now.strftime("%H-%M-%S")
                        video_path = os.path.join(
                            video_folder_date, f"{dt_string}_{camera_id}.avi"
                        )
                        result = cv2.VideoWriter(
                            video_path, cv2.VideoWriter_fourcc(*"MJPG"), config_video_output['fps'], size
                        )
                        # video_flag = True
                        # # Đường dẫn tới video
                        video_path_original = os.path.join(
                            video_folder_date_original, f"{dt_string}_{camera_id}.avi"
                        )
                        # Khởi tạo result_original nếu chưa tồn tại
                        result_original = cv2.VideoWriter(
                            video_path_original, cv2.VideoWriter_fourcc(*"MJPG"), config_video_output['fps'], size
                        )
                    stop_time=None
                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        if track_id not in object_data:
                            object_data[track_id] = {'y_centers': [int(y)], 'last_seen': current_time}
                        else:
                            object_data[track_id]['y_centers'].append(int(y))
                            object_data[track_id]['last_seen'] = current_time
                            
                        cv2.circle(annotated_frame, (int(x), int(y)), 5, (255, 0, 0), -1)
                        label_text = f"{track_id}"
                        cv2.putText(annotated_frame, label_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                else:
                    if stop_time is None:
                        stop_time = current_time
                    if result is not None and current_time - stop_time > save_timeout:
                        result.release()
                        result_original.release()
                        result = None
                        result_original = None
                        # video_flag = False

                for object_id, data in list(object_data.items()):
                    if current_time - data['last_seen'] > 1:
                        print(f"P {object_id} disappeared: {data['y_centers']}")
                        print(len(data['y_centers']))
                        print(data['y_centers'][0] - data['y_centers'][-1])
                        error = data['y_centers'][0] - data['y_centers'][-1]
                        if len(data['y_centers']) > 5 and abs(error) > y_center_threshold:
                            print("duoc dem ", object_id)
                            if error > 0:
                                if data['y_centers'][0] > middle_line:
                                    vao += 1
                                    data_x_in = {
                                        "camera_id": camera_id
                                    }
                                    increaseIN(data_x_in)
                            else:
                                ra += 1
                                data_x_out = {
                                    "camera_id": camera_id
                                }
                                increaseOUT(data_x_out)
                        del object_data[object_id]
                if count_mode==0:
                    cv2.putText(annotated_frame, f"Enter: {vao}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(annotated_frame, f"Exit: {ra}", (10, new_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                # cv2.imshow("Tracking", annotated_frame)
                if result is not None:
                    result.write(annotated_frame)
                if show_imshow:
                    cv2.imshow(camera_id, annotated_frame)
                # Nhấn 'q' để thoát khỏi vòng lặp
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            logger.error(f"code:\t{e}")

    cv2.destroyAllWindows()

def delete_old_videos(output_folder, output_folder_original):
    while True:
        try:
            current_time = datetime.datetime.now()
            output_folder_path = pathlib.Path(output_folder)
            for date_folder in output_folder_path.iterdir():
                if date_folder.is_dir() == False:
                    continue

                for camera_folder in date_folder.iterdir():
                    if camera_folder.is_dir()  == False:
                        continue
                    
                    for file_path in camera_folder.iterdir():
                        if file_path.is_file() == False:
                            continue
                        
                        file_creation_time = datetime.datetime.fromtimestamp(
                            file_path.stat().st_ctime
                        )
                        time_difference = current_time - file_creation_time
                        if time_difference.total_seconds() > config["time"]["time_delete"]:
                            parts = list(file_path.parts)
                            parts[0] = output_folder_original
                            original = os.path.join(*parts)
                            file_path.unlink()
                            pathlib.Path(original).unlink()
                            # Xóa tệp tin
                            with open("delete.log", "a") as f:
                                f.write(f"Deleted old video: {file_path.name} \n ") 
                                

                    # Kiểm tra xem thư mục camera có trống không
                    if not list(camera_folder.iterdir()):
                        camera_folder.rmdir()  # Xóa thư mục camera
                        with open("delete.log", "a") as f:
                            f.write(f"Deleted camera folder: {camera_folder.name} \n" ) 

                # Kiểm tra xem thư mục date có trống không sau khi xóa video
                if not list(date_folder.iterdir()):
                    date_folder.rmdir()  # Xóa thư mục date
                    with open("delete.log", "a") as f:
                        f.write(f"Deleted date folder: {date_folder.name} \n" ) 
        except Exception as e:
            logger.error("delete error: ", e)

if __name__ == '__main__':
    send_requests_interval()
    # Tạo Queue để trao đổi dữ liệu giữa các tiến trình
    vitri = {
        "left_right": 1,
        "top-bottom": 0
    }
    rtsp1 = config["camera_config2"][camera_ip_top]["source"]
    rtsp2= config["camera_config2"][camera_ip_bottom]["source"]

    # detect_process1 = threading.Thread(target=detect_objects, args=(queue1, rtsp1))
    # detect_process2 = threading.Thread(target=detect_objects, args=(queue2,  rtsp2))

    # detect_process1.start()
    # detect_process2.start()

    track_process1 = Process(target=count_people, args=(camera_ip_top, rtsp1, vitri["top-bottom"], "camera1"))
    track_process2 = Process(target=count_people, args=(camera_ip_bottom,rtsp2, vitri["top-bottom"], "camera2"))

    track_process1.start()
    track_process2.start()

    delete_process = threading.Thread(target=delete_old_videos, args=(output_folder, output_folder_original))
    delete_process.start()

    # detect_process1.join()
    # detect_process2.join()

    delete_process.join()
    track_process1.join()
    track_process2.join()
    # Đánh dấu kết thúc cho tiến trình theo dõi

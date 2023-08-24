import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import cv2
import numpy as np
import random
from sort.sort import Sort
from multiprocessing import Process, Queue

# import socket
import datetime
import time
from mongoFunc import increaseIN, increaseOUT, send_requests_interval
import json

# with open('config.json') as config_file:
#     config = json.load(config_file)

# Khởi tạo kích thước mới cho frame
new_width, new_height = 640, 480

# Create an output folder if it doesn't exist
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)

# Đường dẫn đến video source1 và source2
# source1 = "toilet.mp4"
# source2 = "tau.mp4"
source1 = 0
source2 = 1  # 1


def load_config():
    with open("config.json") as config_file:
        config = json.load(config_file)
    return config


# Đường dẫn đến tệp people.pb
model_path = "people.pb"


def detect_objects(queue, source):
    # Tải mô hình
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Tạo session và nạp mô hình vào đó
    with tf.compat.v1.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

        # Lấy các tensor đầu vào và đầu ra
        input_tensor_name = "image_tensor:0"
        output_tensors_names = [
            "detection_boxes:0",
            "detection_scores:0",
            "detection_classes:0",
            "num_detections:0",
        ]
        input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
        output_tensors = [
            sess.graph.get_tensor_by_name(name) for name in output_tensors_names
        ]
        # size = (new_width, new_height)
        # Khởi tạo video từ file source
        while True:
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                print("Camera disconnected. Reconnecting...")
                time.sleep(5)  # Wait for a few seconds before retrying
            else:
                print("Camera reconnected. Displaying video...")
                break

        while True:
            # Đọc frame từ video
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame. Reconnecting...")
                cap.release()
                detect_objects(queue, source)
                break
            
            # Thay đổi kích thước frame thành new_width x new_height
            frame = cv2.resize(frame, (new_width, new_height))
            # Xử lý frame và chuẩn bị đầu vào cho mô hình
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_expanded = np.expand_dims(frame_rgb, axis=0)

            # Đưa frame vào mô hình để phát hiện đối tượng
            detections = sess.run(
                output_tensors, feed_dict={input_tensor: frame_expanded}
            )

            # Xử lý kết quả và đưa vào queue
            boxes = detections[0][0]
            scores = detections[1][0]
            classes = detections[2][0]
            num_detections = int(detections[3][0])
            valid_boxes = boxes[:num_detections]
            queue.put((frame, valid_boxes, scores))

        cap.release()
        cv2.destroyAllWindows()
config = load_config()


def track_objects(queue, source, count_mode, camera_id):
    # Tạo bộ lọc SORT
    # mot_tracker = Sort();

    mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    # Danh sách màu sẵn có để gán cho các đối tượng theo dõi
    colors = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(1000)
    ]
    # Các biến để lưu trữ thông tin về các đối tượng trong vùng quan tâm
    tracked_objects_in_roi = (
        {}
    )  # Dictionary với object_id là key và thông tin về đối tượng là value

    # Khởi tạo tọa độ vùng quan tâm cho source1 và source2
    if source == source1:
        roi_x1, roi_y1, roi_x2, roi_y2 = (
            (int(new_width / 2)),
            0,
            (int(new_width / 2) + 50),
            new_height,
        )
    elif source == source2:
        roi_x1, roi_y1, roi_x2, roi_y2 = (
            0,
            (int(new_height / 2) + 30),
            new_width,
            (int(new_height / 2) + 80),
        )

    count_people_entered_x = 0
    count_people_exited_x = 0
    count_people_entered_y = 0
    count_people_exited_y = 0

    # Biến để xác định phương pháp đếm (trục x hoặc trục y)

    cv2.namedWindow(camera_id)
    size = (new_width, new_height)
    # Khởi tạo video từ file source

    video_flag = False
    video_start_time = datetime.datetime.now()
    result = None
    while True:
        frame, valid_boxes, scores = queue.get()
        if frame is None:
            break
        # Sử dụng SORT để theo dõi các đối tượng
        tracked_objects = mot_tracker.update(valid_boxes)
        if len(tracked_objects) > 0 and video_flag == False:
            now = datetime.datetime.now()
            video_start_time = now
            dt_string = now.strftime("%Y-%m-%d %H-%M-%S")
            video_path = os.path.join(output_folder, f"{dt_string}_{camera_id}.avi")
            result = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*"MJPG"), 10, size
            )
            video_flag = True
        if (
            video_flag == True
            and datetime.datetime.timestamp(datetime.datetime.now())
            - datetime.datetime.timestamp(video_start_time)
            > config["time"]["time_save"]
        ):
            result.release()
            result = None
            video_flag = False

        # Chọn tọa độ vùng quan tâm phù hợp cho từng source
        if count_mode == 1:
            roi_x1, roi_y1, roi_x2, roi_y2 = (
                (int(new_width / 2) - 40),
                0,
                (int(new_width / 2) + 40),
                new_height,
            )
        elif count_mode == 0:
            roi_x1, roi_y1, roi_x2, roi_y2 = (
                0,
                (int(new_height / 2) + 40),
                new_width,
                (int(new_height / 2) + 120),
            )

        # Cập nhật thông tin về các đối tượng trong vùng quan tâm và đếm người đi vào và đi ra
        objects_in_roi = []
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, object_id = obj
            left, right, top, bottom = (
                int(ymin * new_width),
                int(ymax * new_width),
                int(xmin * new_height),
                int(xmax * new_height),
            )
            x_center = int(left + (right - left) / 2)
            y_center = int(top + (bottom - top) / 2)

            # Kiểm tra xem đối tượng có nằm trong vùng quan tâm không
            if roi_x1 <= x_center <= roi_x2 and roi_y1 <= y_center <= roi_y2:
                objects_in_roi.append(object_id)

                # Kiểm tra xem đối tượng có đã được lưu trữ trước đó không
                if object_id in tracked_objects_in_roi:
                    tracked_objects_in_roi[object_id].append((x_center, y_center))
                else:
                    tracked_objects_in_roi[object_id] = [(x_center, y_center)]
            else:
                # Nếu đối tượng ra khỏi vùng quan tâm, kiểm tra xem nó có đi lên/xuống hay sang trái/phải
                if object_id in tracked_objects_in_roi:
                    path = tracked_objects_in_roi[object_id]
                    first_x = path[0][0]
                    last_x = path[-1][0]
                    first_y = path[0][1]
                    last_y = path[-1][1]
                    x_diff = last_x - first_x
                    y_diff = last_y - first_y

                    if count_mode == 1:
                        if x_diff > 0:
                            if config["camera_config1"][source]["IN"] == "left-right":
                                count_people_entered_x += 1
                                data_x_in = {
                                    "camera_id": camera_id,
                                    "number_of_guest_in": 1,
                                    "number_of_guest_out": 0,
                                }
                                increaseIN(
                                    data_x_in
                                )  # Call the function with the defined data
                            else:
                                count_people_exited_x += 1
                                data_x_out = {
                                    "camera_id": camera_id,
                                    "number_of_guest_in": 0,
                                    "number_of_guest_out": 1,
                                }
                                increaseOUT(
                                    data_x_out
                                )  # Call the function with the defined data

                        elif x_diff < 0:
                            if config["camera_config1"][source]["IN"] == "right-left":
                                count_people_entered_x += 1
                                data_x_in = {
                                    "camera_id": camera_id,
                                    "number_of_guest_in": 1,
                                    "number_of_guest_out": 0,
                                }
                                increaseIN(
                                    data_x_in
                                )  # Call the function with the defined data
                            else:
                                count_people_exited_x += 1
                                data_x_out = {
                                    "camera_id": camera_id,
                                    "number_of_guest_in": 0,
                                    "number_of_guest_out": 1,
                                }
                                increaseOUT(
                                    data_x_out
                                )  # Call the function with the defined data

                    # Đếm số người đi lên và đi xuống theo trục y
                    elif count_mode == 0:
                        if y_diff > 0:
                            if config["camera_config2"][source]["IN"] == "top-bottom":
                                count_people_entered_y += 1
                                data_y_in = {
                                    "camera_id": camera_id,
                                    "number_of_guest_in": 1,
                                    "number_of_guest_out": 0,
                                }
                                increaseIN(
                                    data_y_in
                                )  # Call the function with the defined data
                            else:
                                count_people_exited_y += 1
                                data_y_out = {
                                    "camera_id": camera_id,
                                    "number_of_guest_in": 0,
                                    "number_of_guest_out": 1,
                                }
                                increaseOUT(data_y_out)
                        elif y_diff < 0:
                            if config["camera_config2"][source]["IN"] == "bottom-top":
                                count_people_entered_y += 1
                                data_y_in = {
                                    "camera_id": camera_id,
                                    "number_of_guest_in": 1,
                                    "number_of_guest_out": 0,
                                }
                                increaseIN(data_y_in)
                            else:
                                count_people_exited_y += 1
                                data_y_out = {
                                    "camera_id": camera_id,
                                    "number_of_guest_in": 0,
                                    "number_of_guest_out": 1,
                                }
                                increaseOUT(
                                    data_y_out
                                )  # Call the function with the defined data

                    del tracked_objects_in_roi[object_id]

        # Vẽ đối tượng và hiển thị frame với các hộp giới hạn và đường đi
        for obj in tracked_objects:
            xmin, ymin, xmax, ymax, object_id = obj
            object_color = colors[int(object_id)]
            left, right, top, bottom = (
                int(ymin * new_width),
                int(ymax * new_width),
                int(xmin * new_height),
                int(xmax * new_height),
            )
            # Vẽ đường đi của đối tượng (các chấm tròn thể hiện các điểm trong quá khứ mà đối tượng đã đi qua)
            for point in tracked_objects_in_roi.get(object_id, []):
                cv2.rectangle(frame, (left, top), (right, bottom), object_color, 2)
                cv2.circle(frame, point, 2, object_color, -1)
        # Vẽ hình chữ nhật vùng quan tâm lên frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        if count_mode == 1:
            cv2.putText(
                frame,
                f"Ra: {count_people_exited_x}, Vao: {count_people_entered_x}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        elif count_mode == 0:
            cv2.putText(
                frame,
                f"Ra: {count_people_exited_y}, Vao: {count_people_entered_y}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        if result is not None:
            result.write(frame)
        # Hiển thị frame với các hộp giới hạn
        cv2.imshow(camera_id, frame)

        # Nhấn 'q' để thoát khỏi vòng lặp
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def delete_old_videos(output_folder):
    while True:
        current_time = datetime.datetime.now()
        # Lấy danh sách các tệp trong thư mục "output_videos"
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                file_creation_time = datetime.datetime.fromtimestamp(
                    os.path.getctime(file_path)
                )
                time_difference = current_time - file_creation_time
                if time_difference.total_seconds() > config["time"]["time_delete"]:
                    os.remove(file_path)
                    print(f"Deleted old video: {filename}")


if __name__ == "__main__":
    # Tạo Queue để trao đổi dữ liệu giữa các tiến trình
    send_requests_interval()
    queue1 = Queue()
    queue2 = Queue()

    # Tạo hai tiến trình cho từng source
    detect_process1 = Process(target=detect_objects, args=(queue1, source1))
    track_process1 = Process(target=track_objects, args=(queue1, source1, 1, "camera1"))

    detect_process2 = Process(target=detect_objects, args=(queue2, source2))
    track_process2 = Process(target=track_objects, args=(queue2, source2, 1, "camera2"))
    # Bắt đầu hai tiến trình
    detect_process1.start()
    track_process1.start()

    detect_process2.start()
    track_process2.start()
    # delete
    delete_process = Process(target=delete_old_videos, args=(output_folder,))
    delete_process.start()
    # Đợi hai tiến trình hoàn thành
    detect_process1.join()
    track_process1.join()

    detect_process2.join()
    track_process2.join()

    # Đánh dấu kết thúc cho tiến trình theo dõi
    queue1.put((None, None, None))
    queue2.put((None, None, None))

    # Giải phóng tài nguyên Queue
    queue1.close()
    queue1.join_thread()

    queue2.close()
    queue2.join_thread()

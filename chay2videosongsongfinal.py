import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import cv2
import numpy as np
from sort.sort import Sort
from multiprocessing import Process, Queue
import time
import json
import datetime
import pathlib
import threading
from mongoFunc import send_requests_interval, increaseIN, increaseOUT

from threading import Thread

output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)


output_folder_original="output_videos_original"
os.makedirs(output_folder_original, exist_ok=True)

# usb_drive = "E:\\"  # Thay đổi thành đường dẫn đúng của USB của bạn
# output_folder_usb = "E:\\output_videos"
# output_folder_original_usb = "E:\\output_videos_original"
# os.makedirs(output_folder_usb, exist_ok=True)


def load_config():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    return config


config = load_config()
mid_line = 120
distance = 50

# Đường dẫn đến video source1 và source2
# source1 = "xanh (2).avi"
source1 = 0
source2 = 1

# Đường dẫn đến tệp people.pb
model_path = "people.pb"
y_diff = 0


def detect_objects(queue, source):
    new_width = config["camera_config1"][source]["roi"]["new_width"]
    new_height = config["camera_config1"][source]["roi"]["new_height"]
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

        while True:
            
            # Đọc frame từ video
            cap = cv2.VideoCapture(source)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"FPS của video: {fps}")
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

        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()

# def overwrite_file(source_file, destination_file):
#     if os.path.isdir(source_file):
#         os.makedirs(destination_file, exist_ok=True)
#     else:
#         with open(source_file, "rb") as f_in:
#             with open(destination_file, "wb") as f_out:
#                 f_out.write(f_in.read())
#             f_out.close()
#         f_in.close()

# def copy_to_usb(output_folder,output_folder_original):
#     destination_folder = os.path.join(usb_drive, "output_videos")
#     destination_folder_original=os.path.join(usb_drive,"output_videos_original")
#     shutil.copytree(output_folder, destination_folder, copy_function=overwrite_file, dirs_exist_ok=True)   
#     shutil.copytree(output_folder_original, destination_folder_original, copy_function=overwrite_file, dirs_exist_ok=True) 

def track_objects(queue, source, count_mode, camera_id):
    global y_diff
    new_width = config["camera_config1"][source]["roi"]["new_width"]
    new_height = config["camera_config1"][source]["roi"]["new_height"]
    # Tạo bộ lọc SORT
    mot_tracker = Sort(20, 2, 0.01)
    # threads:list[Thread] = [] 
    # colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(1000)]
    # colors = (0, 255, 255)
    # Các biến để lưu trữ thông tin về các đối tượng trong vùng quan tâm
    tracked_objects_in_roi = (
        {}
    )  # Dictionary với object_id là key và thông tin về đối tượng là value
    cv2.namedWindow(camera_id)
    # Biến để lưu thời gian cuối cùng mà mỗi đối tượng đã được cập nhật
    last_update_times = {}
    left_to_right_count = 0
    right_to_left_count = 0
    # Trước vòng lặp track_objects
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(os.path.join(output_folder,f"{source1}_daxuly.avi", fourcc, 20.0, (new_width, new_height)))
    # save = True
    video_flag = False
    video_start_time = datetime.datetime.now()
    result = None
    result_original = None
    show_imshow = True
    size = (new_width, new_height)
    while True:
        try:
            # for thread in threads:
            #     if thread.ident== None:
            #         if thread.is_alive()==None:
            #             thread.join()
            frame, valid_boxes, scores = queue.get()
            if frame is None:
                break
            if result_original is not None:
                result_original.write(frame)
            # Sử dụng SORT để theo dõi các đối tượng
            tracked_objects = mot_tracker.update(valid_boxes)
            current_date = datetime.datetime.now().strftime("%d-%m-%Y")
            current_camera = camera_id

            output_folder_date = os.path.join(output_folder, current_date, current_camera)
            os.makedirs(output_folder_date, exist_ok=True)
            # Đường dẫn tới video
            video_folder_date = os.path.join(output_folder_date)
            os.makedirs(video_folder_date, exist_ok=True)

            output_folder_date_original = os.path.join(output_folder_original, current_date, current_camera)
            os.makedirs(output_folder_date_original, exist_ok=True)
            # Tạo thư mục cho video theo ngày
            video_folder_date_original = os.path.join(output_folder_date_original)
            os.makedirs(video_folder_date_original, exist_ok=True)
            now = datetime.datetime.now()
            # Ghi khung hình vào video

            if len(tracked_objects) > 0 and video_flag == False:
                video_start_time = now
                dt_string = now.strftime("%H-%M-%S")
                video_path = os.path.join(
                    video_folder_date, f"{dt_string}_{camera_id}.avi"
                )
                result = cv2.VideoWriter(
                    video_path, cv2.VideoWriter_fourcc(*"MJPG"), 10, size
                )
                video_flag = True
                # # Đường dẫn tới video
                video_path_original = os.path.join(
                    video_folder_date_original, f"{dt_string}_{camera_id}.avi"
                )
                # Khởi tạo result_original nếu chưa tồn tại
                result_original = cv2.VideoWriter(
                    video_path_original, cv2.VideoWriter_fourcc(*"MJPG"), 10, size
                )

            if (
                video_flag == True
                and datetime.datetime.timestamp(datetime.datetime.now())
                - datetime.datetime.timestamp(video_start_time)
                > config["time"]["time_save"]
            ):
                result.release()
                result_original.release()
                result = None
                result_original = None
                video_flag = False
                # threads.append(Thread(target=copy_to_usb, args=(output_folder,output_folder_original)))
                # threads[-1].start()
            # Cập nhật thông tin về các đối tượng trong vùng quan tâm và đếm người đi vào và đi ra
            objects_in_roi = []
            for obj in tracked_objects:
                xmin, ymin, xmax, ymax, object_id = obj
                # width=round(float(xmax - xmin), 3)
                # height=round(float(ymax - ymin), 3)
                # squad = round(float(width * height),3)
                # if (squad) < 0.070:
                # print(object_id,f"Squad: {squad}")
                object_color = (0, 255, 255)  # màu vàng
                left, right, top, bottom = (
                    int(ymin * new_width),
                    int(ymax * new_width),
                    int(xmin * new_height),
                    int(xmax * new_height),
                )
                x_center = int(left + (right - left) / 2)
                y_center = int(top + (bottom - top) / 2)
                cv2.rectangle(frame, (left, top), (right, bottom), object_color, 2)
                cv2.circle(frame, (x_center, y_center), 2, object_color, -1)
                # object_score=score
                cv2.putText(
                    frame,
                    f"ID:{object_id}",
                    (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                # Kiểm tra xem đối tượng có nằm trong vùng quan tâm không
                objects_in_roi.append(object_id)

                # Cập nhật thời gian cuối cùng cho đối tượng
                last_update_times[object_id] = time.time()

                # Kiểm tra xem đối tượng có đã được lưu trữ trước đó không
                if object_id in tracked_objects_in_roi:
                    tracked_objects_in_roi[object_id].append((x_center, y_center))
                else:
                    tracked_objects_in_roi[object_id] = [(x_center, y_center)]

            # Kiểm tra và in thông tin về đối tượng không còn trong khung hình
            current_time = time.time()
            objects_to_remove = []
            for object_id, last_update_time in last_update_times.items():
                if (
                    current_time - last_update_time > 1
                ):  # Kiểm tra trong khoảng 1 giây
                    objects_to_remove.append(object_id)
            for object_id_to_remove in objects_to_remove:
                if object_id_to_remove in tracked_objects_in_roi:
                    removed_object_info = tracked_objects_in_roi.pop(
                        object_id_to_remove
                    )
                    # Xác định hướng đi của đối tượng
                    if removed_object_info and len(removed_object_info) > 10:
                        # print(object_id_to_remove,removed_object_info)
                        start_x, _ = removed_object_info[
                            0
                        ]  # Tọa độ x của điểm đầu tiên
                        end_x, _ = removed_object_info[
                            -1
                        ]  # Tọa độ x của điểm cuối cùng
                        # Xác định ngưỡng (threshold) để loại bỏ các trường hợp nhiễu, bắt đầu từ 0 kết thúc ở 320 mới đếm
                        threshold = 75  # Có thể điều chỉnh ngưỡng này theo ý muốn
                        # Tính khoảng cách giữa start_x và end_x
                        distance_x = abs(end_x - start_x)
                        # print(object_id_to_remove,distance_x,end_x,start_x,removed_object_info)
                        # if distance_x >= threshold:
                        #     direction = "ra" if start_x < end_x else "vao"
                        #     print(f"ID {object_id_to_remove} out frame. Info: {removed_object_info}")
                        #     if direction == "ra":
                        #         if end_x > 200:
                        #             left_to_right_count += 1
                        #     else:
                        #         if end_x < 100:
                        #             right_to_left_count += 1
                        #     print(f"ID {object_id_to_remove} {direction}: {removed_object_info}")
                        if distance_x >= threshold:
                            if config["camera_config1"][source]["IN"] == "right-left":
                                direction = "vao" if start_x > end_x else "ra"
                                # print(f"ID {object_id_to_remove} out frame. Info: {removed_object_info}")
                                if direction == "ra":
                                    if end_x > 200:
                                        left_to_right_count += 1
                                        data_x_out = {
                                            "camera_id": camera_id
                                        }
                                        increaseOUT(data_x_out)
                                else:
                                    if end_x < 100:
                                        right_to_left_count += 1
                                        data_x_in = {
                                            "camera_id": camera_id
                                        }
                                        increaseIN(data_x_in)
                            else:
                                direction = "vao" if start_x < end_x else "ra"
                                # print(f"ID {object_id_to_remove} out frame. Info: {removed_object_info}")
                                if direction == "ra":
                                    if end_x < 100:
                                        right_to_left_count += 1
                                        data_x_out = {
                                            "camera_id": camera_id
                                        }
                                        increaseOUT(data_x_out)
                                else:
                                    if end_x > 200:
                                        left_to_right_count += 1
                                        data_x_in = {
                                            "camera_id": camera_id
                                        }
                                        increaseIN(data_x_in)
                            # Hiển thị tổng số lượt đi trên khung hình
            if count_mode == 1:
                # Lấy chiều rộng của khung hình
                frame_width = frame.shape[1]
                # Đặt văn bản "ra" ở bên trái ngoài cùng

                text_width_vao = cv2.getTextSize(
                    f"vao: {right_to_left_count}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )[0][0]
                x_coordinate_ra = (
                    frame_width - text_width_vao - 10
                )  # 10 là khoảng cách từ mép phải
                if config["camera_config1"][source]["IN"] == "left-right":
                    # Đặt văn bản "ra" ở bên trái ngoài cùng

                    cv2.putText(
                        frame,
                        f"ra: {right_to_left_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"vao: {left_to_right_count}",
                        (x_coordinate_ra, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                if config["camera_config1"][source]["IN"] == "right-left":
                    cv2.putText(
                        frame,
                        f"ra: {left_to_right_count}",
                        (x_coordinate_ra, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"vao: {right_to_left_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    # Đặt văn bản "vào" ở bên trái cùng

            if result is not None:
                result.write(frame)
                # Hiển thị frame với các hộp giới hạn
            if cv2.waitKey(1) & 0xFF == ord("s"):  # ấn s để ngừng imshow
                show_imshow = True if show_imshow == False else False
            if show_imshow:
                cv2.imshow(camera_id, frame)

            # Nhấn 'q' để thoát khỏi vòng lặp
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print("Error :", e)
    # cv2.destroyAllWindows()

def delete_old_videos(output_folder,output_folder_original):
    while True:
        try:
            current_time = datetime.datetime.now()
            output_folder_path = pathlib.Path(output_folder)
            for date_folder in output_folder_path.iterdir():
                    if date_folder.is_dir():
                        for camera_folder in date_folder.iterdir():
                            if camera_folder.is_dir():
                                for file_path in camera_folder.iterdir():
                                    if file_path.is_file():
                                        file_creation_time = datetime.datetime.fromtimestamp(
                                            file_path.stat().st_ctime
                                        )
                                        time_difference = current_time - file_creation_time
                                        if time_difference.total_seconds() > config["time"]["time_delete"]:
                                            parts = list(file_path.parts)
                                            parts[0]=output_folder_original
                                            original=os.path.join(*parts)
                                            file_path.unlink()
                                            pathlib.Path(original).unlink()
                                              # Xóa tệp tin
                                            print(f"Deleted old video: {file_path.name}")

                                # Kiểm tra xem thư mục camera có trống không
                                if not list(camera_folder.iterdir()):
                                    camera_folder.rmdir()  # Xóa thư mục camera
                                    print(f"Deleted camera folder: {camera_folder.name}")

                        # Kiểm tra xem thư mục date có trống không sau khi xóa video
                        if not list(date_folder.iterdir()):
                            date_folder.rmdir()  # Xóa thư mục date
                            print(f"Deleted date folder: {date_folder.name}")
        except Exception as e:
            print("Error: ", e)

# def get_total_space(path):
#     # Tính tổng dung lượng có thể lưu trữ trên đường dẫn path (ổ USB) và trả về kết quả trong đơn vị GB
#     total_space = psutil.disk_usage(path).total
#     return total_space / (1024**3)  # Chuyển từ byte sang GB


# def get_used_space(path):
#     # Tính dung lượng đã sử dụng trên đường dẫn path (ổ USB) và trả về kết quả trong đơn vị GB
#     used_space = psutil.disk_usage(path).used
#     return used_space / (1024**3)  # Chuyển từ byte sang GB


# def delete_old_videos_and_check_space(usb_drive):
#     while True:
#         try:
#             # current_time = datetime.datetime.now()
#             usb_drive_path = pathlib.Path(usb_drive)
#             # output_folder_usb_path = pathlib.Path(output_folder_usb)
#             # output_folder_original_usb_path=pathlib.Path(output_folder_original_usb)

#             # Chuyển đổi đường dẫn sang chuỗi trước khi sử dụng psutil
#             usb_drive_path_str = usb_drive_path.as_posix()

#             # Tính tổng dung lượng có thể lưu trữ trên ổ USB trong GB
#             total_space_gb = get_total_space(usb_drive_path_str)
#             used_space_gb = get_used_space(usb_drive_path_str)
#             free_space_gb = total_space_gb - used_space_gb

#             # Ghi thông tin dung lượng vào log
#             with open("ahihi.log", "w+") as f:
#                 f.write(f"Total space: {total_space_gb} GB\n")
#                 f.write(f"Used space: {used_space_gb} GB\n")
#                 f.write(f"Free space: {free_space_gb} GB\n")
#             abc = list()
#             if free_space_gb < config["space"]["space_threshold"]:
#                 oldest_video = None
#                 # oldest_video_time = current_time
#                 for output_videos_folder in usb_drive_path.iterdir():
#                     if len(output_videos_folder.parts) != 0 :
#                         if output_videos_folder.parts[-1] in ["output_videos", "output_videos_original"]:
#                             for date_folder in output_videos_folder.iterdir():
#                                 if date_folder.is_dir():
#                                     for camera_folder in date_folder.iterdir():    
#                                         if camera_folder.is_dir():
#                                             for file_path in camera_folder.iterdir():
#                                                 if file_path.is_file():
#                                                     file_creation_time = (
#                                                         datetime.datetime.fromtimestamp(
#                                                             file_path.stat().st_ctime
#                                                         )
#                                                     )
#                                                     abc.append((file_path, file_creation_time))

#                                                 # if file_creation_time < oldest_video_time:
#                                                 #     oldest_video = file_path
#                                                 #     oldest_video_time = file_creation_time
#             abc.sort(key=lambda x: x[1])
#             if len(abc)>0:
#                 while free_space_gb < config["space"]["space_threshold"]:
#                     oldest_video = abc.pop(0)[0]
#                     if oldest_video:
#                         oldest_video.unlink()

#                         print(f"Deleted old video from usb: {oldest_video.name}")
#                     else:
#                         print("No more old videos to delete.")

#                     # Tính lại dung lượng sau khi xóa video
#                     used_space_gb = get_used_space(usb_drive_path_str)
#                     free_space_gb = total_space_gb - used_space_gb

#             # Kiểm tra và xóa các thư mục rỗng
#             for date_folder in usb_drive_path.iterdir():
#                 if date_folder.is_dir():
#                     for camera_folder in date_folder.iterdir():
#                         if camera_folder.is_dir() and not list(camera_folder.iterdir()):
#                             camera_folder.rmdir()
#                             print(
#                                 f"Deleted camera folder from usb: {camera_folder.name}"
#                             )
#                     if not list(date_folder.iterdir()):
#                         date_folder.rmdir()
#                         print(f"Deleted date folder from usb: {date_folder.name}")
#         except Exception as e:
#             print("Error : ", e)


if __name__ == "__main__":
    # Tạo Queue để trao đổi dữ liệu giữa các tiến trình
    send_requests_interval()
    queue1 = Queue()
    queue2 = Queue()

    detect_process1 = Process(target=detect_objects, args=(queue1, source1))
    detect_process2 = Process(target=detect_objects, args=(queue2, source2))

    track_process1 = Process(target=track_objects, args=(queue1, source1, 1, "camera1"))
    track_process2 = Process(target=track_objects, args=(queue2, source2, 1, "camera2"))


    detect_process1.start()
    detect_process2.start()

    track_process1.start()
    track_process2.start()
    # delete output_folder
    delete_process = threading.Thread(target=delete_old_videos, args=(output_folder,output_folder_original,))
    delete_process.start()

    # delete_usb = threading.Thread(target=delete_old_videos_and_check_space, args=(usb_drive,))
    # delete_usb.start()

    # Đợi hai tiến trình hoàn thành
    detect_process1.join()
    detect_process2.join()

    # delete_usb.join()

    delete_process.join()

    track_process1.join()
    track_process2.join()
    # Đánh dấu kết thúc cho tiến trình theo dõi
    queue1.put((None, None, None))
    queue2.put((None, None, None))
    # Giải phóng tài nguyên Queue
    queue1.close()
    queue2.close()

    queue1.join_thread()
    queue2.join_thread()

import requests  # pip install requests
import pytz
from datetime import datetime
from pymongo.mongo_client import MongoClient  # pip install pymongo
from pymongo.server_api import ServerApi
import threading
import json
import logging

client = MongoClient("mongodb://localhost:27017/", server_api=ServerApi("1"))
mydb = client["counter"]
mycol = mydb["in_train"]
API_ENDPOINT = "https://demnguoi.halovi.com.vn/api/counter-data"

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filemode="a", format=log_format, level=logging.DEBUG)


# Create a FileHandler with a dynamic file name based on the current date
log_file_name = datetime.now().strftime("log_%Y-%m-%d.log")
file_handler = logging.FileHandler(log_file_name)

# Set the formatter for the FileHandler
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger = logging.getLogger()
logger.addHandler(file_handler)

# The line `camera_id='camera_id'` is initializing a variable `camera_id` with the value
# `'camera_id'`. This is the default value for the `camera_id` parameter in the functions `increaseIN`
# and `increaseOUT`. If the `camera_id` parameter is not provided when calling these functions, it
# will default to `'camera_id'`.
#   data format:
#   {
#   "camera_id": "f50f83cb-3b03-4d1c-b290-0f3590c6399b",
#   "number_of_guest_in": 10,
#   "number_of_guest_out": 4
#   }


def saveData(data):
    data["time"] = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]
    mycol.insert_one(data)
    return True


def increaseIN(data):
    camera_id = data["camera_id"]
    data_format = {
        "camera_id": camera_id,
        "number_of_guest_in": 1,
        "number_of_guest_out": 0,
    }
    saveData(data_format)


def increaseOUT(data):
    camera_id = data["camera_id"]
    data_format = {
        "camera_id": camera_id,
        "number_of_guest_in": 0,
        "number_of_guest_out": 1,  # Giữ số người ra nguyên
    }
    saveData(data_format)


def deleteData(camera_id, time):
    return mycol.delete_many({"time": {"$lte": time}, "camera_id": camera_id})


def sendRequest():
    time = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]
    result = mycol.find({"time": {"$lte": time}})
    data = []
    for doc in result:
        for index, dictionary in enumerate(data):
            if dictionary.get("camera_id") == doc["camera_id"]:
                dictionary["data"].append(
                    {
                        "id": str(doc["_id"]),
                        "number_of_guest_in": doc["number_of_guest_in"],
                        "number_of_guest_out": doc["number_of_guest_out"],
                        "time": doc["time"],
                    }
                )
                break
        else:
            data.append(
                {
                    "camera_id": doc["camera_id"],
                    "time": time,
                    "data": [
                        {
                            "id": str(doc["_id"]),
                            "number_of_guest_in": doc["number_of_guest_in"],
                            "number_of_guest_out": doc["number_of_guest_out"],
                            "time": doc["time"],
                        }
                    ],
                }
            )

    if len(data) > 0:
        for body in data:
            print(body)
            logger.info(time)
            logger.info(body)
            try:
                res = requests.post(url=API_ENDPOINT, json=body)
                if res.status_code == 200:
                    deleteData(body["camera_id"], time)
                    print("SUCCESS")
                    logger.info("SUCCESS")
                else:
                    logger.info(res.content)
                    logger.info("FAIL")
                    print("FAIL")
                    print(res.content)
            except requests.exceptions.RequestException as e:
                logger.error(e)
                print(e)
                return "UNKNOWN_ERROR"


def load_config():
    with open("config.json") as config_file:
        config = json.load(config_file)
    return config


config = load_config()


def send_requests_interval():
    sendRequest()
    threading.Timer(config["timeAPI"]["time_api"], send_requests_interval).start()


if __name__ == "__main__":
    send_requests_interval()

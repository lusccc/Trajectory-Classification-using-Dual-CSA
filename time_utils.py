from datetime import datetime
import time
def datatime_to_timestamp(dt):
    # print dt
    """时间转换为时间戳，单位秒"""
    # 转换成时间数组
    time_array = time.strptime(str(dt), "%Y-%m-%d %H:%M:%S")
    # 转换成时间戳
    timestamp = time.mktime(time_array)  # 单位 s
    # print timestamp
    return int(timestamp)

def timestamp_to_hour(ts):
    return datetime.fromtimestamp(ts).hour
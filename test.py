#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import rospy
import rosbag
import cv2
from cv_bridge import CvBridge
import os
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2
import numpy as np

import tqdm
import bisect
    


def _find_timestamp(timestamp,search_list):
    idx = bisect.bisect_left(search_list, timestamp)
    if idx == 0:
        return 0
    if idx == len(search_list):
        return 0
    before = search_list[idx - 1]
    after = search_list[idx]

    af = after-timestamp
    be = timestamp-before
    return af if (af < be) else be

def find_nearest_timestamp(current_sensor_timestamp,search_list):
    tf = []
    for ts in current_sensor_timestamp:
        time_differ = _find_timestamp(ts,search_list)
        #print(f'_{time_differ}')
        # print(time_differ)
        tf.append(time_differ)
        # print(f'time differ with nearest timestamp: {time_differ}')
    #print(f'average timestamp: {np.mean(np.array(tf))}')
    
    # avg_td=np.mean(np.array(tf,dtype=np.float64))
    #print(f'_________{tf[-1]}')
    avg_td = np.mean(np.array(tf))
    max_td = np.max(np.array(tf))
    return avg_td, max_td

CAMERA_MAPPING = {
        '/cameras/front/image_color/compressed': 'camera_FRONT',
        '/cameras/front_right/image_color/compressed': 'camera_FRONT_RIGHT',
        '/cameras/rear_right/image_color/compressed': 'camera_BACK_RIGHT',
        '/cameras/rear/image_color/compressed': 'camera_BACK',
        '/cameras/rear_left/image_color/compressed': 'camera_BACK_LEFT',
        '/cameras/front_left/image_color/compressed': 'camera_FRONT_LEFT',
        '/pandar_points': 'Lidar'
}

def match2front_camera(bag_file):
    timestamps=defaultdict(list)

    with rosbag.Bag(bag_file, 'r') as bag:
        msgs =bag.read_messages()
        for topic, msg, t in msgs:#tqdm.tqdm():
            if topic in CAMERA_MAPPING.keys():
                timestamp = msg.header.stamp.to_sec()
                timestamps[topic].append(timestamp)
                #timestamps[topic].append(t.to_sec())
    for k in timestamps.keys():
        timestamps[k] = [_ - timestamps[k][0] for _ in timestamps[k]]
            
    front_key = '/cameras/front/image_color/compressed'
    for k in CAMERA_MAPPING.keys():
        if k == front_key:
            continue
        else:
            avg_td, max_td =find_nearest_timestamp(timestamps[k],timestamps[front_key])
            print(f'mean time difference [{k}]: {avg_td}')
            print(f'max time difference [{k}]: {max_td}')

def td_with_prev(bag_file):
    tds=defaultdict(list)

    pre_timestamp=dict()
    with rosbag.Bag(bag_file, 'r') as bag:
        msgs =bag.read_messages()
        for topic, msg, t in msgs:#tqdm.tqdm():
            if topic not in pre_timestamp.keys():
                pre_timestamp[topic]=0
            new_timestamp = rospy.Time.to_nsec(t)
            time_diff = new_timestamp-pre_timestamp[topic]
            pre_timestamp[topic]=new_timestamp
            #print(time_diff)
            tds[topic].append(time_diff)
                
    return tds


def test1(bag_file):
    tds=defaultdict(list)
    with rosbag.Bag(bag_file, 'r') as bag:
        msgs = bag.read_messages()
        msgs = sorted(msgs,key=lambda x: (x[1].header.stamp.secs,x[1].header.stamp.nsecs))
        for topic, msg, t in msgs:
            timestamp = msg.header.stamp
            #print(f'{type(timestamp)}')
            #print(f'{timestamp.secs}.{timestamp.nsecs}')
            
            tds[topic].append(timestamp)
    
    idx=100
    fk = '/cameras/front/image_color/compressed'
    print(f'{tds[fk][0]}  {tds[fk][-1]}')
    print(f'{msgs[0][1].header.stamp}           {msgs[-1][1].header.stamp}')
    print(f'{msgs[0][-1]}                    {msgs[-1][-1]}         {msgs[-1][-1]}')
    for k in tds.keys():
        #print(f'{k}:[{tds[fk][idx]}][{tds[k][idx]}]')
        #print(f'{k}:[{tds[fk][idx]}][{tds[k][idx]}] {tds[k][idx]-tds[fk][idx]}')
        print(f'{k} \t\t [0]{tds[k][0].secs}.{tds[k][0].nsecs} \t [-1]{tds[k][-1].secs}.{tds[k][-1].nsecs}')

if __name__ =='__main__':
    # dir = '/media/mokou/34E9-1E36/parkinglot2/raw_data'
    # out = '/home/mokou/ext_1_02'
    # if not os.path.exists(out):
    #     os.makedirs(out)
    # #print(os.listdir(dir))
    # fns = os.listdir(dir)
    # fn =fns[0]
    # print(fn)
    # # extract_images_from_rosbag(os.path.join(dir,fn),out)
    # bag_file = os.path.join(dir,fn)

    bag_file = '/home/hqlab/workspace/dataset/parkinglot/raw_data/circle_expose.bag'
    #bag_file='/media/mokou/34E9-1E36/parkinglot2/raw_data/2025-01-02-20-03-58.bag'
    #bag_file='/media/mokou/34E9-1E36/parkinglot2/raw_data/cycle.bag'
    # bag_file='/media/mokou/34E9-1E36/parkinglot2/raw_data/10hz_cycle.bag'
    # bag_file='/media/mokou/b9795ed4-8132-4cfb-8048-e3705c71f80b/home/hqlab/byd_dataset_ws/10hz_sync.bag'
    #bag_file='/media/mokou/34E9-1E36/parkinglot2/raw_data/10hz_cycle.bag'
    #bag_file

    match2front_camera(bag_file)

    # tds = td_with_prev(bag_file)
    # for key in tds.keys():
    #     print(f'avg td: {np.mean(tds[key][1:])}')
    #test1(bag_file)

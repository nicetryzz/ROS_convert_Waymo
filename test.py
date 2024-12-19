import pickle

with open('/home/hqlab/workspace/dataset/Waymo/demo/segment-10061305430875486848_1080_000_1100_000_with_camera_labels/scenario.pt', 'rb') as f:
    data1 = pickle.load(f)

with open('/home/hqlab/workspace/dataset/parkinglot/data/10_26/scenario.pt', 'rb') as f:
    data2 = pickle.load(f)

print(data1)
print(data2)


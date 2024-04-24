import numpy as np

# pls enter path
data = np.load('')
num_frames = len(data)

# enter target length
target_num_frames = 7080

# Calculate the number of missing frames
num_missing_frames = target_num_frames - num_frames
interval = max(1, len(data) // num_missing_frames)
start = 0

for i in range(num_missing_frames):

    if i == 0:
        data_inter = data[start:interval]
        data_inter = np.append(data_inter, data[interval-1:interval], axis=0)
    else:
        start = interval*i
        i += 1

        if i == num_missing_frames:
            data_inter = np.append(data_inter, data[interval * i - 1:interval * i], axis=0)
            data_inter = np.append(data_inter, data[start:len(data)], axis=0)
            break

        data_inter = np.append(data_inter, data[start:interval*i], axis=0)
        data_inter = np.append(data_inter, data[interval*i-1:interval*i], axis=0)

print(data_inter.shape)  # (length, keypoints, 3d_data)

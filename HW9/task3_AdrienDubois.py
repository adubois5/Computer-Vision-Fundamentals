# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# %%
def open_image_in_grayscale(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

# %%
task3_img_dir = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW9/Task3Images/"

left_img_path = task3_img_dir+"im2.png"
right_img_path = task3_img_dir + "im6.png"

accuracy_maps = []
for M in [5, 9, 16, 25]:
    accuracy_map = []
    for d_max in [10, 30, 54, 70]:
        # Maximum disparity from original image is 54
        window_shape = (M,M)

        left_img = open_image_in_grayscale(left_img_path)
        right_img = open_image_in_grayscale(right_img_path)

        # Pad the left and right images such that the windows are created for all image pixels
        left_img = np.pad(left_img, M)
        right_img = np.pad(right_img, M)

        # Create windows from the images. This outputs a shape of (H, W, M, M) for an MxM window centered on each pixel
        left_windows = np.lib.stride_tricks.sliding_window_view(left_img, window_shape)
        right_windows = np.lib.stride_tricks.sliding_window_view(right_img, window_shape)

        # Get the center pixels for each window:
        left_center_pixels = left_windows[:, :, M//2, M//2][..., np.newaxis, np.newaxis]
        right_center_pixels = right_windows[:, :, M//2, M//2][..., np.newaxis, np.newaxis]

        # Create new windows with only the center pixel values:
        left_center_windows = np.ones_like(left_windows) * left_center_pixels
        right_center_windows = np.ones_like(right_windows) * right_center_pixels

        # For each entry in both windows, check if the entry in the img window is strictly larger than the center pixel value
        # If strictly greater than, put in a 1, otherwise put in a 0 (right now these are True and False values)
        # I do this for entries from (x-d,y) to (x,y)
        disparity_map = np.zeros_like(left_img)

        for row_idx in range(left_windows.shape[0]):
            for col_idx in range(left_windows.shape[1]):
                if col_idx > d_max:
                    left_window = left_windows[row_idx, col_idx, ...]
                    left_center_pixel_window = left_center_pixels[row_idx, col_idx, ...]
                    bitvec_sums = []
                    for d in range(d_max, 0, -1):
                        right_window = right_windows[row_idx, col_idx-d]
                        right_center_pixel_window = right_center_pixels[row_idx, col_idx-d, ...]
                        
                        # Now I can compare the two windows
                        left_bitvec_comparator = left_window > left_center_pixel_window
                        right_bitvec_comparator = right_window > right_center_pixel_window

                        # XOR the two comparators. This is treated as a bitwise XOR since the bits are represented as logical True/False values
                        # This results in a list of windows: (M, M)
                        xor_bitvec = np.logical_xor(left_bitvec_comparator, right_bitvec_comparator)
                        # I can then get the sum of the bitvector. This is the data cost
                        bitvec_sums.append(np.sum(xor_bitvec))
                        
                    # Calculate the disparity value:
                    disparity = d_max - np.argmin(bitvec_sums)
                    # Fill in that location in the disparity map:
                    disparity_map[row_idx, col_idx] = disparity
        
        ground_truth_dmap = open_image_in_grayscale(task3_img_dir+"disp2.png")
        # Lower the resolution to match out images:
        ground_truth_dmap = ground_truth_dmap.astype(np.float32) / 4
        ground_truth_dmap = ground_truth_dmap.astype(np.uint8)
        # Pad the ground_truth_dmap as needed:
        ground_truth_dmap = np.pad(ground_truth_dmap, M)

        # Calculate the error (H, W), we only want to compute for non-black (padded) pixels
        non_black_mask = ground_truth_dmap > 0
        diff = np.abs(ground_truth_dmap - disparity_map)
        # We also allow for an error of a few pixels between the ground truth and the computed disparity map
        accuracy_value = np.count_nonzero(diff[non_black_mask] < 3) / np.count_nonzero(non_black_mask)
        accuracy_map.append(accuracy_value)
        
        accuracy_printout = np.zeros_like(diff)
        accuracy_printout[non_black_mask] = (diff[non_black_mask] < 3) * 255
        
        plt.imshow(accuracy_printout, cmap="gray")
        plt.axis("off")
        plt.show()
        plt.close()
        
        print(f"M: {M}, Dmax: {d_max}, accuracy: {np.round(accuracy_value * 100, 4)}")
    accuracy_maps.append(accuracy_map)

# %%
np_accuracy_maps = np.array(accuracy_maps)
with open("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW9/accuracy_maps.pkl", "wb") as file:
    pickle.dump(np_accuracy_maps, file)

# %%
plt.bar(range(5, 30), np_accuracy_maps[:, 3]*100)
plt.show()

# %%
plt.imshow(disparity_map, cmap="gray")
plt.axis("off")
plt.show()

# %%
ground_truth_dmap = open_image_in_grayscale(task3_img_dir+"disp2.png")
# Lower the resolution to match out images:
ground_truth_dmap = ground_truth_dmap.astype(np.float32) / 4
ground_truth_dmap = ground_truth_dmap.astype(np.uint8)
# Pad the ground_truth_dmap as needed:
ground_truth_dmap = np.pad(ground_truth_dmap, M)

# Calculate the error, we only want to compute for non-black (padded) pixels
non_black_mask = ground_truth_dmap > 0
diff = np.abs(ground_truth_dmap - disparity_map)
# We also allow for an error of a few pixels between the ground truth and the computed disparity map
accuracy_value = np.count_nonzero(diff[non_black_mask] < 3) / np.count_nonzero(non_black_mask)

print(np.round(accuracy_value * 100, 4))



# %% [markdown]
# # Key global variables:

# %%
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from scipy.optimize import least_squares
import pickle

# %%
result_path = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW8/Results/"
imgs_dir_path = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW8/Images/"

# %% [markdown]
# # Zhang's Algorithm

# %% [markdown]
# ### Corner Detection

# %%
class Point():
    def __init__(self, x, y):
        """Defines a point using its physical space coordinates"""
        self.x = x
        self.y = y
        self.hc = self.get_hc()
    @classmethod
    def from_hc(cls, hc):
        """Defines a point from its representation in homogeneous coordinates"""
        if np.isclose(hc[2],0):
            x = hc[0]
            y = hc[1]
        else:
            x = hc[0] / hc[2]
            y = hc[1] / hc[2]
        return cls(x, y)
    def get_hc(self):
        """Returns the point in homogeneous coordinates"""
        return np.array([self.x, self.y, 1])
    def __repr__(self):
        """To string method for debugging"""
        return f"Point(x={self.x}, y={self.y}, hc={self.hc})"

# %%
class Line():
    def __init__(self, point1, point2):
        """Defines a line that passes through 2 points in the physical space"""
        assert isinstance(point1, Point) and isinstance(point2, Point), "A line should be created by 2 Points, or by its angle to the x-axis"
        self.hc = self.get_hc(point1, point2)
    @classmethod
    def from_angle_to_x_axis(cls, angle, y_int):
        """Defines a line by its angle to the x-axis and y intercept"""
        intercept = Point(0, y_int) # Point1 is defined by the y-intercept
        point2 = Point(1, y_int + math.tan(math.radians(angle))) # Point 2 is the point on the line at x=1
        return cls(intercept, point2)
    @classmethod
    def from_slope_and_intercept(cls, slope, y_int):
        # Define a line in homogenous coordinates from its slope and y_intercept
        intercept = Point(0, y_int)
        point2 = Point(1, y_int + slope)
        return cls(intercept, point2)
    def get_hc(self, point1, point2):
        """Returns the line in homogeneous coordinates"""
        line = np.cross(point1.hc, point2.hc)
        return line / (line[2] + 1e-6)
    def get_intersection(self, line2):
        intersection_point = np.cross(self.hc, line2.hc)
        return Point.from_hc(intersection_point)
    def __str__(self):
        return f"Line(algebraic={self.hc[1]} * y = {-self.hc[0]} * x + {-self.hc[2]}, hc={self.hc})"

# %%
class Homography():
    def __init__(self):
        pass
    def get_first_six_cols(self, x_points):
        zero_vec = np.zeros(3)
        eqn1_first6cols = np.vstack([np.hstack((point.hc, zero_vec)) for point in x_points])
        eqn2_first6cols = np.vstack([np.hstack((zero_vec, point.hc)) for point in x_points])
        return np.vstack([[eqn1, eqn2] for eqn1, eqn2 in zip(eqn1_first6cols, eqn2_first6cols)]) # Stack the rows in an interleaved fashion
        
    def estimate_projective_homography(self, x_points, x_prime_points):
        # Lets build the first 6 columns of the matrix we are interested in:
        first6cols = self.get_first_six_cols(x_points)
        
        # Now we only need the last 2 columns
        # x' is in the shape: x1' y1' 1, x2' y2' 1 etc
        # x is: x1 y1 1, x2 y2 1, ...
        # So we take the outer product of the first two elements in the homogeneous form to get the remaining portion:
        last2cols_list = []
        for x, x_prime in zip(x_prime_points, x_points):
            last2cols_list.append(-np.outer(x.get_hc()[:2], x_prime.get_hc()[:2]))
        last2cols = np.vstack(last2cols_list)
        full_matrix = np.hstack((first6cols, last2cols))
        
        x_prime_matrix = np.vstack([np.vstack((x_prime.hc[0], x_prime.hc[1])) for x_prime in x_prime_points])
        H = np.linalg.pinv(full_matrix) @ x_prime_matrix
        H = np.vstack((H, np.array(1))).reshape((3,3))

        return H

# %%
def get_most_dissimilar_lines(hp):
    """Finds the 10 most dissimilar horizontal lines and 8 most dissimilar vertical lines to remove noise in the output of the hough transform.

    Args:
        hp (np.array): (N, 4) array of lines in the form (x1,y1,x2,y2)
    """
    # Calculate angle of the lines with the y axis:
    angles = np.degrees(np.arctan2((hp[:, 2] - hp[:, 0]), (hp[:, 3] - hp[:, 1])))
        
    horizontal_mask = np.logical_and(np.abs(angles) > 45, np.abs(angles) < 135)
    vertical_mask = np.logical_not(horizontal_mask)
    
    # Remove noise from the horizontal lines, we do this through the y-intercept
    if np.any(horizontal_mask):
        horizontal_lines = hp[horizontal_mask]
        
        # I first find the slope and intersept of each line and join them into one array
        # Since we want the clustering to mainly occur due to differences in the y-int, it is fine to not normalize these value
        # Slopes ~= 0, y-ints = [100, 600] so this scaling issue would be important for normal clustering
        slopes = (horizontal_lines[:, 3] - horizontal_lines[:, 1]) / (horizontal_lines[:, 2] - horizontal_lines[:, 0] + 1e-10)  # avoid division by zero
        y_intercepts = horizontal_lines[:, 1] - slopes * horizontal_lines[:, 0]
        hline_params = np.vstack((slopes, y_intercepts)).transpose(1, 0)
        
        # Apply kmeans on these lines and return the average slope/y-intercept for each cluster. This joins noisy line pairs together
        hkm = KMeans(n_clusters=10, random_state=0, n_init='auto')
        hkm.fit(hline_params)
        hlines_avg = hkm.cluster_centers_
        
    # Remove noise from the vertical lines, we do this through the x-intercept:
    if np.any(vertical_mask):
        vertical_lines = hp[vertical_mask]
        
        slopes = (vertical_lines[:, 3] - vertical_lines[:, 1]) / (vertical_lines[:, 2] - vertical_lines[:, 0] + 1e-10)  # avoid division by zero
        y_intercepts = vertical_lines[:, 1] - slopes * vertical_lines[:, 0]
        x_intercepts = -y_intercepts/slopes

        # Perform KMeans clustering on the x-intercepts:
        x_intercepts_2d = x_intercepts.reshape(-1, 1)
        vkm = KMeans(n_clusters=10, random_state=0, n_init='auto')
        vkm.fit(x_intercepts_2d)
        labels = vkm.labels_

        # Calculate average slope and y_intercept for each cluster:
        avg_slopes = []
        avg_y_intercepts = []

        # Loop over each cluster index (0 to 7 for 8 clusters)
        for cluster_id in range(8):
            # Get the indices of the lines in the cluster
            indices = np.where(labels == cluster_id)[0]
            
            # Compute average slope and y-intercept
            avg_slope = slopes[indices].mean()
            avg_y_intercept = y_intercepts[indices].mean()
            
            # Append the averages to the lists
            avg_slopes.append(avg_slope)
            avg_y_intercepts.append(avg_y_intercept)
        vlines_avg = np.stack((avg_slopes, avg_y_intercepts)).transpose(1, 0)
    
    return hlines_avg, vlines_avg
    

# %%
def open_image_in_black_and_white(img_path):
    # Open the image:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to pure black and white:
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    return img
    
def get_sorted_lines(hlines, vlines):
    # Sort the horizontal lines in ascending order by y_intercept:
    hlines = hlines[np.argsort(hlines[:, 1])]
    
    # Sort the vertical lines in ascending order by x_intercept:
    x_intercepts = []
    for (slope, y_intercept) in vlines:
        x_intercept = -y_intercept / slope if slope != 0 else np.inf
        x_intercepts.append(x_intercept)
    x_intercepts = np.array(x_intercepts)
    vlines = vlines[np.argsort(x_intercepts)]
    
    return hlines, vlines

def get_real_intersection_coordinates(real_world_scaling):
    real_world_points = []
    for row in range(10):
        for col in range(8):
            real_world_points.append(Point(real_world_scaling*col, real_world_scaling*row))
    return real_world_points

# %%
def get_image_homography(img_list, real_world_scaling):
    homography_list = []
    all_intersection_points = []
    all_real_world_points = []
    for img_full_name in img_list:
        img_path = imgs_dir_path + img_full_name
        
        # Open the image thresholded to be black and white
        img = open_image_in_black_and_white(img_path)
        
        # Run canny operator for corner detection on the image
        edges = cv2.Canny(image=img, threshold1=40, threshold2=500)
        
        # Calculate the hough points
        hp = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=50, maxLineGap=100).squeeze()
        
        # Get the slope,y_intercept for the best 18 lines (removing noise)
        hlines, vlines = get_most_dissimilar_lines(hp)
        
        # Get the sorted lines:
        hlines, vlines = get_sorted_lines(hlines, vlines)
        
        # Convert the lines to homogenous coordinates:
        intersection_points = []
        for hline in hlines:
            for vline in vlines:
                hhline = Line.from_slope_and_intercept(hline[0], hline[1])
                hvline = Line.from_slope_and_intercept(vline[0], vline[1])

                intersection_points.append(hhline.get_intersection(hvline))
        all_intersection_points.append([point.hc for point in intersection_points])
        
        # Create world_coordinates
        real_world_points = get_real_intersection_coordinates(real_world_scaling=real_world_scaling)
        all_real_world_points.append([point.hc for point in real_world_points])
        
        # detected = H * real_world_points
        homography_list.append(Homography().estimate_projective_homography(x_points=real_world_points, x_prime_points=intersection_points))
        
    return homography_list, np.array(all_intersection_points), np.array(all_real_world_points)

# %%
# Code to estimate omega
def get_V_ij_matrix(h, i, j):
    """ Computes V_ij from the 3by3 homography.

    Args:
        H (np.array): (3, 3) homography from intersection coordinates to real world points there
        i (int): row index
        j (int): col index
    """
    # Calculate the V_ij matrix from Avi's lecture 21 as described in Zhang's method
    V_ij = np.array([h[0][i] * h[0][j],
                     h[0][i] * h[1][j] + h[1][i] * h[0][j],
                     h[1][i] * h[1][j],
                     h[2][i] * h[0][j] + h[0][i] * h[2][j],
                     h[2][i] * h[1][j] + h[1][i] * h[2][j],
                     h[2][i] * h[2][j]])
    return V_ij

def get_complete_V_matrix(homography_list):
    V_matrix = []
    for homography in homography_list:
        v_11 = get_V_ij_matrix(homography, 0, 0)
        v_12 = get_V_ij_matrix(homography, 0, 1)
        v_22 = get_V_ij_matrix(homography, 1, 1)

        V_matrix.append(v_12.T)
        V_matrix.append((v_11 - v_22).T)
    v_mat = np.array(V_matrix)
    return v_mat

def get_omega_parameters(v_mat):
    # Perform Singular Value Decomposition
    _, _, Vt = np.linalg.svd(v_mat)
    
    # The null space is the last row of Vt
    omega = Vt[-1]
    return omega

# %%
# Code for camera specific parameters:
def get_K_matrix(omega):
    # Omega is: [w11, w12, w22, w13, w23, w33]
    w11, w12, w22, w13, w23, w33 = omega[0], omega[1], omega[2], omega[3], omega[4], omega[5]
    
    # Calculate the parameters of K using omega following Avi's lecture 21 page 3
    y_0                 = (w12*w13 - w11*w23) / (w11*w22 - w12**2)
    lambda_param        = w33 - (( w13**2 + y_0*(w12*w13 - w11*w23) ) / w11)
    alpha_x             = np.sqrt(lambda_param / w11)
    alpha_y             = np.sqrt((lambda_param*w11) / (w11*w22 - w12**2))
    s                   = -((w12 * alpha_x**2 * alpha_y)/lambda_param)
    x_0                 = ((s*y_0)/alpha_y) - ((w13 * alpha_x**2)/lambda_param)
    
    # Reconstruct K:
    K_mat = np.array([[alpha_x, s, x_0],
                      [0, alpha_y, y_0],
                      [0, 0, 1]])
    return K_mat

def get_r_and_t_mats_from_K_and_H(K_mat, homography_list):
    K_inv = np.linalg.inv(K_mat)
    
    R_mats = []
    t_mats = []
    for homography in homography_list:
        # Get the columns of the homography:
        h1, h2, h3 = homography[:, 0], homography[:, 1], homography[:, 2]
        
        # We need to rescale the rotation matrix to make it othernormal:
        scaling = 1 / np.linalg.norm(K_inv @ h1)
        r1 = scaling * K_inv @ h1
        r2 = scaling * K_inv @ h2
        r3 = np.cross(r1, r2)
        t = scaling * K_inv @ h3
        
        # Next, I have to orthonomalize R by getting R = UDVt and setting R to UVt
        # R is made from its three column vector components
        R = np.vstack([r1, r2, r3]).T
        U, _, Vt = np.linalg.svd(R)
        R_mats.append(U @ Vt)
        
        t_mats.append(t)
        
    return np.array(R_mats), np.array(t_mats)


# %%
# LM relevant code:
def get_lm_learnable_params(R_mats, t_mats, K_mat):
    # Create the w_matrix from R, where w only has 3 DoF
    w_mats = []
    for R in R_mats:
        phi = np.arccos(( np.trace(R) - 1 ) / 2)
        w = (phi / 2 * np.sin(phi)) * np.array([[R[2][1] - R[1][2]],
                                                [R[0][2] - R[2][0]],
                                                [R[1][0] - R[0][1]]])
        w_mats.append(w)
        
    k_params = np.array([K_mat[0, 0], K_mat[0, 1], K_mat[0, 2], K_mat[1,1], K_mat[1, 2]])
    
    return np.array(np.concatenate((np.array(w_mats).flatten(), t_mats.flatten(), k_params.flatten())))


def compute_R_mats_from_w(w_mats):
    R_mats_list = []
    for w_mat in w_mats:
        phi = np.linalg.norm(w_mat)
        
        sin_phi_over_phi = np.sin(phi) / phi
        one_minus_cos_phi_over_phi2 = (1 - np.cos(phi)) / (phi ** 2)
        
        wx, wy, wz = w_mat[0], w_mat[1], w_mat[2]
        
        # Calculate [w]_x matrices as per the skew-symmetric definition
        w_cross = np.zeros((3, 3))
        w_cross[0, 1] = -wz
        w_cross[0, 2] = wy
        w_cross[1, 0] = wz
        w_cross[1, 2] = -wx
        w_cross[2, 0] = -wy
        w_cross[2, 1] = wx
        
        # Identity matrix repeated for each w
        I = np.eye(3)  # Shape: (3, 3)

        # Rodrigues formula for each w in W_mats
        R_mats = I + sin_phi_over_phi * w_cross + one_minus_cos_phi_over_phi2 * np.matmul(w_cross, w_cross)
        
        R_mats_list.append(R_mats)

    return np.array(R_mats_list)


def unpack_parameters_for_lm(l_params, num_views):
    # l_params is:
    # - W_mats: 117 params, representing a list of 39 rotation matrices (3-element vectors each)
    # - t_mats: 117 params, representing a list of 39 translation vectors (3-element vectors each)
    # - K: 9 params, representing the 3x3 intrinsic matrix

    # First we extract a list of 3 element vectors for the rotation matrix from w
    w_mats = l_params[:num_views*3].reshape((-1, 3))
    
    # Next we need to regenerate the R matrices used for LM
    R_mats = compute_R_mats_from_w(w_mats)
    
    # T_mats is a list of 3 element vectors for the translation of the camera
    t_mats = l_params[num_views*3:2*num_views*3].reshape((-1, 3))
    
    # We need to make K a 3by3 array again
    K_params = l_params[-5:]
    K = np.array([[K_params[0], K_params[1], K_params[2]], 
                  [0, K_params[3], K_params[4]],
                  [0, 0, 1]])
    return R_mats, t_mats, K

def cost_func(l_params, real_world_points, intersection_points, num_views):
    # First we extract the camera parameters from the learnable parameter array
    R_mats, t_mats, K = unpack_parameters_for_lm(l_params, num_views)
    projection_mat = K @ np.array([R_mats[..., 0], R_mats[..., 1], t_mats]).transpose(1, 0, 2)

    # Expand the arrays for broadcasting
    proj_mat_expanded = projection_mat[:, np.newaxis, ...] # (39, 3, 3) to (39, 1, 3, 3)
    real_points_expanded = real_world_points[..., np.newaxis] # (39, 80, 3) to (39, 80, 3, 1)

    # Calculate the projected points:
    projected_points = (proj_mat_expanded @ real_points_expanded).squeeze(-1)

    # Normalize by third_coord
    projected_points = projected_points / projected_points[..., 2][..., np.newaxis]
    
    # Calculate the residuals:
    residuals = intersection_points[..., 0:2] - projected_points[..., 0:2]

    return residuals.ravel()

# %%
# imgs_dir_path = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW8/HW8-Files/Dataset1/"
imgs_dir_path = "/mnt/cloudNAS3/Adubois/Classes/ECE661/HW8/Images/"
img_list = os.listdir(imgs_dir_path)
img_list = img_list[0:12]
num_views = len(img_list)
homography_list, intersection_points, real_world_points = get_image_homography(img_list, real_world_scaling=2.5)

# Calculate the V matrix:
v_mat = get_complete_V_matrix(homography_list)

# Solve for the omega paramters
omega = get_omega_parameters(v_mat)

# Find K matrix
K_mat = get_K_matrix(omega)

# Find R and T initial solution
R_mats, t_mats = get_r_and_t_mats_from_K_and_H(K_mat, homography_list)

# Get Levangerg Marquadt improved matrices
learnable_params = get_lm_learnable_params(R_mats, t_mats, K_mat)
optimal_params = least_squares(cost_func, learnable_params, args=(real_world_points, intersection_points, num_views), method='lm', verbose=2)
optim_R, optim_t, optim_K = unpack_parameters_for_lm(optimal_params.x, num_views)


# %%
img_list

# %%
result_dict = {"intersection_points": intersection_points,
               "real_world_points": real_world_points,
                "K_mat": K_mat,
               "R_mats": R_mats,
               "t_mats": t_mats,
               "optim_R": optim_R,
               "optim_t": optim_t,
               "optim_K": optim_K}
with open("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW8/images_lm_2.pkl", "wb") as file:
    pickle.dump(result_dict, file)

# %%
projection_mat = optim_K @ np.array([optim_R[..., 0], optim_R[..., 1], optim_t]).transpose(1, 0, 2)

# Expand the arrays for broadcasting
proj_mat_expanded = projection_mat[:, np.newaxis, ...] # (39, 3, 3) to (39, 1, 3, 3)
real_points_expanded = real_world_points[..., np.newaxis] # (39, 80, 3) to (39, 80, 3, 1)

# Calculate the projected points:
projected_points = (proj_mat_expanded @ real_points_expanded).squeeze(-1)

for img_name, image_points, img_proj_points in zip(img_list, intersection_points, projected_points):
    # Regenerate the image background:
    img_path = imgs_dir_path + img_name
    # Open the image thresholded to be black and white
    img = open_image_in_black_and_white(img_path)
    
    for point, proj_point in zip(image_points, img_proj_points):
        x, y, _ = point
        proj_point = proj_point / proj_point[2]
        x2, y2, _ = proj_point
        
        plt.plot(x, y, "bo", markersize=2)
        plt.plot(x2, y2, "ro", markersize=2)
    
    plt.imshow(img, 'gray')
    plt.axis("off")
    plt.show()
    plt.close()

# %%
projection_mat = K_mat @ np.array([R_mats[..., 0], R_mats[..., 1], t_mats]).transpose(1, 0, 2)

# Expand the arrays for broadcasting
proj_mat_expanded = projection_mat[:, np.newaxis, ...] # (39, 3, 3) to (39, 1, 3, 3)
real_points_expanded = real_world_points[..., np.newaxis] # (39, 80, 3) to (39, 80, 3, 1)

# Calculate the projected points:
projected_points = (proj_mat_expanded @ real_points_expanded).squeeze(-1)

for img_name, image_points, img_proj_points in zip(img_list, intersection_points, projected_points):
    # Regenerate the image background:
    img_path = imgs_dir_path + img_name
    # Open the image thresholded to be black and white
    img = open_image_in_black_and_white(img_path)
    
    for point, proj_point in zip(image_points, img_proj_points):
        x, y, _ = point
        proj_point = proj_point / proj_point[2]
        y2, x2, _ = proj_point
        
        plt.plot(x, y, "bo", markersize=2)
        plt.plot(x2, y2, "ro", markersize=2)
    
    plt.imshow(img, 'gray')
    plt.axis("off")
    plt.show()
    plt.close()

# %%
C_mats = []
X_mats = []
for R_img, t_img in zip(optim_R, optim_t):
    C = - R_img.T @ t_img
    C = C / C[2] / 10
    C_mats.append(C)
    X_cam = np.eye(3)
    
    X = R_img.T @ X_cam + C
    X = X / X[2] / 10
    X_mats.append(X)

# %%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the calibration pattern image
calibration_pattern = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW8/HW8-Files/calibration_pattern.png")  # Replace with actual path

# Define camera axis colors and settings
axis_colors = {'X': 'red', 'Y': 'green', 'Z': 'blue'}
camera_count = len(C)  # Assuming `C` and `X` are lists of centers and axes for each camera pose

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Plot camera axes and principal planes
for i in range(camera_count):
    # Camera center and axis for each pose
    center = C_mats[i]
    axes = X_mats[i]
    
    # Plot the camera axes
    for j, (axis_name, color) in enumerate(axis_colors.items()):
        ax.quiver(center[0], center[1], center[2], 
                  axes[j, 0], axes[j, 1], axes[j, 2], color=color, length=0.1)

    # Plot the principal plane as a semi-transparent 3D rectangle
    principle_plane = [center + 0.1 * axes[0], center + 0.1 * axes[1], 
                    center - 0.1 * axes[0], center - 0.1 * axes[1]]
    ax.add_collection3d(Poly3DCollection([principle_plane], color=np.random.rand(3,), alpha=0.3))

# Set the desired dimensions of the image on the plot

img_width = 0.2
img_height = 0.2

# Create the X, Y, Z grid to place the image at Z=0
x_img = np.linspace(-img_width / 2, img_width / 2, calibration_pattern.shape[1])
y_img = np.linspace(-img_height / 2, img_height / 2, calibration_pattern.shape[0])

x_img, y_img = np.meshgrid(x_img, y_img)
z_img = np.zeros_like(x_img)  # Z=0 plane

ax.plot_surface(x_img, y_img, z_img, rstride=5, cstride=5, facecolors=calibration_pattern / 255, shade=False)
# Set axis limits and labels
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([0, 0.5])
ax.set_xlabel('X_cam')
ax.set_ylabel('Y_cam')
ax.set_zlabel('Z_cam')

plt.show()


# %%




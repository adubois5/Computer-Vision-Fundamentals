# %%
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

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
        eqn1_first6cols = np.vstack([np.hstack((point.get_hc(), zero_vec)) for point in x_points])
        eqn2_first6cols = np.vstack([np.hstack((zero_vec, point.get_hc())) for point in x_points])
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
        
        x_prime_matrix = np.vstack([np.vstack((x_prime.get_hc()[0], x_prime.get_hc()[1])) for x_prime in x_prime_points])
        H = np.matmul(np.linalg.inv(full_matrix), x_prime_matrix)
        H = np.vstack((H, np.array(1))).reshape((3,3))
        return H
    
    def estimate_affine_homography(self, x_points, x_prime_points):
        # ONly 3 points are needed
        x_points = x_points[:3]
        x_prime_points = x_prime_points[:3]
        
        # Get first 6 columns similar to the general projective homography
        # However, in the case of an affine matrix, we only need those 6 columns
        full_matrix = self.get_first_six_cols(x_points)
        
        # Create x' matrix: [x_1', y_1', x_2', y_2', ...]
        x_prime_matrix = np.vstack([np.vstack((x_prime.get_hc()[0], x_prime.get_hc()[1])) for x_prime in x_prime_points])
        
        H = np.matmul(np.linalg.inv(full_matrix), x_prime_matrix)
        last_row = np.array((0, 0, 1)).reshape(-1, 1)
        H = np.vstack((H, last_row)).reshape((3,3))
        return H

    # Used for the bonus. Explanations of the code are included in my report
    def get_normalizing_homography(self, image):
        h, w = image.shape[:2]
        print(h, w)
        H_norm = [2/w, 0, -1, 0, 2/h, -1, 0, 0, 1]
        H_norm = np.array(H_norm, dtype=np.float32).reshape((3,3))
        return H_norm
    def get_denormalizing_homography(self, image):
        h, w = image.shape[:2]
        H_norm = [w/2, 0, w/2, 0, h/2, h/2, 0, 0, 1]
        H_norm = np.array(H_norm, dtype=np.float32).reshape((3,3))
        return H_norm
    def get_rotational_homography(self, alpha):
        H_rot = [math.cos(alpha), -math.sin(alpha), 0, math.sin(alpha), math.cos(alpha), 0, 0, 0, 1]
        H_rot = np.array(H_rot, dtype=np.float32).reshape((3,3))
        return H_rot
    def get_horizontal_tilt_homography(self, alpha):
        H = [1,0,0,0,1,0, math.cos(alpha),0,1]
        H = np.array(H, dtype=np.float32).reshape((3,3))
        return H
    def get_vertical_tilt_homography(self, alpha):
        H = [1,0,0,0,1,0,0,math.sin(alpha), 1]
        H = np.array(H, dtype=np.float32).reshape((3,3))
        return H    

# %%
def reverseMapHomography(source_img, homography, target_size):
    new_image = np.zeros((target_size[0], target_size[1], source_img.shape[2]), dtype=source_img.dtype)
    H_inv = np.linalg.pinv(homography)
    
    # Create pixel coordinate list for physical space -> HC conversion
    x_range = np.arange(target_size[0])
    y_range = np.arange(target_size[1])
    x_1, x_2 = np.meshgrid(x_range, y_range)
    real_coords = np.vstack([x_2.ravel(), x_1.ravel()]).T
    x_3 = np.ones((real_coords.shape[0],1))
    hc_coord = np.hstack((real_coords, x_3))
    
    # Get normalized HC coordinates
    src_hc_coords = np.matmul(H_inv, hc_coord.T).T
    norm_mask = ~np.isclose(src_hc_coords[:, 2], 0)
    src_hc_coords[norm_mask] = src_hc_coords[norm_mask] / src_hc_coords[norm_mask, 2][:, np.newaxis]

    # Convert to integer coordinates for image coordinates
    src_x = src_hc_coords[:, 0].astype(int)
    src_y = src_hc_coords[:, 1].astype(int)
    
    
    # Check pixel locations vs bounds of the images
    valid_mask = (src_x >= 0) & (src_x < source_img.shape[1]) & (src_y >= 0) & (src_y < source_img.shape[0])
    valid_x = src_x[valid_mask]
    valid_y = src_y[valid_mask]
    
    # Map valid coordinates from source image to the new image
    new_image[x_1.flatten()[valid_mask], x_2.flatten()[valid_mask]] = source_img[valid_y, valid_x]

    return new_image

# %%
def map_image1_onto_roi_on_image2_inrgb(source_img, target_img, target_roi, homography):
    # This function is used to map the photo of Alex onto the ROI in the frame-photo
    # Warp the alex image following the previously calculated homography
    warped_img = reverseMapHomography(source_img, homography, target_img.shape)
    
    # Create a mask of the polygon created by the image frame that I want to cover
    # I will keep the area outside of the mask
    # # Note: cv2 polygons only accept integer pixels:
    polygon_coords = np.array([[int(x), int(y)] for x,y in target_roi], dtype=np.int32)
    mask = np.zeros_like(target_img, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_coords], (255, 255, 255))

    # Remove the frame portion from the image
    mask_inv = cv2.bitwise_not(mask)
    img_background = cv2.bitwise_and(target_img, mask_inv)

    # Add the Alex photo in with projective transformation warping in
    final_image = cv2.add(img_background, cv2.bitwise_and(warped_img, mask))

    # Convert the image to RGB for matplotlib
    final_rgb_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    return final_rgb_image


# %%
def get_scaling_and_translation_homography(corner_points, homography, target_shape):
    # Create the matrix of corner points
    corner_mat = np.zeros((3, 4))
    for i in range(4):
        corner_mat[:, i] = corner_points[i].hc
    
    # Apply the homography following C' = HC
    C_prime = homography @ corner_mat
    
    # C_pn = normalize_corner_points(C_prime)
    C_pn = C_prime / (C_prime[2] + 1e-6)
    y_min, x_min, y_max, x_max = np.min(C_pn[0]), np.min(C_pn[1]), np.max(C_pn[0]), np.max(C_pn[1])
    
    w_1 = x_max-x_min
    h_1 = y_max-y_min
    H_scale = np.array([[target_shape[0]/w_1, 0, 0], [0, target_shape[1]/h_1, 0], [0,0,1]])

    H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    return H_translate @ H_scale

# %% [markdown]
# Point measurements for given images

# %%
board_img = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/board_1.jpeg")
corridor_img = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/corridor.jpeg")
window_img = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/window.jpg")
kallax_img = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/kallax.jpg")

# %%
# Points(y,x) are in order:
# P Q
# R S
# corridor = [(816.4353052498776, 578.3866615265997), (1693.0316464568582, 419.4184131211887), (1671.8358800028036, 1498.8885189598373), (842.1730216583727, 1064.3753066517138)]
corridor = [(532.923076923077, 547.8076923076924), (1700.5314685314688, 100.70279720279723), (1673.3426573426577, 1497.9055944055945), (552.5594405594405, 913.3461538461539)]
corridor_true = [(0,0), (corridor_img.shape[1], 0), (corridor_img.shape[1], corridor_img.shape[0]), (0, corridor_img.shape[0])]

board = [(73.64820002801525, 425.5899285614232), (1221.3719708642668, 139.16325815940627), (1356.5169491525423, 1954.5435635243032), (422.6046365037121, 1795.1935144978288)]
board_true = [(0,0), (board_img.shape[1], 0), (board_img.shape[1], board_img.shape[0]), (0, board_img.shape[0])]

kallax = [(379.79478214093547, 2102.9276492738036), (2724.281065088757, 1044.2939752555144), (2543.5387305002687, 2164.8964497041425), (818.7404518558362, 3626.3273265196344)]
kallax_true = [(0,0), (kallax_img.shape[1], 0), (kallax_img.shape[1], kallax_img.shape[0]), (0, kallax_img.shape[0])]

window = [(783.1886163689683, 180.8532391362305), (3182.548786990136, 944.8095174620103), (3247.8113836310317, 2046.5957078112503), (744.7988536390299, 2426.6543588376435)]
window_true = [(0,0), (window_img.shape[1], 0), (window_img.shape[1], window_img.shape[0]), (0, window_img.shape[0])]

# %%
corridor_points = [Point(coordinate[0], coordinate[1]) for coordinate in corridor]
corridor_true_points = [Point(coordinate[0], coordinate[1]) for coordinate in corridor_true]
board_points = [Point(coordinate[0], coordinate[1]) for coordinate in board]
board_true_points = [Point(coordinate[0], coordinate[1]) for coordinate in board_true]
kallax_points = [Point(coordinate[0], coordinate[1]) for coordinate in kallax]
kallax_true_points = [Point(coordinate[0], coordinate[1]) for coordinate in kallax_true]
window_points = [Point(coordinate[0], coordinate[1]) for coordinate in window]
window_true_points = [Point(coordinate[0], coordinate[1]) for coordinate in window_true]

# %%
# img1_points = H_41 alex_points
H_corridor = Homography().estimate_projective_homography(corridor_points, corridor_true_points)
H_board = Homography().estimate_projective_homography(board_points, corridor_true_points)
H_kallax = Homography().estimate_projective_homography(kallax_points, kallax_true_points)
H_window = Homography().estimate_projective_homography(window_points, window_true_points)

# %%
# Print an example homography
print(H_corridor)

# %% [markdown]
# Sanity Check:

# %%
# Sanity check, the results of the original points, and H times the points from Alex Honnold's image should be the same
# These were manually checked for a random subset of points accross image1, 2 and 3
pointidx_to_test = 1
mapped_alex_onto_img1 = np.matmul(H_corridor, corridor_points[pointidx_to_test].get_hc())
mapped_alex_onto_img1 = mapped_alex_onto_img1/mapped_alex_onto_img1[2] # Normalize the matrix so that x_3 = 1
print(corridor_true_points[pointidx_to_test].get_hc(), mapped_alex_onto_img1)

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/board_1.jpeg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/board_1.jpeg")

target_roi = board_true
homography = H_board
H_resize = get_scaling_and_translation_homography(board_points, homography, (target.shape[1], target.shape[0]))
img = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@homography)
plt.imshow(img)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/corridor.jpeg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/corridor.jpeg")

target_roi = corridor_true
homography = H_corridor
H_resize = get_scaling_and_translation_homography(corridor_points, homography, (target.shape[1], target.shape[0]))
img = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@homography)
plt.imshow(img)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/window.jpg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/window.jpg")

target_roi = window_true
homography = H_window
H_resize = get_scaling_and_translation_homography(window_points, homography, (target.shape[0], target.shape[1]))
img = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@homography)
plt.imshow(img)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/kallax.jpg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/kallax.jpg")

target_roi = kallax_true
homography = H_kallax
H_resize = get_scaling_and_translation_homography(kallax_points, homography, (target.shape[0], target.shape[1]))
img = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@homography)
plt.imshow(img)
plt.show()

# %% [markdown]
# # Two Step approach:

# %%
vert_line_coords_board = [(422.93241350329197, 1789.1422468132791), (73.97597702759504, 427.6070177896065), (1356.8447261521223, 1954.5435635243032), (1223.71683709203, 141.18034738758934), (487.6557641126207, 1609.6213055049727), (215.34871830788614, 451.81208852780514), (1129.0901386748844, 1756.8688191623478), (957.6375542793107, 280.35950413223145), (793.8538858617717, 1712.9637223974764), (544.0116145683962, 374.965443074276), (1051.9554057929447, 1807.945081732148), (847.5390020074556, 304.761829652997)]
horiz_line_coords_board = [(149.8692307692304, 464.73076923076906), (1078.3167832167833, 255.27622377622356), (423.77132867132855, 1594.5769230769229), (1213.253846153846, 1683.1923076923076), (73.18058968058949, 425.28308178308134), (1223.5631800631795, 142.23639873639831), (426.9889434889433, 1798.0594945594944), (1356.999473499473, 1953.7351702351702), (203.86923076923063, 744.674825174825), (1108.148951048951, 621.8216783216781), (336.79230769230753, 1244.1433566433566), (1172.5965034965034, 1254.2132867132866)]

horiz_line_coords_corridor = [(532.923076923077, 547.8076923076924), (1700.5314685314688, 100.70279720279723), (1673.3426573426577, 1497.9055944055945), (552.5594405594405, 913.3461538461539), (737.1969696969696, 591.1363636363637), (1693.2575757575755, 421.439393939394), (1303.8636363636365, 1386.590909090909), (809.9242424242424, 1094.1666666666665), (1693.7582879033448, 384.4618388094889), (735.8092677176957, 580.1889641962578), (732.626712833358, 459.2518785914249), (1725.5838367467218, 59.84124060704312)]
vert_line_coords_corridor = [(807.7651515151517, 1092.651515151515), (812.3106060606062, 579.0151515151515), (918.3712121212122, 1127.5), (922.9166666666667, 562.3484848484848), (1298.3230107075376, 1340.5170060885996), (1307.392924627336, 489.4567499475122), (1674.724438379173, 1497.728847365106), (1694.3759185387364, 418.409090909091), (730.2748586631195, 1046.3517459261725), (736.979215164616, 593.8076820751583), (1372.7692878918156, 444.2982490424454), (1374.457711248339, 258.57167982490455)]

horiz_line_coords_kallax = [(528.7220741135689, 2155.6738203145833), (2714.3792322047434, 1116.5909090909095), (728.3488403092501, 2923.469074913357), (2622.243801652891, 1654.0475873100513), (851.1960810450532, 1689.8780325246607), (1793.024926686216, 1311.0990402559323), (825.6029058917616, 3629.8407091442286), (2576.1760863769646, 2145.436550253266), (820.4842708611027, 1444.1835510530532), (1762.3131165022646, 1121.7095441215683), (1398.8900293255124, 2836.4522793921624), (2361.193415089308, 2104.487470007998)]
vert_line_coords_kallax = [(815.3656358304438, 3624.7220741135698), (380.2816582244727, 2099.3688349773397), (1444.9577446014387, 3041.197680618502), (1347.7036790189272, 1771.7761930151964), (1905.6348973607023, 2652.181418288457), (1931.228072513995, 1485.132631298321), (2253.7020794454797, 2355.300586510264), (2371.430685150625, 1275.268595041323), (2550.582911223672, 2119.843375099974), (2714.3792322047434, 1121.7095441215683), (1240.212343375099, 2554.9273527059454), (1189.0259930685138, 2140.3179152226076)]

horiz_line_coords_window = [(783.1886163689683, 119.42961876832851), (2330.296054385497, 656.8862969874704), (2433.948413756332, 691.4370834444148), (3186.3877632631297, 914.0977072780593), (2483.855105305252, 2146.409090909091), (3243.972407358038, 2027.4008264462811), (763.9937350039991, 1532.172887230072), (2341.8129832044783, 1612.7913889629433), (2564.4736070381227, 2488.0779792055455), (3105.769261530258, 2338.3579045587844), (894.5189282857905, 15.777259397494163), (2192.0929085577177, 503.32724606771535)]
vert_line_coords_window = [(729.4429485470541, 2967.950013329779), (763.9937350039991, 261.4717408691017), (2506.8889629432147, 2687.7047454012263), (2410.914556118368, 821.9622767262063), (2464.6602239402823, 2895.009464142895), (2337.9740069314844, 549.3949613436416), (664.1803519061586, 2998.66182351373), (694.8921620901092, 207.72607304718758), (2295.745267928552, 2864.2976539589445), (2211.287789922687, 864.191015729139), (1220.8319114902692, 1927.5874433484405), (1205.476006398294, 503.32724606771535)]


# %%
def get_vert_horiz_lines(vert_line_coords, horiz_line_coords):
    vert_line_points = [Point(coordinate[1], coordinate[0]) for coordinate in vert_line_coords]
    horiz_line_points = [Point(coordinate[1], coordinate[0]) for coordinate in horiz_line_coords]
    
    vert_lines = [Line(vert_line_points[i], vert_line_points[i+1]) for i in range(0, len(vert_line_points) - 1, 2)]
    horiz_lines = [Line(horiz_line_points[i], horiz_line_points[i+1]) for i in range(0, len(horiz_line_points) - 1, 2)]
    
    return vert_lines, horiz_lines

# %%
vert_board_lines, horiz_board_lines = get_vert_horiz_lines(vert_line_coords_board, horiz_line_coords_board)
vert_corridor_lines, horiz_corridor_lines = get_vert_horiz_lines(vert_line_coords_corridor, horiz_line_coords_corridor)
vert_window_lines, horiz_window_lines = get_vert_horiz_lines(vert_line_coords_window, horiz_line_coords_window)
vert_kallax_lines, horiz_kallax_lines = get_vert_horiz_lines(vert_line_coords_kallax, horiz_line_coords_kallax)

# %%
def get_projective_homography_from_vanishing_line(horiz_lines, vert_lines):
    # Estimate vanihsing points
    vanishing_point1 = horiz_lines[0].get_intersection(horiz_lines[1])
    vanishing_point2 = vert_lines[0].get_intersection(vert_lines[1])
    
    # Estimate the vanishing line
    vanishing_line = Line(vanishing_point1, vanishing_point2)
    vanishing_line_hc = vanishing_line.hc / vanishing_line.hc[2]
    
    # Calculate the homography and normalize
    H = np.vstack((np.array([[1, 0, 0], [0, 1, 0]]), vanishing_line_hc))
    return H

# %%
def get_purely_affine_homography_from_dual_degen_conic(horiz_lines, vert_lines):
    A = np.zeros((len(horiz_lines), 2))
    b = np.zeros((len(horiz_lines),))
    
    # Fill in the A and b matrices
    for i, lines in enumerate(zip(horiz_lines, vert_lines)):
        hline = lines[0].hc
        vline = lines[1].hc
        
        # Normalize line coordinates
        hline = hline / hline[2]
        vline = vline / vline[2]
        
        A[i] = np.array([hline[0]*vline[0], hline[0]*vline[1] + hline[1]*vline[0]])
        b[i] = -1 * np.array([hline[1] * vline[1]])

    S = (np.linalg.inv((A.T) @ A) @ A.T) @ b
    S = np.array([[S[0], S[1]], [S[1], 1]])
    U,Sigma,V = np.linalg.svd(S)
    
    H_2by2 = V @ np.diag(np.sqrt(Sigma)) @ V.T
    H = np.vstack((np.hstack((H_2by2, np.array([[0], [0]]))), np.array([0, 0, 0])))
    return H
    

# %%
h_board_proj = get_projective_homography_from_vanishing_line(horiz_board_lines, vert_board_lines)
h_corridor_proj = get_projective_homography_from_vanishing_line(horiz_corridor_lines, vert_corridor_lines)
h_window_proj = get_projective_homography_from_vanishing_line(horiz_window_lines, vert_window_lines)
h_kallax_proj = get_projective_homography_from_vanishing_line(horiz_kallax_lines, vert_kallax_lines)

h_board_aff = get_purely_affine_homography_from_dual_degen_conic(horiz_board_lines, vert_board_lines)
h_corridor_aff = get_purely_affine_homography_from_dual_degen_conic(horiz_corridor_lines, vert_corridor_lines)
h_window_aff = get_purely_affine_homography_from_dual_degen_conic(horiz_window_lines, vert_window_lines)
h_kallax_aff = get_purely_affine_homography_from_dual_degen_conic(horiz_kallax_lines, vert_kallax_lines)


# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/board_1.jpeg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/board_1.jpeg")

target_roi = board_true
H_resize = get_scaling_and_translation_homography(board_true_points, h_board_proj, (target.shape[1], target.shape[0]))
warped_img_1 = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@h_board_proj)
plt.imshow(warped_img_1)
plt.show()
H_resize2 = get_scaling_and_translation_homography(board_true_points, h_board_aff, (target.shape[1], target.shape[0]))
warped_img_2 = map_image1_onto_roi_on_image2_inrgb(warped_img_1, target, target_roi, H_resize2@h_board_aff)
warped_img_2 = cv2.cvtColor(warped_img_2, cv2.COLOR_BGR2RGB)
plt.imshow(warped_img_2)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/corridor.jpeg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/corridor.jpeg")

target_roi = corridor_true
H_resize = get_scaling_and_translation_homography(corridor_true_points, h_corridor_proj, (target.shape[1], target.shape[0]))
warped_img_1 = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@h_corridor_proj)
plt.imshow(warped_img_1)
plt.show()
H_resize2 = get_scaling_and_translation_homography(corridor_true_points, h_corridor_aff, (target.shape[1], target.shape[0]))
warped_img_2 = map_image1_onto_roi_on_image2_inrgb(warped_img_1, target, target_roi, H_resize2@h_corridor_aff)
plt.imshow(warped_img_2)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/window.jpg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/window.jpg")

target_roi = window_true
H_resize = get_scaling_and_translation_homography(window_true_points, h_window_proj, (target.shape[1], target.shape[0]))
warped_img_1 = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@h_window_proj)
plt.imshow(warped_img_1)
plt.show()
H_resize2 = get_scaling_and_translation_homography(window_true_points, h_window_aff, (target.shape[1], target.shape[0]))
warped_img_2 = map_image1_onto_roi_on_image2_inrgb(warped_img_1, target, target_roi, H_resize2@h_window_aff)
warped_img_2 = cv2.cvtColor(warped_img_2, cv2.COLOR_BGR2RGB)
plt.imshow(warped_img_2)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/kallax.jpg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/kallax.jpg")

target_roi = kallax_true
H_resize = get_scaling_and_translation_homography(kallax_true_points, h_kallax_proj, (target.shape[1], target.shape[0]))
warped_img_1 = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@h_kallax_proj)
plt.imshow(warped_img_1)
plt.show()
H_resize2 = get_scaling_and_translation_homography(kallax_true_points, h_kallax_aff, (target.shape[1], target.shape[0]))
warped_img_2 = map_image1_onto_roi_on_image2_inrgb(warped_img_1, target, target_roi, H_resize2@h_kallax_aff)
warped_img_2 = cv2.cvtColor(warped_img_2, cv2.COLOR_BGR2RGB)
plt.imshow(warped_img_2)
plt.show()

# %% [markdown]
# # One Step approach:

# %%
def get_one_step_homography(horiz_lines, vert_lines):
    A = np.zeros((len(horiz_lines), 5))
    b = np.zeros((len(horiz_lines),))
    
    # Fill in the A and b matrices
    for i, lines in enumerate(zip(horiz_lines, vert_lines)):
        hline = lines[0].hc
        vline = lines[1].hc
        
        # Normalize line coordinates
        hline = hline / hline[2]
        vline = vline / vline[2]
        
        A[i] = np.array([hline[0]*vline[0], hline[1]*vline[1], hline[0]*vline[1] + hline[1]*vline[0], hline[0]*vline[2] + hline[2]*vline[0], hline[1]*vline[2] + hline[2]*vline[1]])
        b[i] = -1 * np.array([hline[2] * vline[2]])
        
    S = (np.linalg.inv((A.T) @ A) @ A.T) @ b
    S = np.array([[S[0], S[2], S[3]], [S[2], S[1], S[4]], [S[3], S[4], 1]])
    U,Sigma,V = np.linalg.svd(S)
    return U

# %%
H_onestep_board = get_one_step_homography(horiz_board_lines, vert_board_lines)
H_onestep_corridor = get_one_step_homography(horiz_corridor_lines, vert_corridor_lines)
H_onestep_kallax = get_one_step_homography(horiz_kallax_lines, vert_kallax_lines)
H_onestep_window = get_one_step_homography(horiz_window_lines, vert_window_lines)

# %%
for i in range(len(horiz_board_lines)):
    print(f"Checking for line {i}: = {horiz_board_lines[i].hc.T @ H_onestep_board @ np.array([[1,0,0],[0,1,0],[0,0,0]]) @ H_onestep_board.T @ vert_board_lines[i].hc}")

# %%
h_rotate = Homography().get_rotational_homography(math.radians(180))

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/board_1.jpeg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/board_1.jpeg")

target_roi = board_true
homography = H_onestep_board
H_resize = get_scaling_and_translation_homography(board_points, h_rotate@homography, (target.shape[1], target.shape[0]))
img = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, h_rotate@H_resize@homography)
plt.imshow(img)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/corridor.jpeg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/corridor.jpeg")

target_roi = corridor_true
homography = H_onestep_corridor
H_resize = get_scaling_and_translation_homography(corridor_true_points, homography, (target.shape[1], target.shape[0]))
img = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@homography)
plt.imshow(img)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/window.jpg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/window.jpg")

target_roi = window_true
homography = H_onestep_window
H_resize = get_scaling_and_translation_homography(window_true_points, homography, (target.shape[1], target.shape[0]))
img = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@homography)
plt.imshow(img)
plt.show()

# %%
source = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/kallax.jpg")
target = cv2.imread("/mnt/cloudNAS3/Adubois/Classes/ECE661/HW3/HW3_Images/kallax.jpg")

target_roi = kallax_true
homography = H_onestep_kallax
H_resize = get_scaling_and_translation_homography(kallax_true_points, homography, (target.shape[1], target.shape[0]))
img = map_image1_onto_roi_on_image2_inrgb(source, target, target_roi, H_resize@homography)
plt.imshow(img)
plt.show()

# %%




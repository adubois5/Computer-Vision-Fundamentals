import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ExifTags

click_list = []

def onclick(event):
    # Get the x and y pixel coordinates
    x, y = event.xdata, event.ydata
    # Append the coordinates to the list
    if x is not None and y is not None:
        click_list.append((x, y))
        print(f"Clicked at x={x}, y={y}")
    # Stop recording clicks after four clicks
    if len(click_list) == 4:
        plt.close()

# Load and display an image
image_path = "C:/Users/adubo/Documents/School/Purdue/Graduate/PhD/ECE661/HW2/img4_elecpanel.png"
img = Image.open(image_path)

# Apply any necessary EXIF orientation corrections
try:
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    exif = img._getexif()

    if exif is not None:
        orientation = exif.get(orientation)

        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)

except (AttributeError, KeyError, IndexError):
    # Cases where image has no EXIF orientation data
    pass

# Display the image
img_array = np.array(img)
plt.imshow(img_array, origin='upper')

# Create a canvas element to get the clicks
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Print the recorded click positions
print("Click positions:", click_list)

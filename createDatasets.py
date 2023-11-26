import os
import cv2
import pandas as pd
from tkinter import Tk, filedialog

# Open image directory dialog
Tk().withdraw()
image_dir = filedialog.askdirectory(title='Select Image Directory')

# Saving folder name
folder_name = os.path.basename(image_dir)

# Read all image in directory
image_files = [filename for filename in os.listdir(image_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(len(image_files))

# Initialize global variable
drawing = False
center_pt, radius = (-1, -1), -1
inner = []
outer = []
image = []
label = []

def draw_circle(event, x, y, flags, param):
    global drawing, center_pt, radius

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        center_pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = max(abs(center_pt[0] - x), abs(center_pt[1] - y))
        cv2.circle(img, center_pt, radius, (0, 255, 0), 2)
        # annotations.append([image_filename, *center_pt, radius, folder_name])

def save_to_csv(image, inner):
    Tk().withdraw()
    csv_filename = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')],
                                                title='Save CSV as')
    if csv_filename:
        image_df = pd.DataFrame({'filename': image[:len(image_files)]})
        inner_df = pd.DataFrame(inner, columns=['innerX', 'innerY', 'innerRadius'])
        outer_df = pd.DataFrame(outer, columns=['outerX', 'outerY', 'outerRadius'])
        label_df = pd.DataFrame({'label': label[:len(image_files)]})
        df = pd.concat([image_df, inner_df, outer_df, label_df], axis=1)

        df.to_csv(csv_filename, index=False)
        print(df)

i = 0
for i in range(2):
    for image_filename in image_files:
        # Read image
        img = cv2.imread(os.path.join(image_dir, image_filename))
        cv2.namedWindow('Image')
        if i == 0:
            cv2.setMouseCallback('Image', draw_circle)
        else:
            cv2.setMouseCallback('Image', draw_circle)
        while True:
            if i == 0:
                cv2.putText(img, 'Draw Inner', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                cv2.imshow('Image', img)
            else:
                cv2.putText(img, 'Draw Outer', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                cv2.imshow('Image', img)
            key = cv2.waitKey(1) & 0xFF

            # Press 's' for saving bounding box annotations and continue to next image
            if key == ord('s'):
                if i == 0:
                    inner.append([*center_pt, radius])
                    image.append(image_filename)
                    label.append(folder_name)
                else:
                    outer.append([*center_pt, radius])
                    image.append(image_filename)
                    label.append(folder_name)
                break

            # Press 'q' for exit without saving and continue to next image
            if key == ord('q'):
                # annotations.append([image_filename, None, None, None, folder_name])
                break



cv2.destroyAllWindows()

# Save all bounding box annotations to csv file
save_to_csv(image, inner)

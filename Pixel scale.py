import cv2
import numpy as np
import os
import pandas as pd
import re

# Define a function to find the scale length within the given ROI of the image.
def find_scale_length(image, roi_coords):
    roi = image[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2]]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    scale_length = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:  # Vertical line
                line_length = abs(y2 - y1)
                scale_length = max(scale_length, line_length)
                cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return scale_length, roi

# Define the folder path and ROI coordinates
folder_path = r'C:\Users\Yu Ran\Desktop\Pixel size'
roi_coords = (130, 64, 300, 600)

# Initialize a list to store the results
results = []

# Iterate over the images in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Extract the number from the filename
        number = int(re.search(r'\d+', filename).group())
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        scale_length, marked_roi = find_scale_length(image, roi_coords)
        results.append({'Filename': filename, 'Number': number, 'Scale Length (px)': scale_length})
        # Save the marked ROI image
        marked_image_path = os.path.join(folder_path, f"marked_{filename}")
        cv2.imwrite(marked_image_path, marked_roi)

# Convert the results into a pandas DataFrame
df_results = pd.DataFrame(results)

# Sort the DataFrame based on the extracted number
df_results = df_results.sort_values(by='Number')

# Drop the 'Number' column as it's no longer needed
df_results = df_results.drop(columns=['Number'])

# Define the Excel file path
excel_file_path = os.path.join(folder_path, 'scale_lengths.xlsx')

# Save the results to an Excel file
df_results.to_excel(excel_file_path, index=False)

print(f"Results have been saved to {excel_file_path}")

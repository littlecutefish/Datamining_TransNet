import numpy as np

try:
    # Load data with comma as the delimiter and set dtype to float32 for 32-bit precision
    data = np.loadtxt(r"C:\Users\Jolie\Downloads\data\data\dblp\raw\dblp_labels.txt", delimiter=",", dtype=np.int64) # np.int64, np.float32
    # Save as .npy file
    np.save("data/input/dblp_y.npy", data)
    print("Saved successfully")
except ValueError as e:
    print("Error loading the data:", e)

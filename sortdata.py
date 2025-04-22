import os
import pandas as pd

# Paths
base_path = r"C:\Users\om.jindal\Desktop\activity"
accel_path = os.path.join(base_path, "raw", "phone", "accel")
gyro_path = os.path.join(base_path, "raw", "phone", "gyro")
output_csv = os.path.join(base_path, "merged_edge_impulse.csv")
output_folder = os.path.join(base_path, "edge_windows")
os.makedirs(output_folder, exist_ok=True)

# Labels to keep
keep_labels = ['Walking', 'Sitting', 'Standing', 'Drinking', 'Kicking', 'Clapping']

# Activity code map
activity_code = {
    'Walking': 'A', 'Sitting': 'D', 'Standing': 'E',
    'Drinking': 'K', 'Kicking': 'M', 'Clapping': 'R'
}
code_to_label = {v: k for k, v in activity_code.items()}

def load_sensor_data(folder):
    all_data = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder, filename), 'r') as f:
                for line in f:
                    parts = line.strip().strip(';').split(',')
                    if len(parts) == 6:
                        user, act, ts, x, y, z = parts
                        all_data.append({
                            'user': int(user),
                            'activity': act,
                            'timestamp': int(ts),
                            'x': float(x),
                            'y': float(y),
                            'z': float(z)
                        })
    return pd.DataFrame(all_data)

print("[INFO] Loading accelerometer data...")
accel_df = load_sensor_data(accel_path)
accel_df = accel_df[accel_df['activity'].isin(code_to_label.keys())]
accel_df['label'] = accel_df['activity'].map(code_to_label)
accel_df = accel_df[accel_df['label'].isin(keep_labels)]

print("[INFO] Loading gyroscope data...")
gyro_df = load_sensor_data(gyro_path)
gyro_df = gyro_df[gyro_df['activity'].isin(code_to_label.keys())]
gyro_df['label'] = gyro_df['activity'].map(code_to_label)
gyro_df = gyro_df[gyro_df['label'].isin(keep_labels)]

# Rename columns
accel_df = accel_df.rename(columns={'x': 'ax', 'y': 'ay', 'z': 'az'})
gyro_df = gyro_df.rename(columns={'x': 'gx', 'y': 'gy', 'z': 'gz'})

print("[INFO] Merging accel + gyro data for all labels...")
merged_df = pd.merge(
    accel_df,
    gyro_df[['user', 'activity', 'timestamp', 'gx', 'gy', 'gz']],
    on=['user', 'activity', 'timestamp'],
    how='inner'
)

# Assign label again based on activity code
merged_df['label'] = merged_df['activity'].map(code_to_label)

final_df = merged_df[['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label']].copy()
final_df = final_df.sort_values(by='timestamp')

print(f"[INFO] Saving merged CSV to {output_csv}")
final_df.to_csv(output_csv, index=False)

# Segment into 2-second windows (assuming 50 Hz sampling)
SAMPLES_PER_WINDOW = 100
MAX_FOLDER_SIZE_MB = 100
MAX_FOLDER_SIZE_BYTES = MAX_FOLDER_SIZE_MB * 1024 * 1024
window_count = 0
folder_size_bytes = 0

print(f"[INFO] Segmenting into {SAMPLES_PER_WINDOW}-sample windows with max folder size {MAX_FOLDER_SIZE_MB} MB...")

for label in keep_labels:
    label_data = final_df[final_df['label'] == label].reset_index(drop=True)
    out_dir = os.path.join(output_folder, label)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(0, len(label_data) - SAMPLES_PER_WINDOW + 1, SAMPLES_PER_WINDOW):
        if folder_size_bytes >= MAX_FOLDER_SIZE_BYTES:
            print(f"[INFO] Reached folder size limit of {MAX_FOLDER_SIZE_MB} MB. Stopping window generation.")
            break

        window = label_data.iloc[i:i + SAMPLES_PER_WINDOW]
        out_file = os.path.join(out_dir, f"{label}_{window_count}.csv")
        window.drop(columns=['label']).to_csv(out_file, index=False)

        file_size = os.path.getsize(out_file)
        folder_size_bytes += file_size
        window_count += 1

    if folder_size_bytes >= MAX_FOLDER_SIZE_BYTES:
        break

print(f"[DONE] Generated {window_count} windows in '{output_folder}' — total size: {folder_size_bytes / (1024 * 1024):.2f} MB")

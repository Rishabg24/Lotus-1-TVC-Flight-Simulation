import csv

eng_file_path = "Estes_D12.eng"
csv_file_path = "motor_thrust_curve_d12.csv"

data_points = []

with open(eng_file_path, 'r') as eng_file:
    for line in eng_file:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith(';'):
            continue
        
        parts = line.split()
        
        # If the first item doesn't contain numeric data, it's the header line
        try:
            time = float(parts[0])
            thrust = float(parts[1])
            data_points.append([time, thrust])
        except ValueError:
            # This handles ignoring the text-based header line
            continue

# Write to CSV
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Time (s)", "Thrust (N)"]) # Headers
    writer.writerows(data_points)

print(f"Successfully converted to {csv_file_path}!")
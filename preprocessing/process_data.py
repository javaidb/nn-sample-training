import pandas as pd
import numpy as np

# Function to generate a single CSV file
def generate_csv(filename):
    # Initialize lists to hold data
    time = []
    speed = []
    boolean = []

    # Random speed changes for the first 10 seconds
    for t in np.arange(0, 10, 0.1):
        time.append(t)
        speed.append(np.random.uniform(-5, 5))  # Random speed between -5 and 5
        boolean.append(np.random.choice([True, False]))

    # Special region: Speed goes from 0 to positive to 0 to negative
    for t in np.arange(10, 14, 0.1):
        time.append(t)
        if t < 12:  # First 2 seconds: Speed goes from 0 to positive
            speed.append(2 * (t - 10))  # Linearly increase speed
        elif t < 13:  # Next second: Speed is 0
            speed.append(0)
        else:  # Last second: Speed goes to negative
            speed.append(-2 * (t - 13))  # Linearly decrease speed
        boolean.append(np.random.choice([True, False]))

    # More random speed changes for the next 10 seconds
    for t in np.arange(14, 24, 0.1):
        time.append(t)
        speed.append(np.random.uniform(-5, 5))  # Random speed between -5 and 5
        boolean.append(np.random.choice([True, False]))

    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'time': time,
        'speed': speed,
        'boolean': boolean
    })
    df.to_csv(filename, index=False)

# Generate 10 CSV files
for i in range(10):
    data_dir = "./data/sample_1"
    filename = f"{data_dir}/speed_data_{i+1}.csv"
    generate_csv(filename)
    print(f"Generated: {filename}")

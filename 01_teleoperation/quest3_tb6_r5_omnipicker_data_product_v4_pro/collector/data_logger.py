"""Small XR-style logger: one Python list becomes one episode PKL file."""

import os
import pickle
from datetime import datetime


class DataLogger:
    """Collect timestep dictionaries and save them as one pickle episode."""

    def __init__(self, log_dir):
        self.log_data = []
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.count = 0
        os.makedirs(self.log_dir, exist_ok=True)

    def add_entry(self, data_entry):
        """Append one timestep dictionary to the current episode."""
        self.log_data.append(data_entry)

    def save(self):
        """Write the current list of timestep dictionaries to one PKL file."""
        if not self.log_data:
            print("No data to save")
            return None

        self.count += 1
        log_file = os.path.join(
            self.log_dir, f"teleop_log_{self.timestamp}_{self.count}.pkl"
        )
        with open(log_file, "wb") as file:
            pickle.dump(self.log_data, file)
        return log_file

    def reset(self):
        """Clear the in-memory episode data."""
        self.log_data = []

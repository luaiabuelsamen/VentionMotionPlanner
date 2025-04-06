import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

# Create global figure and axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def plot_trajectory():
    """
    Plot 3D trajectory from CSV file located in the same directory as this script
    """
    global fig, ax
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # CSV file path (in the same directory as the script)
    csv_file = os.path.join(script_dir, 'end_effector_positions.csv')
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        return
    
    # Initialize lists for coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    
    # Read data from CSV file
    print(f"Reading trajectory data from {csv_file}")
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        
        # Skip header if present (check if first row contains non-numeric values)
        first_row = next(csv_reader, None)
        if first_row:
            try:
                # Try to convert first row values to float
                float(first_row[0])
                float(first_row[1])
                float(first_row[2])
                # If successful, it's data - add to coordinates
                x_coords.append(float(first_row[0]))
                y_coords.append(float(first_row[1]))
                z_coords.append(float(first_row[2]))
            except (ValueError, IndexError):
                # If conversion fails, it's likely a header - skip
                print("Skipping header row")
        
        # Read remaining rows
        for row in csv_reader:
            if len(row) >= 3:  # Ensure we have x,y,z values
                try:
                    x_coords.append(float(row[0]))
                    y_coords.append(float(row[1]))
                    z_coords.append(float(row[2]))
                except ValueError:
                    print(f"Skipping invalid row: {row}")
    
    # Check if we have data to plot
    if not x_coords:
        print("No valid data points found in the CSV file")
        return
    
    print(f"Loaded {len(x_coords)} data points")
    
    # Clear any previous plot data
    ax.clear()
    
    # Plot the trajectory line
    ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, label='Trajectory')
    
    # Plot start and end points
    ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, label='Start')
    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, label='End')
    
    # Add axis labels
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    
    # Add title and legend
    ax.set_title('End Effector Trajectory')
    ax.legend()
    
    # Set equal aspect ratio for all axes
    max_range = np.array([
        max(x_coords) - min(x_coords),
        max(y_coords) - min(y_coords),
        max(z_coords) - min(z_coords)
    ]).max() / 2.0
    
    mid_x = (max(x_coords) + min(x_coords)) / 2
    mid_y = (max(y_coords) + min(y_coords)) / 2
    mid_z = (max(z_coords) + min(z_coords)) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Show the plot with grid
    ax.grid(True)
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(script_dir, 'trajectory_plot.png')
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved as '{output_file}'")
    
    return fig, ax

def display_plot():
    """
    Display the current plot
    """
    global fig
    plt.show()

if __name__ == "__main__":
    plot_trajectory()
    display_plot()
import matplotlib.pyplot as plt
import numpy as np
import os

class Plots():
    def __init__(self, planing):
        self.planning = planing

    def generate_plots(self):
        """Generate and save performance plots"""
        self.planning.get_logger().info('Generating performance plots...')
        
        if not self.planning.ee_trajectory_data:
            self.planning.get_logger().error('No trajectory data collected. Cannot generate plots.')
            return
            
        try:
            # 3D trajectory plot
            self.plot_3d_trajectory()
            
            # Cycle times plot
            self.plot_cycle_times()
            
            # Planning times plot
            self.plot_planning_times()
            
            # 2D projections plot
            self.plot_2d_projections()
            
            # Print statistics
            self.print_performance_statistics()
            
            self.planning.get_logger().info('All plots generated and saved successfully')
            
        except Exception as e:
            self.planning.get_logger().error(f'Error generating plots: {str(e)}')
    
    def plot_3d_trajectory(self):
        """Plot 3D trajectory of end-effector movements"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cycle in range(self.planning.current_cycle):
            cycle_data = [d for d in self.planning.ee_trajectory_data if d['cycle'] == cycle]
            
            # Plot pick trajectory
            pick_data = [d for d in cycle_data if d['goal'] == "Pick"]
            if pick_data:
                pick_x = [d['x'] for d in pick_data]
                pick_y = [d['y'] for d in pick_data]
                pick_z = [d['z'] for d in pick_data]
                ax.plot(pick_x, pick_y, pick_z, 'r-', linewidth=2, alpha=0.7, 
                        label=f"Cycle {cycle+1} Pick" if cycle == 0 else "")
            
            # Plot place trajectory
            place_data = [d for d in cycle_data if d['goal'] == "Place"]
            if place_data:
                place_x = [d['x'] for d in place_data]
                place_y = [d['y'] for d in place_data]
                place_z = [d['z'] for d in place_data]
                ax.plot(place_x, place_y, place_z, 'b-', linewidth=2, alpha=0.7, 
                        label=f"Cycle {cycle+1} Place" if cycle == 0 else "")
        
        # Plot the goal positions
        pick_pos = self.planning.pick_pose.position
        place_pos = self.planning.place_pose.position
        ax.scatter([pick_pos.x], [pick_pos.y], [pick_pos.z], 
                   color='red', s=100, marker='o', label='Pick Position')
        ax.scatter([place_pos.x], [place_pos.y], [place_pos.z], 
                   color='blue', s=100, marker='o', label='Place Position')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('End-Effector Trajectory During Pick and Place Operations')
        ax.legend()
        
        plt.savefig(os.path.join(self.planning.data_dir, 'ee_trajectory_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.planning.get_logger().info('3D trajectory plot saved')
        
    def plot_cycle_times(self):
        """Plot cycle times for all completed cycles"""
        if not self.planning.cycle_times:
            self.planning.get_logger().warn('No cycle time data to plot')
            return
            
        plt.figure(figsize=(10, 6))
        cycles = range(1, len(self.planning.cycle_times) + 1)
        plt.bar(cycles, self.planning.cycle_times, color='skyblue')
        
        # Add average line
        avg_cycle_time = np.mean(self.planning.cycle_times)
        plt.axhline(y=avg_cycle_time, color='r', linestyle='-', label=f'Average: {avg_cycle_time:.2f}s')
        
        plt.xlabel('Cycle Number')
        plt.ylabel('Time (seconds)')
        plt.title('Pick and Place Cycle Times')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.planning.data_dir, 'cycle_times.png'), dpi=300)
        plt.close()
        
        self.planning.get_logger().info('Cycle times plot saved')
        
    def plot_planning_times(self):
        """Plot planning times for pick and place operations"""
        if not self.planning.planning_times_pick or not self.planning.planning_times_place:
            self.planning.get_logger().warn('No planning time data to plot')
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pick planning times
        pick_cycles = range(1, len(self.planning.planning_times_pick) + 1)
        ax1.bar(pick_cycles, self.planning.planning_times_pick, color='coral')
        
        # Add average line for pick planning times
        avg_pick_plan = np.mean(self.planning.planning_times_pick)
        ax1.axhline(y=avg_pick_plan, color='r', linestyle='-', label=f'Avg: {avg_pick_plan:.2f}s')
        
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Pick Planning Times')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Place planning times
        place_cycles = range(1, len(self.planning.planning_times_place) + 1)
        ax2.bar(place_cycles, self.planning.planning_times_place, color='skyblue')
        
        # Add average line for place planning times
        avg_place_plan = np.mean(self.planning.planning_times_place)
        ax2.axhline(y=avg_place_plan, color='r', linestyle='-', label=f'Avg: {avg_place_plan:.2f}s')
        
        ax2.set_xlabel('Cycle Number')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Place Planning Times')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.planning.data_dir, 'planning_times.png'), dpi=300)
        plt.close()
        
        self.planning.get_logger().info('Planning times plot saved')
        
    def plot_2d_projections(self):
        """Create 2D projections of the end-effector trajectory"""
        if not self.planning.ee_trajectory_data:
            self.planning.get_logger().warn('No trajectory data to plot 2D projections')
            return
            
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        
        # XY Projection
        axs[0, 0].set_title('XY Projection')
        axs[0, 0].set_xlabel('X (m)')
        axs[0, 0].set_ylabel('Y (m)')
        axs[0, 0].grid(True)
        
        # XZ Projection
        axs[0, 1].set_title('XZ Projection')
        axs[0, 1].set_xlabel('X (m)')
        axs[0, 1].set_ylabel('Z (m)')
        axs[0, 1].grid(True)
        
        # YZ Projection
        axs[1, 0].set_title('YZ Projection')
        axs[1, 0].set_xlabel('Y (m)')
        axs[1, 0].set_ylabel('Z (m)')
        axs[1, 0].grid(True)
        
        # 3D view in the last subplot
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        ax3d.set_title('3D Trajectory')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        
        # Limit to 5 cycles for clarity
        max_cycles_to_plot = min(self.planning.current_cycle, 5)
        
        for cycle in range(max_cycles_to_plot):
            cycle_data = [d for d in self.planning.ee_trajectory_data if d['cycle'] == cycle]
            
            # Process pick and place separately for coloring
            for goal_type, color, marker in [("Pick", 'red', '-'), ("Place", 'blue', '-')]:
                goal_data = [d for d in cycle_data if d['goal'] == goal_type]
                if goal_data:
                    x = [d['x'] for d in goal_data]
                    y = [d['y'] for d in goal_data]
                    z = [d['z'] for d in goal_data]
                    
                    # Plot projections
                    axs[0, 0].plot(x, y, color=color, linestyle=marker, alpha=0.7)
                    axs[0, 1].plot(x, z, color=color, linestyle=marker, alpha=0.7)
                    axs[1, 0].plot(y, z, color=color, linestyle=marker, alpha=0.7)
                    
                    # Plot 3D
                    ax3d.plot(x, y, z, color=color, linestyle=marker, alpha=0.7)
        
        # Mark goal positions on all plots
        pick_pos = self.planning.pick_pose.position
        place_pos = self.planning.place_pose.position
        
        for ax, coord1, coord2, p1, p2, q1, q2 in [
            (axs[0, 0], 'x', 'y', pick_pos.x, pick_pos.y, place_pos.x, place_pos.y),
            (axs[0, 1], 'x', 'z', pick_pos.x, pick_pos.z, place_pos.x, place_pos.z),
            (axs[1, 0], 'y', 'z', pick_pos.y, pick_pos.z, place_pos.y, place_pos.z)
        ]:
            ax.scatter(p1, p2, color='red', s=100, marker='o', label='Pick Position')
            ax.scatter(q1, q2, color='blue', s=100, marker='o', label='Place Position')
            ax.legend()
        
        # Add 3D markers
        ax3d.scatter(pick_pos.x, pick_pos.y, pick_pos.z, color='red', s=100, marker='o', label='Pick Position')
        ax3d.scatter(place_pos.x, place_pos.y, place_pos.z, color='blue', s=100, marker='o', label='Place Position')
        ax3d.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.planning.data_dir, 'trajectory_projections.png'), dpi=300)
        plt.close()
        
        self.planning.get_logger().info('2D trajectory projections plot saved')
    
    def print_performance_statistics(self):
        """Calculate and print performance statistics"""
        self.planning.get_logger().info("\nPerformance Statistics:")
        
        if self.planning.cycle_times:
            avg_cycle_time = np.mean(self.planning.cycle_times)
            std_cycle_time = np.std(self.planning.cycle_times)
            self.planning.get_logger().info(f"Average Cycle Time: {avg_cycle_time:.4f} ± {std_cycle_time:.4f} seconds")
        
        if self.planning.planning_times_pick:
            avg_pick_plan = np.mean(self.planning.planning_times_pick)
            std_pick_plan = np.std(self.planning.planning_times_pick)
            self.planning.get_logger().info(f"Average Pick Planning Time: {avg_pick_plan:.4f} ± {std_pick_plan:.4f} seconds")
        
        if self.planning.planning_times_place:
            avg_place_plan = np.mean(self.planning.planning_times_place)
            std_place_plan = np.std(self.planning.planning_times_place)
            self.planning.get_logger().info(f"Average Place Planning Time: {avg_place_plan:.4f} ± {std_place_plan:.4f} seconds")
        
        # Calculate average total planning time
        all_planning_times = self.planning.planning_times_pick + self.planning.planning_times_place
        if all_planning_times:
            avg_planning_time = np.mean(all_planning_times)
            std_planning_time = np.std(all_planning_times)
            self.planning.get_logger().info(f"Average Planning Time: {avg_planning_time:.4f} ± {std_planning_time:.4f} seconds")
        
        # Calculate execution statistics
        if self.planning.execution_times_pick and self.planning.execution_times_place:
            avg_pick_exec = np.mean(self.planning.execution_times_pick)
            avg_place_exec = np.mean(self.planning.execution_times_place)
            avg_exec_time = np.mean(self.planning.execution_times_pick + self.planning.execution_times_place)
            self.planning.get_logger().info(f"Average Pick Execution Time: {avg_pick_exec:.4f} seconds")
            self.planning.get_logger().info(f"Average Place Execution Time: {avg_place_exec:.4f} seconds")
            self.planning.get_logger().info(f"Average Execution Time: {avg_exec_time:.4f} seconds")
        
        # Save statistics to file
        stats_file = os.path.join(self.planning.data_dir, 'performance_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("UR5e Pick and Place Performance Statistics\n")
            f.write("=========================================\n\n")
            
            if self.planning.cycle_times:
                f.write(f"Number of completed cycles: {len(self.planning.cycle_times)}\n")
                f.write(f"Average Cycle Time: {avg_cycle_time:.4f} ± {std_cycle_time:.4f} seconds\n")
            
            if all_planning_times:
                f.write(f"Average Planning Time: {avg_planning_time:.4f} ± {std_planning_time:.4f} seconds\n")
                f.write(f"Average Pick Planning Time: {avg_pick_plan:.4f} ± {std_pick_plan:.4f} seconds\n")
                f.write(f"Average Place Planning Time: {avg_place_plan:.4f} ± {std_place_plan:.4f} seconds\n")
            
            if self.planning.execution_times_pick and self.planning.execution_times_place:
                f.write(f"Average Execution Time: {avg_exec_time:.4f} seconds\n")
                f.write(f"Average Pick Execution Time: {avg_pick_exec:.4f} seconds\n")
                f.write(f"Average Place Execution Time: {avg_place_exec:.4f} seconds\n")
        
        self.planning.get_logger().info(f"Performance statistics saved to {stats_file}")

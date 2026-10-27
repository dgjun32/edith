# Alternative: PyQtGraph (much faster for real-time data)
# Install: pip install pyqtgraph

import os
import sys
from typing import List, Tuple, Dict

# Set Qt environment variables before importing pyqtgraph
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'

import pyqtgraph as pg
import numpy as np
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class GraphConfig:
    names: List[str]
    y_ranges: Optional[Tuple[float, float]] = None

class GraphVisualizer:
    def __init__(self, dict_config: Dict[str, GraphConfig], window_size=10, max_points=1000):
        """
        Ultra-fast visualizer using PyQtGraph
        
        Args:
            dict_config: dict of plot_name -> GraphConfig
                        ex) {'joint1': GraphConfig(['position', 'action'], (-4, 4))}
            window_size: float, time window in seconds
            max_points: int, maximum number of points to display
        """
        # Extract names and y_ranges from config
        self.names = {plot_name: config.names for plot_name, config in dict_config.items()}
        self.y_ranges = {plot_name: config.y_ranges for plot_name, config in dict_config.items() 
                        if config.y_ranges is not None}
        self.window_size = window_size
        self.max_points = max_points
        
        # Use deque for fast append/pop operations
        self.x_data = deque(maxlen=max_points)
        self.y_data = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_points)))
        
        # Create main window
        self.app = pg.mkQApp("Graph Visualizer")
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Plots")
        
        # Create plots
        self.plots = {}
        self.curves = {}
        
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
        
        plots_per_column = 3
        
        for i, (plot_name, var_names) in enumerate(self.names.items()):
            # Calculate row and column positions
            row = i % plots_per_column
            col = i // plots_per_column
            
            # Create plot widget at specific position
            plot = self.win.addPlot(title=plot_name, row=row, col=col)
            plot.setLabel('left', plot_name)
            plot.setLabel('bottom', 'Time (s)')
            plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Set y-axis range if specified
            if plot_name in self.y_ranges:
                y_min, y_max = self.y_ranges[plot_name]
                plot.setYRange(y_min, y_max, padding=0)
                plot.enableAutoRange(axis='y', enable=False)  # Disable auto-scaling for y-axis
            else:
                plot.enableAutoRange(axis='y', enable=True)   # Enable auto-scaling for y-axis
            
            self.plots[plot_name] = plot
            
            # Create curves for each variable
            self.curves[plot_name] = {}
            for j, var_name in enumerate(var_names):
                color = colors[j % len(colors)]
                curve = plot.plot(pen=color, name=var_name)
                self.curves[plot_name][var_name] = curve
            
            # Add legend if multiple variables
            if len(var_names) > 1:
                plot.addLegend()
    
    def update(self, x, y_dict):
        """
        Update plots - extremely fast with PyQtGraph
        
        Args:
            x: float, current time
            y_dict: dict, data values
        """
        # Add new data point
        self.x_data.append(x)
        
        # Remove old data outside window
        cutoff_time = x - self.window_size
        while self.x_data and self.x_data[0] < cutoff_time:
            self.x_data.popleft()
            for plot_name in self.y_data:
                for var_name in self.y_data[plot_name]:
                    if self.y_data[plot_name][var_name]:
                        self.y_data[plot_name][var_name].popleft()
        
        # Add new y data
        for plot_name, variables in y_dict.items():
            if plot_name in self.names:
                for var_name, value in variables.items():
                    if var_name in self.names[plot_name]:
                        self.y_data[plot_name][var_name].append(value)
        
        # Update all curves
        x_array = np.array(self.x_data)
        for plot_name, var_names in self.names.items():
            plot = self.plots[plot_name]
            
            # Set x-axis range
            plot.setXRange(x - self.window_size, x, padding=0)
            
            for var_name in var_names:
                if var_name in self.y_data[plot_name]:
                    y_array = np.array(self.y_data[plot_name][var_name])
                    if len(y_array) > 0:
                        self.curves[plot_name][var_name].setData(x_array, y_array)
    
    def close(self):
        """Close the visualizer"""
        self.win.close()
        self.app.quit()


# Example usage with PyQtGraph
if __name__ == "__main__":
    # Create config using GraphConfig
    dict_config = {
        'trigonometric': GraphConfig(['sin', 'cos'], (-1.5, 1.5)),
        'exponential': GraphConfig(['exp', 'log'], (0, 2))
    }
    
    viz = GraphVisualizer(dict_config, window_size=5)
    
    start_time = time.time()
    last_t = start_time
    try:
        while True:
            t = time.time() - start_time
            dt = t - last_t
            print(f"dt: {dt:.4f}, hz: {1/dt:.1f}")
            last_t = t
            y_data = {
                'trigonometric': {
                    'sin': np.sin(t * 2),
                    'cos': np.cos(t * 2)
                },
                'exponential': {
                    'exp': np.exp(-t/5),
                    'log': np.log(t + 1) if t > 0 else 0
                }
            }
            viz.update(t, y_data)
            
            # Process Qt events to keep the GUI responsive
            viz.app.processEvents()
            
            # Small delay to control update rate

            time.sleep(0.001)  # 1ms delay
            
    except KeyboardInterrupt:
        print("Stopping visualization...")
    finally:
        viz.close()
    
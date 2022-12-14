import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Process, Pipe


class GraphBase:
    """
    Base class for 2D/3D Graphs.
    """

    def __init__(self):
        rc_params = {
            'toolbar': 'None',
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelweight': 'bold',
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'axes.edgecolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
        }
        plt.rcParams.update(rc_params)

        self.colors = iter(['#A0E4B0', '#74D2FF', '#F92727', '#FFF200',
                            '#BFCFFF', '#FF99FE', '#FF75A0'] * 2)
        self.fig = plt.figure(figsize=(10, 6), facecolor='black')
        self.fig.text(0.5, 0.86, 'REGRESSION ', transform=self.fig.transFigure,
                      ha='right', va='bottom', fontweight='light', fontsize=16)
        self.fig.text(0.5, 0.86, 'ANALYSIS', transform=self.fig.transFigure,
                      ha='left', va='bottom', fontweight='bold', fontsize=16)
        self.legend = None
        self.bg = None
        self.outputs = {}
        self.benchmark = {}

    class BenchmarkLine:
        def __init__(self, graph, name):
            self.name = name
            self.dtype = np.dtype([('epoch', 'int32'), ('value', 'float32')])
            self.data = np.array([(0, np.nan)], dtype=self.dtype)
            self.line, = graph.ax_benchmarks.semilogy(self.data['epoch'], self.data['value'], ':',
                                                      color=next(graph.colors),
                                                      linewidth=1,
                                                      label=self.name,
                                                      animated=True)
            graph.refresh_legend()

        def update_data(self, epoch, value):
            if not value: return
            if epoch == self.data['epoch'][-1]:
                self.data['value'][-1] = value
            else:
                self.data = np.concatenate((
                    self.data, np.array([(epoch, value)], dtype=self.dtype)
                ))
            self.line.set_data(self.data['epoch'], self.data['value'])

    def flush(self):
        self.fig.canvas.flush_events()


class Graph2D(GraphBase):
    def __init__(self, coords: dict, *targets):
        super().__init__()
        self.ax = self.fig.add_subplot()
        self._ax = self.ax.twinx()  # this controls the y-axis of ax_benchmarks
        self.ax_benchmarks = self._ax.twiny()  # this controls the x-axis of itself
        self.animated_axis = (self.ax.yaxis, self._ax.yaxis, self.ax_benchmarks.xaxis)
        for axis in self.animated_axis:
            axis.set(animated=True)

        self.x = list(coords.values())[0].reshape(-1, 1)
        self.targets = [
            self.ax.plot(self.x, y, '--',
                         color=next(self.colors),
                         label=f'Target {i}',
                         linewidth=3,
                         animated=True)[0] for i, y in enumerate(targets, 1)
        ]

        self.ax.set(xlabel=list(coords)[0], facecolor='black')
        self.ax_benchmarks.set(xlabel='Epoch')
        self._ax.set_ylabel(ylabel='Loss', rotation=-90, labelpad=10)
        plt.tight_layout(rect=(0.03, 0.03, 0.80, 0.8))

        plt.show(block=False)
        self.refresh_background()

    def update(self, outputs: dict, epoch: int, benchmark: dict):
        # update data
        for name, data in outputs.items():
            if name not in self.outputs:
                self.outputs[name] = self.OutputLine(name, self.ax, self.x, next(self.colors))
                self.refresh_legend()
            self.outputs[name].line.set_ydata(data)
        for name, data in benchmark.items():
            if name not in self.benchmark:
                self.benchmark[name] = self.BenchmarkLine(self, name)
            self.benchmark[name].update_data(epoch, data)

        # refresh canvas
        if not benchmark: return
        self.fig.canvas.restore_region(self.bg)
        for axes in (self.ax, self._ax, self.ax_benchmarks):
            axes.relim(visible_only=True)
            axes.autoscale_view(scalex=axes.xaxis.get_visible(), scaley=axes.yaxis.get_visible())
        for axis in self.animated_axis:
            self.fig.draw_artist(axis)
        for benchmark in self.benchmark.values():
            self.ax_benchmarks.draw_artist(benchmark.line)
        for line in self.targets:
            self.ax.draw_artist(line)
        for output in self.outputs.values():
            self.ax.draw_artist(output.line)
        self.fig.canvas.blit(self.fig.bbox)

    class OutputLine:
        def __init__(self, name, ax, x, color):
            self.name = name
            self.data = np.full_like(x, np.nan)
            self.line, = ax.plot(x, self.data, color=color, label=self.name, animated=True)

    def refresh_legend(self):
        if self.legend: self.legend.remove()
        self.legend = self.fig.legend(loc='center left',
                                      bbox_to_anchor=(0.8, 0.45),
                                      bbox_transform=self.fig.transFigure,
                                      facecolor='#333333')
        self.refresh_background()

    def refresh_background(self):
        # store a copy of everything except animated artists
        plt.pause(0.01)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)


class Surface(GraphBase):
    def __init__(self, coords: dict, cmap='viridis'):
        super().__init__()
        self.ax = self.fig.add_subplot(projection='3d')
        self.shape = tuple(arr.size for arr in coords.values())
        self.x, self.y = np.meshgrid(*coords.values(), indexing='ij')
        self.surf = None
        self.cmap = cmap

        coords_name = list(coords)
        self.ax.set(xlabel=coords_name[0], ylabel=coords_name[1], facecolor='black')
        self.ax.grid(False)
        for axis in self.ax.xaxis, self.ax.yaxis, self.ax.zaxis:
            axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.8))
        plt.pause(0.01)

    def update(self, outputs: dict, epoch: int, benchmark: dict):
        for name, data in outputs.items():
            if name not in self.outputs:
                self.outputs[name] = self.OutputSurface(name)
            self.outputs[name].data = data

        if self.surf:
            self.surf.remove()
        for output in self.outputs.values():
            z = output.data.reshape(self.shape).transpose(0, 1)
            self.surf = self.ax.plot_surface(self.x, self.y, z,
                                             rstride=1, cstride=1, cmap=self.cmap)
        self.fig.canvas.draw()

    class OutputSurface:
        def __init__(self, name):
            self.name = name
            self.data = None


class Contour(GraphBase):
    def __init__(self, coords: dict, cmap='viridis'):
        super().__init__()
        self.ax = self.fig.add_subplot()
        self.shape = tuple(arr.size for arr in coords.values())
        self.x, self.y = np.meshgrid(*coords.values(), indexing='ij')
        self.coords_name = list(coords)
        self.colorbar = None
        self.cmap = cmap
        plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.8))
        plt.pause(0.01)

    def update(self, outputs: dict, epoch: int, benchmark: dict):
        for name, data in outputs.items():
            if name not in self.outputs:
                self.outputs[name] = self.OutputSurface(name)
            self.outputs[name].data = data

        self.ax.clear()
        if self.colorbar:
            self.colorbar.remove()
        self.ax.set(xlabel=self.coords_name[0], ylabel=self.coords_name[1], facecolor='black')
        for output in self.outputs.values():
            z = output.data.reshape(self.shape).transpose(0, 1)
            surf = self.ax.contourf(self.x, self.y, z, 100, cmap=self.cmap)
            self.colorbar = self.fig.colorbar(surf)
        self.fig.canvas.draw()

    class OutputSurface:
        def __init__(self, name):
            self.name = name
            self.data = None


class Graph(Process):
    def __init__(self, coords: dict, *targets, graph='contour', cmap='viridis'):
        super().__init__(daemon=True)
        self.coords = coords
        self.targets = targets
        self.type = graph
        self.cmap = cmap
        self.pipe_out, self.pipe_in = Pipe(duplex=False)
        self.start()

    def run(self):
        dim = len(self.coords)
        if dim == 1:
            graph = Graph2D(self.coords, *self.targets)
        elif dim == 2:
            if self.type == 'contour':
                graph = Contour(self.coords, self.cmap)
            elif self.type == 'surface':
                graph = Surface(self.coords, self.cmap)
            else:
                raise Exception(f'Unknown graph type {self.type}. `graph` shall either be "contour" or "surface".')
        else:
            raise Exception(f'Cannot visualize data with dimension {dim}.')
        while True:
            if self.pipe_out.poll():
                graph.update(*self.pipe_out.recv())
            graph.flush()

    def __call__(self, *data):
        self.pipe_in.send(data)

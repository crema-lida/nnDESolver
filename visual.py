import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
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
        self.fig = plt.figure(figsize=(8, 5), facecolor='black')
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
            if value == 0: return
            if epoch == self.data['epoch'][-1]:
                self.data['value'][-1] = value
            else:
                self.data = np.concatenate((
                    self.data, np.array([(epoch, value)], dtype=self.dtype)
                ))
            self.line.set_data(self.data['epoch'], self.data['value'])


class Graph2D(GraphBase):
    """
    Args:
        inputs: The x-axis, a tensor or numpy array.
        targets (optional): Y-value of targets, tensors or numpy arrays.

    Examples:
        update_graph = visual.Graph2D(inputs, target1, target2, ...)

        To plot new data, simply call

        update_graph(epoch, outputs, **kwargs)

        `epoch`:
            The current loop number.
        `outputs`:
            Array or sequence of arrays. Tensors and numpy arrays are both acceptable.
        `kwargs`:
            To plot some losses or other benchmarks, pass keyword arguments like:
            'Total Loss'=1e-2
    """

    def __init__(self, inputs, *targets):
        super().__init__()
        self.ax = self.fig.add_subplot()
        self._ax = self.ax.twinx()  # this controls the y-axis of ax_benchmarks
        self.ax_benchmarks = self._ax.twiny()  # this controls the x-axis of itself
        self.animated_axis = (self.ax.yaxis, self._ax.yaxis, self.ax_benchmarks.xaxis)
        for axis in self.animated_axis:
            axis.set(animated=True)

        self.inputs = inputs
        self.targets = [
            self.ax.plot(inputs, y, '--',
                         color=next(self.colors),
                         label=f'Target {i}',
                         linewidth=3,
                         animated=True)[0] for i, y in enumerate(targets, 1)
        ]

        self.ax.set(xlabel='X', ylabel='Y', facecolor='black')
        self.ax_benchmarks.set(xlabel='Epoch')
        self._ax.set_ylabel(ylabel='Loss', rotation=-90, labelpad=10)
        plt.tight_layout(rect=(0.03, 0.03, 0.80, 0.8))

        plt.show(block=False)
        self.refresh_background()

    def __call__(self, indices, outputs: dict, epoch: int, benchmark: dict):
        # update data
        for name, data in outputs.items():
            if name not in self.outputs:
                self.outputs[name] = self.OutputLine(self, name)
            self.outputs[name].update_data(indices, data)
        for name, data in benchmark.items():
            if name not in self.benchmark:
                self.benchmark[name] = self.BenchmarkLine(self, name)
            self.benchmark[name].update_data(epoch, data)

        # refresh canvas
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
        self.fig.canvas.flush_events()

    class OutputLine:
        def __init__(self, graph, name):
            self.name = name
            self.data = np.empty(graph.inputs.shape)
            self.data.fill(np.nan)
            self.line, = graph.ax.plot(graph.inputs, self.data,
                                       color=next(graph.colors),
                                       label=self.name,
                                       animated=True)
            graph.refresh_legend()

        def update_data(self, indices, data):
            self.data[indices] = data
            self.line.set_ydata(self.data)

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


class Graph3d(GraphBase):
    def __init__(self, inputs, shape, *targets):
        super().__init__()
        self.ax = self.fig.add_subplot(projection='3d')
        self.shape = shape
        self.x = inputs[:, 0].reshape(shape).transpose(0, 1)
        self.y = inputs[:, 1].reshape(shape).transpose(0, 1)
        self.targets = [
            z.reshape(self.shape).transpose(0, 1) for z in targets
        ]

        self.ax.set(xlabel='X', ylabel='Y', zlabel='z', facecolor='black')
        plt.tight_layout(rect=(0.05, 0.05, 0.9, 0.8))
        plt.pause(0.01)

    def __call__(self, indices, outputs: dict, epoch: int, benchmark: dict):
        self.ax.clear()
        for i, z in enumerate(self.targets):
            self.ax.plot_surface(self.x, self.y, z, label=f'Target{i}')
        for name, data in outputs.items():
            if name not in self.outputs:
                self.outputs[name] = self.OutputSurface(name, (np.prod(self.shape), 1))
            self.outputs[name].update_data(indices, data)
            self.ax.plot_surface(self.x, self.y, self.outputs[name].data.reshape(self.shape).transpose(0, 1))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    class OutputSurface:
        def __init__(self, name, output_shape):
            self.name = name
            self.data = np.empty(output_shape)
            self.data.fill(np.nan)

        def update_data(self, indices, data):
            self.data[indices] = data


class Graph(Process):
    def __init__(self, inputs, *targets, shape=None):
        super().__init__(daemon=True)
        self.inputs = inputs
        self.targets = targets
        self.shape = shape
        self.data = ()
        self.data_id = id(self.data)
        self.pipe_out, self.pipe_in = Pipe(duplex=False)
        self.start()

    def recv(self):
        while self.pipe_out.poll(1.):
            self.data = self.pipe_out.recv()

    def run(self):
        inputs_shape = tuple(self.inputs.shape)
        if len(inputs_shape) == 1 or (dim := inputs_shape[1]) == 1:
            graph = Graph2D(self.inputs, *self.targets)
        elif dim == 2:
            graph = Graph3d(self.inputs, self.shape, *self.targets)
        else:
            raise Exception(f'Could not parse inputs shape {inputs_shape}. '
                            'Shape of `inputs` should be (N, 1) or (N, 2)')
        listener = Thread(target=self.recv)
        listener.start()
        while listener.is_alive():
            if id(self.data) == self.data_id: continue
            self.data_id = id(self.data)
            graph(*self.data)
        plt.show(block=True)

    def __call__(self, *data):
        self.pipe_in.send(data)

    def freeze(self):
        self.join()

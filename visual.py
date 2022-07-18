import numpy as np
import matplotlib.pyplot as plt


class Graph:
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
        self.outputs = []
        self.benchmark = {}

    class BenchmarkLine:
        def __init__(self, graph, name):
            self.name = name
            self.data = np.array([[], []])
            self.line, = graph.ax_benchmarks.semilogy(self.data[0], self.data[1], ':',
                                                      color=next(graph.colors),
                                                      linewidth=1,
                                                      label=self.name,
                                                      animated=True)
            graph.refresh_legend()

        def update_data(self, epoch, value):
            value = value.cpu().detach().numpy()
            if value == 0: return
            self.data = np.hstack((
                self.data, np.array([[epoch], [value]])
            ))
            self.line.set_data(self.data)

    def refresh_background(self):
        # store a copy of everything except animated artists
        plt.pause(0.01)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def refresh_legend(self):
        if self.legend: self.legend.remove()
        self.legend = self.fig.legend(loc='center left',
                                      bbox_to_anchor=(0.8, 0.45),
                                      bbox_transform=self.fig.transFigure,
                                      facecolor='#333333')
        self.refresh_background()

    @staticmethod
    def freeze():
        plt.show(block=True)


class Graph2D(Graph):
    """
    Args:
        x: The X axis for trained data or targets, must be 2-d array(s).
        targets (optional): Y-value of targets, must be 2-d array(s).

    Examples:
        update_graph = visual.Graph2D(epoch, x, target1, target2, ...)

        To plot new data, simply call

        update_graph(cycle, y, loss)

        `cycle`:
            The current loop number.
        `y`:
            Array or sequence of arrays. Tensors and numpy arrays are both acceptable.
        `loss`:
            A Dictionary of tensors.
            ONLY one tensor for each key; the tensor should not have dimensions.

            e.g., loss={'Total Loss': tensor(1.), 'Other Benchmark': tensor(0.01)}
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
                         label=f'Target{i}',
                         linewidth=3,
                         animated=True)[0] for i, y in enumerate(targets, 1)
        ]

        self.ax.set(xlabel='X', ylabel='Y', facecolor='black')
        self.ax_benchmarks.set(xlabel='Epoch')
        self._ax.set_ylabel(ylabel='Loss', rotation=-90, labelpad=10)
        plt.tight_layout(rect=(0.03, 0.03, 0.80, 0.8))

        plt.show(block=False)
        self.refresh_background()

    def __call__(self, epoch, output, benchmark: dict):
        # update data
        output = output if isinstance(output, (tuple, list)) else (output, )
        if self.outputs:
            for i, data in enumerate(output):
                self.outputs[i].set_ydata(data)
        else:
            self.outputs = [self.ax.plot(self.inputs, data,
                                         color=next(self.colors),
                                         label=f'Output{i}',
                                         animated=True)[0] for i, data in enumerate(output, 1)]
            self.refresh_legend()
        for name, value in benchmark.items():
            if name not in self.benchmark:
                self.benchmark[name] = self.BenchmarkLine(self, name)
            self.benchmark[name].update_data(epoch, value)

        # refresh canvas
        self.fig.canvas.restore_region(self.bg)
        for axes in (self.ax, self._ax, self.ax_benchmarks):
            axes.relim(visible_only=True)
            axes.autoscale_view(scalex=axes.xaxis.get_visible(), scaley=axes.yaxis.get_visible())
        for axis in self.animated_axis:
            self.fig.draw_artist(axis)
        for benchmark in self.benchmark.values():
            self.ax_benchmarks.draw_artist(benchmark.line)
        for line in self.targets + self.outputs:
            self.ax.draw_artist(line)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()


class Graph3d(Graph):
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

    def __call__(self, epoch, output, benchmark: dict = None):
        output = output if isinstance(output, (tuple, list)) else (output, )
        self.ax.clear()

        for i, z in enumerate(self.targets):
            self.ax.plot_surface(self.x, self.y, z, label=f'Target{i}')

        for i, data in enumerate(output):
            self.ax.plot_surface(self.x, self.y, data.reshape(self.shape).transpose(0, 1),
                                 label=f'Output{i}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def graph(inputs, *targets, shape=None):
    inputs_shape = tuple(inputs.shape)
    if len(inputs_shape) == 1 or (dim := inputs_shape[1]) == 1:
        return Graph2D(inputs, *targets)
    elif dim == 2:
        return Graph3d(inputs, shape, *targets)
    else:
        raise Exception(f'Could not parse inputs shape {inputs_shape}. '
                        'Shape of `inputs` should be (N, 1) or (N, 2)')

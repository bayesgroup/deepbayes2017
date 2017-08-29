import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize

from matplotlib import pyplot

import GPy


def lower_confidence_bound(mean_values, std_values, coefficient=2):
    return mean_values.ravel() - coefficient * std_values.ravel()


def log_expected_improvement(mean_values, variance_values, opt_value):
    estimated_values = mean_values.ravel()
    eps = 0.05/len(estimated_values)

    delta = (opt_value - estimated_values - eps).ravel()

    estimated_errors = (variance_values ** 0.5).ravel()

    non_zero_error_inds = np.where(estimated_errors > 1e-6)[0]
    Z = np.zeros(len(delta))
    Z[non_zero_error_inds] = delta[non_zero_error_inds]/estimated_errors[non_zero_error_inds]
    log_EI = np.log(estimated_errors) + norm.logpdf(Z) + np.log(1 + Z * np.exp(norm.logcdf(Z) - norm.logpdf(Z)))
    return log_EI


def expected_improvement(mean_values, std_values, opt_values):
    improvement = (opt_values.ravel()[0] - mean_values).ravel()
    std_values = std_values.ravel()
    EI = improvement * norm.cdf(improvement / std_values) + std_values * norm.pdf(improvement / std_values)
    return EI


def get_new_point(model, lb, ub, data=None, multistart=10, criterion='ei', k=1, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    lb = np.array(lb).reshape(1, -1)
    ub = np.array(ub).reshape(1, -1)
    x_random = random_state.uniform(size=(multistart, np.array(lb).ravel().shape[0]))
    x_random *= ub - lb
    x_random += lb

    def objective(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mean_values, variance = model.predict(x)
        std_values = np.sqrt(variance)
        if criterion == 'ei':
            return -log_expected_improvement(mean_values, std_values, data[1].min())
        elif criterion == 'lcb':
            return lower_confidence_bound(mean_values, std_values, k)
        else:
            raise NotImplementedError('Criterion is not implemented!')

    criterion_value = objective(x_random)

    best_result = None
    best_value = np.inf
    for x_init in x_random:
        optimization_result = minimize(objective, x_init, method='L-BFGS-B', bounds=np.vstack((lb, ub)).T)

        if optimization_result.fun < best_value:
            best_result = optimization_result
            best_value = best_result.fun[0]
    return best_result.x, best_result.fun


def optimization_step(x_train, y_train, kernel, objective, lb=None, ub=None, criterion='ei', k=1, plot=False):
    model = GPy.models.GPRegression(x_train, y_train, kernel)
    model.optimize_restarts(num_restarts=10, verbose=False)

    x_new, criterion_value = get_new_point(model, data=(x_train, y_train), lb=lb, ub=ub, criterion=criterion, k=k)
    if plot:
        plot1d(x_train, y_train, model, objective, x_new, criterion_value)
        pyplot.show()

    x_new = x_new.reshape(1, -1)
    x_train = np.vstack([x_train, x_new])
    y_train = np.vstack([y_train, np.asarray(objective(x_new)).reshape(1, -1)])
    return x_train, y_train, model


def plot1d(x_train, y_train, model, objective, x_new, criterion_value):
    x_grid = np.linspace(0, 1, 100).reshape(-1, 1)
    y_grid = objective(x_grid)

    prediction, variance = model.predict(x_grid)
    std = np.sqrt(variance)
    prediction = prediction.ravel()
    std = std.ravel()

    pyplot.figure(figsize=(8, 6))
    pyplot.plot(x_train, y_train, 'or', markersize=8, label='Training set')
    pyplot.plot(x_grid, y_grid, '--b', linewidth=2, label='True function')
    pyplot.plot(x_grid, prediction, '-k', linewidth=2, label='Approximation')
    pyplot.fill_between(x_grid.ravel(), prediction - 2 * std, prediction + 2 * std, alpha=0.3)
    pyplot.plot(x_new, objective(x_new), 'og', markersize=10, label='New point')
    pyplot.ylim([-15, 20])
    pyplot.legend(loc='best')


def plot2d(objective, x_train, y_train, model):
    grid_size = 50
    x = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
    x = np.hstack((x[0].reshape(-1, 1), x[1].reshape(-1, 1)))
    y = objective(x)

    prediction, variance = model.predict(x)
    std = np.sqrt(variance).ravel()

    x_train = (x_train + 1) * grid_size / 2
    log_EI = np.exp(log_expected_improvement(prediction, std, y_train.min()))

    values = [prediction, y, std, log_EI]
    names = ['Predicted values', 'Exact values', 'Predicted std', 'log EI']

    figure, axes = pyplot.subplots(nrows=2, ncols=2, figsize=(6, 6))

    for i, ax in enumerate(axes.ravel()):
        if i < 3:
            ax.imshow(values[i].reshape(grid_size, grid_size), vmin=0, vmax=1, alpha=0.8)
        else:
            ax.imshow(values[i].reshape(grid_size, grid_size), alpha=0.8)
        ax.scatter(x_train[:-1, 0], x_train[:-1, 1], c='r', s=20)
        ax.scatter(x_train[-1, 0], x_train[-1, 1], marker='d', edgecolor='k', c='g', s=180)
        ax.set_xlim([-0.5, grid_size + 0.5])
        ax.set_ylim([-0.5, grid_size + 0.5])
        ax.axis('off')
        ax.set_title(names[i])

    figure.tight_layout()


def demo_2d(n_init, budget, kernel, save_path='./library/2d_demo.mp4'):
    global x_train, y_train, model

    def f2d(x):
        t = np.sum((x + 0.6)**2, axis=1) - 0.3
        y = np.sin(t)**2 / np.tanh(t**2 + 0.4)
        return y.reshape(-1, 1)

    lb = [-1, -1]
    ub = [1, 1]
    np.random.seed(42)
    x_train = np.random.rand(n_init, 2) * 2 - 1
    y_train = f2d(x_train)

    model = GPy.models.GPRegression(x_train, y_train, kernel)
    model.optimize()

    # Set up formatting for the movie files
    import matplotlib.animation as animation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    Writer = animation.writers['ffmpeg_file']
    writer = Writer(fps=1, metadata=dict(artist='Yermek Kapushev'))

    grid_size = 50
    x = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
    x = np.hstack((x[0].reshape(-1, 1), x[1].reshape(-1, 1)))
    y = f2d(x)


    def get_model_values(model, x, x_train):
        prediction, variance = model.predict(x)
        std = np.sqrt(variance).ravel()

        log_EI = np.exp(log_expected_improvement(prediction, std, y_train.min()))

        values = [prediction, y, log_EI]
        return values


    values = get_model_values(model, x, x_train)
    history = [y_train.min()]
    names = ['Predicted values', 'Exact values', 'log EI']

    # Set up initial canvas
    figure, axes = pyplot.subplots(nrows=2, ncols=2, figsize=(6, 6))
    heatmaps = []
    scatters = []
    new_point_scatters = []
    for i, ax in enumerate(axes.ravel()[:-1]):
        heatmaps.append(ax.matshow(values[i].reshape(grid_size, grid_size), alpha=0.8))
        x_scatter = (x_train + 1) * grid_size / 2
        scatters.append(ax.scatter(x_scatter[:-1, 0], x_scatter[:-1, 1], c='r', s=20))
        new_point_scatters.append(ax.scatter(x_scatter[-1, 0], x_scatter[-1, 1], marker='d', edgecolor='k',
                                             c='g', s=180))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        figure.colorbar(heatmaps[-1], cax=cax)
        ax.set_xlim([-0.5, grid_size + 0.5])
        ax.set_ylim([-0.5, grid_size + 0.5])
        ax.axis('off')
        ax.set_title(names[i])

    convergence_plot = axes.ravel()[-1].plot([y_train.shape[0]], [y_train.min()], '-')
    axes.ravel()[-1].set_xlabel('iteration')
    axes.ravel()[-1].set_ylabel(r'$y_{min}$')
    axes.ravel()[-1].set_xlim([n_init - 1, n_init + budget])
    axes.ravel()[-1].set_ylim([0, 0.0073])
    figure.tight_layout()


    # Define function that updates figure
    def update_fig(iteration):
        global x_train, y_train, model
        # global y_train
        # global model

        if iteration == 0:
            return [*heatmaps, *scatters, *new_point_scatters, *convergence_plot]

        model = GPy.models.GPRegression(x_train, y_train, model.kern)
        model.optimize()

        x_new, criterion = get_new_point(model, lb, ub, data=(x_train, y_train), multistart=10, random_state=None)
        x_new = x_new.reshape(1, -1)
        x_train = np.vstack([x_train, x_new])
        y_train = np.vstack([y_train, f2d(x_new)])
        history.append(y_train.min())

        values = get_model_values(model, x, x_train)

        for i, val in enumerate(values):
            heatmaps[i].set_array(val.reshape(grid_size, -1))
            x_scatter = (x_train + 1) * grid_size / 2
            scatters[i].set_offsets(x_scatter[:-1])
            new_point_scatters[i].set_offsets(x_scatter[-1:])

            # adjust colorbar for std and log EI plot
            vmin = val.min()
            vmax = val.max()
            heatmaps[i].set_clim(vmin, vmax)

        convergence_plot[0].set_data(range(n_init, y_train.shape[0] + 1), history)

        return [*heatmaps, *scatters, *new_point_scatters, *convergence_plot]



    anim = animation.FuncAnimation(figure, update_fig,
                                   blit=False,
                                   repeat=False,
                                   frames=budget)
    anim.save(save_path, writer=writer)

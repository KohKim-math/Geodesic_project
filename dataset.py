from utils import Gaussian_density
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import multivariate_normal
import plotly.graph_objs as go

def Gaussian_density(n_points, visualize=False):
    scale = 10
    mean = [0, 0]
    cov = [[1, 0], [0, 4]]
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mean, cov)
    Z = scale*rv.pdf(pos)

    points = np.random.uniform(-6,6, size=(3*n_points, 2))
    points2 = np.random.multivariate_normal(mean=mean, cov=cov, size = 3*n_points)

    disk_points = np.zeros((n_points,2))
    count = 0
    twothird = int(2*n_points/3)
    
    for pt in points:
        if np.square(pt).sum() < 36:
            disk_points[count] = pt
            count += 1
        
        if count == twothird:
            break

    for pt in points2:
        if np.square(pt).sum() < 36:
            disk_points[count] = pt
            count += 1
        if count == n_points:
            break

    values = scale*rv.pdf(disk_points)
    data = np.concatenate((disk_points, values[:, np.newaxis]), axis=1)
    if visualize:
        fig = go.Figure(data=[
        go.Scatter3d(x=disk_points[:, 0], y=disk_points[:, 1], z=values,
                    mode='markers', marker=dict(size=3)),
        go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.5)
        ])

        # Set the layout of the plot and show it
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Density'),
                        margin=dict(l=0, r=0, b=0, t=0), width=300, height=300)
        fig.show()
    return data

def paraboloid_samples(a, b, n_points, hole=False):
    """
    Returns n samples from the graph of the function f(x,y)=a*x^2+b*y^2 with domain a*x^2 + b*y^2 <= a*b
    """
    data = []
    while len(data) < n_points:
        x = np.random.uniform(-np.sqrt(b), np.sqrt(b))
        y = np.random.uniform(-np.sqrt(a), np.sqrt(a))
        z = a*x**2 + b*y**2
        if hole:
            if (0.2)*a*b <= z and z <= a*b:
                data.append((x,y,z))
        else:
            if z <= a*b:
                data.append((x, y, z))
        
    return np.asarray(data)

def transform(sample_generator, n_train_points, validation_ratio = 0.3, batch_size=32):
    train_data = sample_generator(n_train_points)
    n_valid_points = int(n_train_points*validation_ratio)
    valid_indices = np.random.choice(n_train_points, size=n_valid_points, replace=False)
    valid_indices_bool = np.zeros(n_train_points, dtype=bool)
    for index in valid_indices:
        valid_indices_bool[index] = True
    valid_data = train_data[valid_indices_bool]
    train_data = train_data[~valid_indices_bool]
    
    train_data = TensorDataset(torch.from_numpy(train_data))
    valid_data = TensorDataset(torch.from_numpy(valid_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    if validation_ratio == 0:
        return train_loader

    return train_loader, valid_loader  

def Gaussian_dataset(n_train_points, validation_ratio=.3, batch_size=32):
    return transform(Gaussian_density, n_train_points, validation_ratio=validation_ratio, batch_size=batch_size)
    
def paraboloid_dataset(a, b, n_train_points, hole=False, validation_ratio=.3, batch_size=32):
    def paraboloid(n_points):
        return paraboloid_samples(a,b,n_points, hole=hole)
    return transform(paraboloid, n_train_points, validation_ratio=validation_ratio, batch_size=batch_size)


def swissroll_dataset(n_train_points, validation_ratio =.3, batch_size=32, with_color=True):
    from sklearn.datasets import make_swiss_roll
    swissroll, color = make_swiss_roll(n_samples=n_train_points, noise=0.0, random_state=None, hole=True)
    n_valid_points = int(n_train_points*validation_ratio)
    valid_indices = np.random.choice(n_train_points, size=n_valid_points, replace=False)
    valid_indices_bool = np.zeros(n_train_points, dtype=bool)
    for index in valid_indices:
        valid_indices_bool[index] = True
    valid_data, valid_color = swissroll[valid_indices_bool], color[valid_indices_bool]
    train_data, train_color = swissroll[~valid_indices_bool], color[~valid_indices_bool]
    train_data = TensorDataset(torch.from_numpy(train_data))
    valid_data = TensorDataset(torch.from_numpy(valid_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    if with_color:
        return train_loader, train_color, valid_loader, valid_color
    else:
        return train_loader, valid_loader
import os
import numpy as np
from scipy.stats import multivariate_normal
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib.patches import Ellipse
import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


colors = ['rgb(255, 255, 0)', 'rgb(0, 255, 0)', 'rgb(255, 0, 0)',
              'rgb(0, 0, 255)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
              'rgb(227, 119, 194)', 'rgb(214, 39, 40)', 'rgb(188, 189, 34)',
              'rgb(23, 190, 207)']

color_arrays = np.array([[list(map(int, c[4:-1].split(', ')))] for c in colors])/255.0
    
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


def compare_point_clouds(*args, title='3D Scatter Plot'):
    
    # create list of trace objects for scatter plot
    traces = []
    for i, arr in enumerate(args):
        n_points = len(arr)
        color = colors[i % len(colors)]
        trace = go.Scatter3d(
            x=arr[:, 0],
            y=arr[:, 1],
            z=arr[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                opacity=0.8
            )
        )
        traces.append(trace)

    # create layout for the plot
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )

    # create figure object
    fig = go.Figure(data=traces, layout=layout)

    # display the figure
    pio.show(fig, width=600, height=400)


def ellipse(center, M, scale=1.0, RGBA = (1,0,0,0.5)):

    # Convert center and M to numpy arrays if they are torch tensors
    if isinstance(center, torch.Tensor):
        center = center.numpy()
    if isinstance(M, torch.Tensor):
        M = M.numpy()

    # Compute eigenvalues and eigenvectors of the matrix
    eig_vals, eig_vecs = np.linalg.eig(M)

    # Get major and minor axes lengths from eigenvalues and scale parameter
    major = 2 * scale * np.sqrt(eig_vals[0])
    minor = 2 * scale * np.sqrt(eig_vals[1])

    # Get rotation angle of ellipse from eigenvectors
    angle = np.arctan2(eig_vecs[1][0], eig_vecs[0][0]) * 180 / np.pi

    # Create an ellipse patch using major, minor axes lengths and rotation angle
    ellipse = Ellipse(xy=center, width=major, height=minor, angle=angle, fill=True, facecolor=RGBA)
    return ellipse

def perturb_tensor(input_tensor, std=10e-2):
    # Create a new tensor with the same shape as the input tensor
    output_tensor = torch.zeros(input_tensor.shape)

    # Copy the first and last elements of the input tensor to the output tensor
    output_tensor[0] = input_tensor[0]
    output_tensor[-1] = input_tensor[-1]

    # Perturb the rest of the elements in the output tensor
    noise = torch.normal(mean=torch.zeros(input_tensor.shape), std=std)
    output_tensor[1:-1] = input_tensor[1:-1] + noise[1:-1]

    # Return the perturbed tensor
    return output_tensor

# quoted from https://quokkas.tistory.com/37
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.temp_early_stop = False

    def __call__(self, val_loss, model):
        self.temp_early_stop = False
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        torch.save(model.state_dict(), self.path) # saves only learned parameter
        self.val_loss_min = val_loss
        self.temp_early_stop = True
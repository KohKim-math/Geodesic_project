import torch
from utils import perturb_tensor

def length_of_curve(curve):
    delta = curve[1:]-curve[:-1]
    return torch.sqrt(torch.square(delta).sum())


def linear_interpolation(v, w, num_points, only_internal=False):
    """
    Linearly interpolates between two points represented by 'torch.Tensor' objects 'v' and 'w'
    and returns an array of 'num_points' points.
    """
    assert v.shape == w.shape, "The shapes of the input tensors should be the same."
    assert num_points > 1, "The number of points should be greater than 1."

    t = torch.linspace(0, 1, num_points, requires_grad=True)
    interpolation = torch.outer(t, w) + torch.outer(1 - t, v)

    if only_internal:
        interpolation = interpolation[1:-1].detach().clone().requires_grad_(True)
        # interpolation.requires_grad = True

    return interpolation


def dirichlet_energy(metric, curve, is_tensor=False):
    """
    Computes the Dirichlet energy of a discretized curve using a given metric tensor.

    Args:
        metric_tensor (function): Function that takes a torch.Tensor of shape (n, 2) and returns a
                                torch.Tensor of shape (n, 2, 2).
        curve (torch.Tensor): Input tensor of shape (m, 2), where m is the number of points in
                                the discretized curve.

    Returns:
        float: Dirichlet energy of the curve.

    Raises:
        TypeError: If inputs are not torch.Tensors or if the output of metric_fn is not a
                    torch.Tensor of shape (n, 2, 2).
    """
    # Check inputs are torch.Tensors
    if not isinstance(curve, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensors")
    
    # Compute metric tensor using input function
    if is_tensor:
        metric_tensor = metric
    else:
        metric_tensor = metric(curve)

    
    # Check metric tensor has correct shape
    if metric_tensor.shape != (len(curve), 2, 2):
        raise TypeError("Metric tensor must have shape (n, 2, 2)")
    
    # Compute tangent vectors and lengths along curve
    tangent = curve[1:] - curve[:-1]
    
    # Compute Dirichlet energy
    forward_energy = torch.einsum('ij,ijk,ij->', tangent, metric_tensor[:-1], tangent)
    backward_energy = torch.einsum('ij,ijk,ij->', tangent, metric_tensor[1:], tangent)
    energy = forward_energy + backward_energy
    return energy


def pullback_metric(f, z):
    """
    Computes the (Euclidean) pullback metric of a function f calculated at z.

    Args:
        f (function): Function that takes a torch.Tensor of shape (bs, input_dim) and returns a
                                torch.Tensor of shape (bs, output_dim).
        z (torch.Tensor): Input tensor of shape (bs, input_dim), where bs is the batch size, and 
                                input_dim is the dimension of input.

    Returns:
        float: Euclidean pullback metric of size (bs, input_dim, input_dim)

    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    bs = z.shape[0]
    output_dim = f(z[0]).numel()
    input_dim = z.shape[1]
    J = torch.zeros(bs, output_dim, input_dim)
    # Compute the Jacobian matrix of f at z
    for i in range(bs):
        J[i] = torch.autograd.functional.jacobian(f, z[i], create_graph=True).view(output_dim, input_dim)
    # Compute the pullback metric
    G = torch.einsum('...ki,...kj->...ij', J, J)
    return G

def energy_in_latent(f, curve):
    tangent_z = curve[1:] - curve[:-1]
    _, v = torch.autograd.functional.jvp(f, curve[:-1], tangent_z, create_graph=True)
    energy = 0.5 * (v ** 2).sum()
    return energy


def minimize_length_manual_update(z_init, z_final, generator, num_points, n_iter, lr = 0.001):
    generator.eval()
    
    # initialize discrete curve with a straight line between end points
    curve_internal = linear_interpolation(z_init, z_final, num_points, only_internal=False)
    curve_internal = perturb_tensor(curve_internal[1:-1], std=10e-2).detach().clone().requires_grad_(True)
    curve_z = torch.cat((z_init.unsqueeze(0), curve_internal, z_final.unsqueeze(0)))
    curve_z_original = curve_z.detach().clone()
    
    print(f'initial energy of curve : {energy_in_latent(generator, curve_z_original)}')

    for iter in range(n_iter):
        curve_recon = generator(curve_z)
        for i in range(1,len(curve_z)-1):
            z = curve_z[i]
            v = curve_recon[i+1] - 2*curve_recon[i] + curve_recon[i-1]
            grad = -(num_points-1)*(torch.autograd.functional.vjp(generator, z, v)[1])
            # update z
            curve_z[i] -=lr*grad

    curve_recon = generator(curve_z)
    print(f'optimized energy of curve : {energy_in_latent(generator, curve_z)}')
    return curve_z, curve_z_original

def minimize_length(z_init, z_final, generator, num_points, n_iter, std=10e-2,lr = 0.001):
    curve_internal = linear_interpolation(z_init, z_final, num_points, only_internal=False)
    curve_internal = perturb_tensor(curve_internal[1:-1], std=std).detach().clone().requires_grad_(True)
    curve_z = torch.cat((z_init.unsqueeze(0), curve_internal, z_final.unsqueeze(0)))
    curve_z_original = curve_z.detach().clone()
    optimizer = torch.optim.Adam([curve_internal], lr=lr, weight_decay=10e-6)
    
    metric_tensor_initial = pullback_metric(generator, curve_z)
    loss_initial = dirichlet_energy(metric_tensor_initial, curve_z, is_tensor=True)
    length_initial = length_of_curve(generator(curve_z))
    print(f'Epoch [{0}/{n_iter}], Energy :{loss_initial.item():.4f}, length : {length_initial}')
    for i in range(n_iter):  # iterate for 1000 epochs
        metric_tensor = pullback_metric(generator, curve_z)
        loss = dirichlet_energy(metric_tensor, curve_z, is_tensor=True)
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        curve_z = torch.cat((z_init.unsqueeze(0),curve_internal,z_final.unsqueeze(0)))
        if (i+1)%100 == 0:
            length = length_of_curve(generator(curve_z))
            print(f'Epoch [{i+1}/{n_iter}], Energy :{loss.item():.4f}, length : {length}')
    
    return curve_z, curve_z_original


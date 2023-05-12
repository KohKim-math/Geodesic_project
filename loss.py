import torch

class DeviationFromIsometry(torch.nn.Module):
    def __init__(self, function):
        super(DeviationFromIsometry, self).__init__()
        self.function = function
        
    def forward(self, z):
        bs, z_dim = z.size()

        # Hutchinson's trace estimator is used to estimate the trace
        v = torch.randn(bs, z_dim).to(z).detach()
        fz, Jv = torch.autograd.functional.jvp(self.function, z, v=v, create_graph=True) # (bs, z_dim)
        # tr((J^T)*J) = E[|Jv|^2]; sum of eigenvalues of (J^T)*J
        energy = (Jv**2).sum()

        # Hutchinson's trace estimator is used to estimate the trace
        fz, JTJv = torch.autograd.functional.vjp(self.function, z, v=Jv, create_graph=True) # (bs, z_dim)
         # tr( ((J^T)J)^2 ) = E[|(J^T)Jv|^2]; sum of eigenvalues^2 of (J^T)*J
        squared_sum_energy = (JTJv**2).sum()

        # e = (squared_sum_energy - 2*energy + (v**2).sum())/bs
        e = (squared_sum_energy - 2*energy)/bs 

        # print(f'{squared_sum_energy.grad_fn} || {energy.grad_fn} || {(v**2).sum()} || {e}')
        return e
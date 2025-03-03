import torch
from tqdm import tqdm
from models.selfdeq import UNet
from tomosipo.torch_support import to_autograd

class Block(torch.nn.Module):

    def __init__(self, op, nangles):
        super().__init__()

        self.op = op
        self.A = to_autograd(self.op, num_extra_dims=1)
        self.AT = to_autograd(self.op.T, num_extra_dims=1)

        self.nangles = nangles

        self.alpha = torch.nn.Parameter(torch.ones(1) * 0.4, requires_grad=False)
        self.gamma = torch.nn.Parameter(torch.ones(1) * 1., requires_grad=False)

        self.net = UNet(2, f_root=32, conv_times=2, up_down_times=3, is_spe_norm=True, is_residual=False)

        self.scaling = self.get_scaling()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x, y, mask):

        gradf = self.AT(mask*self.A(x)-y)
        x = x - self.scaling*self.gamma*gradf

        x = self.alpha * self.net(x.float()) + (1 - self.alpha) * x

        x = torch.clamp(x, min=0.)
        
        return x
    
    def get_scaling(self,dim_input=[16,1,336,336]):
        b_k1 = torch.rand(*dim_input).cuda()

        sino = self.A(b_k1)
        mask_temp = torch.arange(0,sino.size(-2),sino.size(-2)//self.nangles)
        mask = torch.zeros_like(sino).cuda()
        mask[:,:,mask_temp,:] = 1.

        for _ in range(10):
            b_k = b_k1
            b_k1 = self.AT(mask*self.A(b_k))
            b_k1_norm = torch.norm(b_k1)
            b_k1 = b_k1/b_k1_norm
        
        b_k2 = self.AT(mask*self.A(b_k))
        norm = (torch.sum(b_k2 * b_k1)/torch.sum(b_k1 * b_k1)).item()
        scaling = 1/norm

        return scaling
    
class DEQModel(torch.nn.Module):

    def __init__(self, op, nangles):

        super().__init__()

        self.op = op

        self.block = Block(self.op, nangles)
    
    def forward(self, x0, y, mask):

        with torch.no_grad():

            z, forward_res = anderson_solver(
                lambda x: self.block(x, y, mask), x0,
                max_iter=100,
                tol=1e-3,
            )

        z = self.block(z, y, mask)
    
        return z
    
    def save(self, path):
        path = f'{path}/model_weights.pth'
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        path = f'{path}/model_weights.pth'
        self.load_state_dict(torch.load(path))
    
    def jacobian_spectral_norm(self, x, y, mask):
        
        torch.set_grad_enabled(True)

        x.requires_grad_()
        z = self.block(x, y, mask)

        operator = lambda vec: torch.autograd.grad(z,x, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]

        lambda_estimate = power_iteration(operator, x.size(), steps=10, eps=1e-2)

        return lambda_estimate


def anderson_solver(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-4, beta=1.0, is_verbose=False):
    """ Anderson acceleration for fixed point iteration. """

    if len(x0.shape) == 5:
        bsz, d, Z, H, W = x0.shape
        X = torch.zeros(bsz, m, d * Z * H * W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d * Z * H * W, dtype=x0.dtype, device=x0.device)

    elif len(x0.shape) == 4:
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    else:
        raise NotImplementedError()

    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []

    iter_ = range(2, max_iter)
    if is_verbose:
        iter_ = tqdm(iter_, desc='anderson_solver')

    for k in iter_:
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1],y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))

        if is_verbose:
            iter_.set_description("forward_res: %.4f" % res[-1])

        if res[-1] < tol:
            break

    return X[:, k % m].view_as(x0), res

def power_iteration(operator, vector_size, steps=10, eps=1e-2):
    with torch.no_grad():
        vec = torch.rand(vector_size).cuda()

        vec /= torch.norm(vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1, 1)

        for i in range(steps):

            new_vec = operator(vec)
            new_vec = new_vec / torch.norm(new_vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1, 1)
            old_vec = vec
            vec = new_vec
            diff_vec = torch.norm(new_vec - old_vec,p=2)
            if diff_vec < eps:
                break

    new_vec = operator(vec)
    div = torch.norm(vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0])
    lambda_estimate = torch.abs(torch.sum(vec.view(vector_size[0], -1) * new_vec.view(vector_size[0], -1), dim=1)) / div

    return lambda_estimate

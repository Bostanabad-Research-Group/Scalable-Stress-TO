import torch
from torch import nn
from utils.utils_general import get_tkwargs
import matplotlib.pyplot as plt

tkwargs = get_tkwargs()
dtype = tkwargs['dtype']
device = tkwargs['device']

class Model_PGCAN(nn.Module):
    
    def __init__(self, network, encoder = None  , data_params = None) -> None:
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.data_params = data_params

    def forward(self,X) :
        x = X[: , :1]
        y = X[: , 1:]
        out = self.encoder(x, y)

        out = self.network(input = X , features = out)
        return out

class Encoder(nn.Module):
    def __init__(self, n_features = 128, res = [9], n_cells = 2, domain_size = [0, 2, 0, 1] , mode = 'cosine',kernel_size = 3, groups = 1) -> None:
        super().__init__()
        self.mode = mode
        self.n_features = n_features 
        self.res = res
        self.n_cells = n_cells
        self.groups = groups
        self.kernel_size = kernel_size

        a = [torch.rand(size=(self.n_cells, self.n_features, self.res[i], self.res[i]),
             dtype=dtype).to(device).data.uniform_(-1e-5,1e-5) for i in range(len(self.res))]
        
        self.F_active = nn.ParameterList(a)
        self.domain_size = domain_size
        self.conv_layer = nn.Conv2d(in_channels=self.n_features , out_channels=self.n_features,
                                               groups = self.groups ,kernel_size=self.kernel_size, padding=1, bias=False).to(device)
        
        #self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2).to(device)

    def forward(self, x, y):
        features  = []
        x_min, x_max, y_min, y_max = self.domain_size
        x = (x - x_min)/(x_max - x_min)
        y = (y - y_min)/(y_max - y_min)
        x = x*2-1
        y = y*2-1

        x = torch.cat([x, y], dim=-1).unsqueeze(0).unsqueeze(0)
        x = x.repeat([self.n_cells,1,1,1])

        for idx , alpha in enumerate(self.F_active):
            alpha = self.conv_layer(alpha)
            self.beta = alpha
            alpha = nn.Tanh()(alpha)
            F = grid_sample_2d(alpha, x, step=self.mode, offset=True)
            #print(f"pixel shape {F.shape}")
            dim = alpha.shape[1]
            features.append(F.sum(0).view(dim,-1).t())

        F = torch.cat(tuple(features) , 1)
        #F_ = torch.cat([F  , boundary_f] , 0)
        #print(F_.shape)
        return F#_
    
def grid_sample_2d(input, grid, step='cosine', offset=True):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW).
        grid: A torch.Tensor of dimension (N, H, W, 2).
    Return:
        torch.Tensor: The bilinearly interpolated values (N, H, W, 2).
    '''
    N, C, IH, IW = input.shape
    _, H, W, _ = grid.shape

    if step=='bilinear':
        step_f = lambda x: x
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError

    ''' (iy,ix) will be the indices of the input
            1. normalize coordinates 0 to 1 (from -1 to 1)
            2. scaling to input size
            3. adding offset to make non-zero derivative interpolation '''
    ix = grid[..., 0]
    iy = grid[..., 1]
    if offset:
        offset = torch.linspace(0,1-(1/N),N).reshape(N,1,1).to(device)
        iy = ((iy+1)/2)*(IH-2) + offset
        ix = ((ix+1)/2)*(IW-2) + offset

    else:
        iy = ((iy+1)/2)*(IH-1)
        ix = ((ix+1)/2)*(IW-1)
    
    # compute corner indices
    with torch.no_grad():
        ix_left = torch.floor(ix)
        ix_right = ix_left + 1
        iy_top = torch.floor(iy)
        iy_bottom = iy_top + 1

    # compute weights
    dx_right = step_f(ix_right-ix)
    dx_left = 1 - dx_right
    dy_bottom = step_f(iy_bottom-iy)
    dy_top = 1 - dy_bottom

    nw = dx_right*dy_bottom
    ne = dx_left*dy_bottom
    sw = dx_right*dy_top
    se = dx_left*dy_top

    # sanity checking
    with torch.no_grad():
        torch.clamp(ix_left, 0, IW-1, out=ix_left)
        torch.clamp(ix_right, 0, IW-1, out=ix_right)
        torch.clamp(iy_top, 0, IH-1, out=iy_top)
        torch.clamp(iy_bottom, 0, IH-1, out=iy_bottom)

    # look up values
    input = input.view(N, C, IH*IW)
    nw_val = torch.gather(input, 2, (iy_top * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    ne_val = torch.gather(input, 2, (iy_top * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))
    sw_val = torch.gather(input, 2, (iy_bottom * IW + ix_left).long().view(N, 1, H*W).repeat(1, C, 1))
    se_val = torch.gather(input, 2, (iy_bottom * IW + ix_right).long().view(N, 1, H*W).repeat(1, C, 1))

    # 2d_cosine/bilinear interpolation
    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

class NetworkM4_fused(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 1, layers = [40 , 40 , 40 , 40 ], activation = 'tanh') -> None:
        super(NetworkM4_fused, self).__init__()
        activation_list = {'tanh':nn.Tanh(), 'relu':nn.ReLU(), 'leaky_relu':nn.LeakyReLU(negative_slope=0.01), 'swish':nn.SiLU(), 'sigmoid':nn.Sigmoid(),'softplus':nn.Softplus(),'sin':torch.sin}
        activation = activation_list[activation]

        self.H1 = nn.Linear(input_dim, layers[0]).to(device)
        self.last= nn.Linear(layers[0], output_dim).to(device)
        
        l = nn.ModuleList()
        for i in range(len(layers)):
            l.append(nn.Linear(layers[i], layers[i]))
            l.append(activation )
        #l.append(nn.Linear(layers[-1], output_dim, bias=True))
        self.layers = nn.Sequential(*l).to(device)

    def forward(self, input , features):

        #features (N,256)
        F = int(features.shape[1]/2)

        U = features[:,:F]

        V = features[:,F:]
        
        H = nn.Tanh()(self.H1(input))

        #out = self.layers[0](input)
        for layer in self.layers:
            # Z = layer(H)
            Z = torch.sigmoid(layer(H))
            H = (1-Z)*U + Z*V

        out = self.last(H)

        return out

class Model_PGCAN3D(nn.Module): 
    def __init__(self, network, encoder = None  , data_params = None) -> None:
        super().__init__()
        self.network = network
        self.encoder = encoder
        self.data_params = data_params
        
    def forward(self, X) :
        #X = torch.cat([x, y], dim=-1)
        x = X[: , 0:1]
        y = X[: , 1:2]
        z = X[: , 2:3]
        out = self.encoder(x, y, z)
        out = self.network(input = X , features = out)
        return out

class Encoder_conv_3d(nn.Module):
    def __init__(self, 
                 n_features=4, 
                 res=[9, 9, 9], 
                 n_cells=2, 
                 q=3, 
                 domain_size=[0, 1, 0, 1, 0, 1], 
                 mode='cosine',
                 kernel_size=(3, 3, 3), 
                 groups=1,
                 dtype=torch.float32, 
                 device='cpu') -> None:
        super().__init__()
        self.n_features = n_features
        self.mode = mode
        self.res = res
        self.n_cells = n_cells
        self.groups = groups
        
        # Ensure kernel_size is a tuple of 3 ints
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        elif len(kernel_size) != 3:
            raise ValueError("kernel_size must be int or tuple of length 3")
        self.kernel_size = kernel_size
        
        # Auto padding for "same" shape
        padding = tuple(k // 2 for k in self.kernel_size)

        a = [torch.rand(size=(self.n_cells, self.n_features, 
                               self.res[0]+1, self.res[1]+1, self.res[2]+1),
                        dtype=dtype, device=device).uniform_(-1e-5, 1e-5)]
                     
        self.F_active = nn.ParameterList(a)
        self.domain_size = domain_size
        self.active = nn.Tanh()
        
        self.conv = nn.Conv3d(in_channels=self.n_features, 
                              out_channels=self.n_features,
                              kernel_size=self.kernel_size, 
                              padding=padding,
                              groups=self.groups,
                              bias=False).to(device)
        
    def forward(self, x, y ,z, inference = False):
        features = []
        x_min,x_max,y_min,y_max,z_min,z_max = self.domain_size
        x = (x - x_min)/(x_max - x_min)
        y = (y - y_min)/(y_max - y_min)
        z = (z - z_min)/(z_max - z_min)
        x = x*2-1
        y = y*2-1
        z = z*2-1

        X = torch.cat([x,y,z] , dim=-1).unsqueeze(0).unsqueeze(0).to(device)
        X3 = X.repeat([self.n_cells, 1, 1, 1])
        for idx, alpha in enumerate(self.F_active):
            #print(alpha.shape)
            alpha = self.conv(alpha)
            alpha = self.active(alpha)
            F = grid_sample_3d_revised2(alpha, X3, step=self.mode, offset=True) 
            features.append(F.sum(0)[:,0,:].t())
            
        F = torch.cat(tuple(features) , 1)
        #self.F = F
        return F
    
def grid_sample_3d_revised2(input, grid, step='cosine', offset=True):
    '''
    Args:
        input : A torch.Tensor of dimension (N, C, IH, IW, IT) in feature space
        grid: A torch.Tensor of dimension (N, 1, W, 3) for each collocation point, W is number of CPs
    Return:
        torch.Tensor: The interpolated feature values on each CP (N, C, 1, W).
    '''
    N, C, IH, IW, IT = input.shape
    IX = IH - 1
    IY = IW - 1
    IZ = IT - 1
    _, _, W, _ = grid.shape
    if step=='trilinear':
        step_f = lambda x: x
    elif step=='cosine':
        step_f = lambda x: 0.5*(1-torch.cos(torch.pi*x))
    else:
        raise NotImplementedError

    ix = grid[..., 0]  # normalized x-coord [-1,1]
    iy = grid[..., 1]  # normalized y-coord [-1,1]
    iz = grid[..., 2]  # normalized z-coord [-1,1]
   
    if offset:
        offset = torch.linspace(0,(1-(1/(N))),N).reshape(N,1,1).to(device)
        ix = ((ix+1)/2)*(IX-1) + offset  # should be -1, not -2
        iy = ((iy+1)/2)*(IY-1) + offset  # should be -1, not -2
        iz = ((iz+1)/2)*(IZ-1) + offset  # should be -1, not -2
    else:
        ix = ((ix+1)/2)*(IX-1)
        iy = ((iy+1)/2)*(IY-1)
        iz = ((iz+1)/2)*(IZ-1)

    # visualize_grids_3d(ix, iy, iz)

    with torch.no_grad():
        ix_nw_front = torch.floor(ix)
        iy_nw_front = torch.floor(iy)
        iz_nw_front = torch.floor(iz)
        
        ix_sw_front = ix_nw_front
        iy_sw_front = iy_nw_front
        iz_sw_front = iz_nw_front+1

        ix_ne_front = ix_nw_front
        iy_ne_front = iy_nw_front+1
        iz_ne_front = iz_nw_front

        ix_se_front = ix_nw_front
        iy_se_front = iy_nw_front+1
        iz_se_front = iz_nw_front+1

        ix_nw_back = ix_nw_front+1
        iy_nw_back = iy_nw_front
        iz_nw_back = iz_nw_front

        ix_ne_back = ix_nw_front+1
        iy_ne_back = iy_nw_front+1
        iz_ne_back = iz_nw_front

        ix_sw_back = ix_nw_front+1
        iy_sw_back = iy_nw_front
        iz_sw_back = iz_nw_front+1

        ix_se_back = ix_nw_front+1
        iy_se_back = iy_nw_front+1
        iz_se_back = iz_nw_front+1
    
    # # compute 3d weights
    step_ix = step_f(ix_se_back - ix)
    step_iy = step_f(iy_se_back - iy)
    step_iz = step_f(iz_se_back - iz)
    
    nw_front = step_ix * step_iy * step_iz
    ne_front = step_ix * (1-step_iy) * step_iz
    sw_front = step_ix * step_iy * (1-step_iz)
    se_front = step_ix * (1-step_iy) * (1-step_iz)

    nw_back = (1-step_ix) * step_iy * step_iz
    ne_back = (1-step_ix) * (1-step_iy) * step_iz
    sw_back = (1-step_ix) * step_iy * (1-step_iz)
    se_back = (1-step_ix) * (1-step_iy) * (1-step_iz)
    
    # sanity checking
    with torch.no_grad():
        torch.clamp(ix_nw_front, 0, IX-1, out=ix_nw_front)
        torch.clamp(iy_nw_front, 0, IY-1, out=iy_nw_front)
        torch.clamp(iz_nw_front, 0, IZ-1, out=iz_nw_front)
        
        torch.clamp(ix_ne_front, 0, IX-1, out=ix_ne_front)
        torch.clamp(iy_ne_front, 1, IY, out=iy_ne_front)
        torch.clamp(iz_ne_front, 0, IZ-1, out=iz_ne_front)
        
        torch.clamp(ix_sw_front, 0, IX-1, out=ix_sw_front)
        torch.clamp(iy_sw_front, 0, IY-1, out=iy_sw_front)
        torch.clamp(iz_sw_front, 1, IZ, out=iz_sw_front)
        
        torch.clamp(ix_se_front, 0, IX-1, out=ix_se_front)
        torch.clamp(iy_se_front, 1, IY, out=iy_se_front)
        torch.clamp(iz_se_front, 1, IZ, out=iz_se_front)

        torch.clamp(ix_nw_back, 1, IX, out=ix_nw_back)
        torch.clamp(iy_nw_back, 0, IY-1, out=iy_nw_back)
        torch.clamp(iz_nw_back, 0, IZ-1, out=iz_nw_back)
        
        torch.clamp(ix_ne_back, 1, IX, out=ix_ne_back)
        torch.clamp(iy_ne_back, 1, IY, out=iy_ne_back)
        torch.clamp(iz_ne_back, 0, IZ-1, out=iz_ne_back)
        
        torch.clamp(ix_sw_back, 1, IX, out=ix_sw_back)
        torch.clamp(iy_sw_back, 0, IY-1, out=iy_sw_back)
        torch.clamp(iz_sw_back, 1, IZ, out=iz_sw_back)
        
        torch.clamp(ix_se_back, 1, IX, out=ix_se_back)
        torch.clamp(iy_se_back, 1, IY, out=iy_se_back)
        torch.clamp(iz_se_back, 1, IZ, out=iz_se_back)
    
    input = input.view(N, C, IH*IW*IT)
    
    # compute flattened linear indices using z-fastest order: iz + IT * (iy + IW * ix)
    nw_front_idx = (iz_nw_front + IT * (iy_nw_front + IW * ix_nw_front)).long().view(N, 1, W).repeat(1, C, 1)
    ne_front_idx = (iz_ne_front + IT * (iy_ne_front + IW * ix_ne_front)).long().view(N, 1, W).repeat(1, C, 1)
    sw_front_idx = (iz_sw_front + IT * (iy_sw_front + IW * ix_sw_front)).long().view(N, 1, W).repeat(1, C, 1)
    se_front_idx = (iz_se_front + IT * (iy_se_front + IW * ix_se_front)).long().view(N, 1, W).repeat(1, C, 1)

    nw_back_idx = (iz_nw_back + IT * (iy_nw_back + IW * ix_nw_back)).long().view(N, 1, W).repeat(1, C, 1)
    ne_back_idx = (iz_ne_back + IT * (iy_ne_back + IW * ix_ne_back)).long().view(N, 1, W).repeat(1, C, 1)
    sw_back_idx = (iz_sw_back + IT * (iy_sw_back + IW * ix_sw_back)).long().view(N, 1, W).repeat(1, C, 1)
    se_back_idx = (iz_se_back + IT * (iy_se_back + IW * ix_se_back)).long().view(N, 1, W).repeat(1, C, 1)

    # gather the values at those indices
    nw_front_val = torch.gather(input, 2, nw_front_idx)
    ne_front_val = torch.gather(input, 2, ne_front_idx)
    sw_front_val = torch.gather(input, 2, sw_front_idx)
    se_front_val = torch.gather(input, 2, se_front_idx)

    nw_back_val = torch.gather(input, 2, nw_back_idx)
    ne_back_val = torch.gather(input, 2, ne_back_idx)
    sw_back_val = torch.gather(input, 2, sw_back_idx)
    se_back_val = torch.gather(input, 2, se_back_idx)

    # 3d_cosine/trilinear interpolation
    out_val = ((nw_front_val.view(N, C, 1, W) * nw_front.view(N, 1, 1, W)) + 
               (ne_front_val.view(N, C, 1, W) * ne_front.view(N, 1, 1, W)) +
               (sw_front_val.view(N, C, 1, W) * sw_front.view(N, 1, 1, W)) +
               (se_front_val.view(N, C, 1, W) * se_front.view(N, 1, 1, W)) + 
               (nw_back_val.view(N, C, 1, W) * nw_back.view(N, 1, 1, W)) + 
               (ne_back_val.view(N, C, 1, W) * ne_back.view(N, 1, 1, W)) +
               (sw_back_val.view(N, C, 1, W) * sw_back.view(N, 1, 1, W)) +
               (se_back_val.view(N, C, 1, W) * se_back.view(N, 1, 1, W))) 
    return out_val

def visualize_grids_3d(ix, iy, iz,ix_vert=None, iy_vert=None, iz_vert=None, vert_label = None):
    """
    Visualize 3D grid samples for each batch in a 3D scatter plot.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    N, _, M = ix.shape
    base_colors = plt.cm.get_cmap('tab10', N)
    
    # Plot base grid
    for i in range(N):
        ax.scatter(ix[i, 0].cpu(), iy[i, 0].cpu(), iz[i, 0].cpu(),
                   color=base_colors(i), label=f'Grid {i}', s=10, alpha=0.7)

    # Define vertex sets and their labels/colors
    vertex_sets = [
        (ix_vert, iy_vert, iz_vert, vert_label, 'red')
    ]

    for ix_v, iy_v, iz_v, label, color in vertex_sets:
        if ix_v is not None:
            for i in range(N):
                ax.scatter(ix_v[i, 0].cpu(), iy_v[i, 0].cpu(), iz_v[i, 0].cpu(),
                           color=color, label = label, s=10, alpha=0.5)

    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    ax.set_zlabel("iz")
    # ax.set_title("3D Grid Samples and Vertex Points")
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.tight_layout()
    plt.show()


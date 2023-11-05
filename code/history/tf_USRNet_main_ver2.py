from tf_USRNet_func import p2o

"""
# --------------------------------------------
# (2) Data module, closed-form solutiontorch.roll
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
#   x_{k-1} ：x_{k}的前一個值
#   s       ：
#   k       ：blur kernel
#   y       ：blur kernel
#   alpha_k ：alpha_kµ_{k}*σ^2
# some can be pre-calculated
# --------------------------------------------
"""

class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        """
        #   FB      ：論文中的式(7)的d
        #   FRFy    ：論文中的 iFFT(k)*FFT(上採樣的y)
        #   s       ：
        #   k       ：blur kernel
        #   y       ：blur kernel
        #   alpha_k ：alpha_kµ_{k}*σ^2
        # --------------------------------------------
        """
        FR = FBFy + tf.signal.fft2d(alpha*x)
        FR = FBFy + torch.fft.fftn(alpha*x, dim=(-2,-1))

        x1 = FB.mul(FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + alpha)
        FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
        FX = (FR-FCBinvWBR)/alpha
        Xest = torch.real(torch.fft.ifftn(FX, dim=(-2,-1)))

        return Xest


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x


"""
# --------------------------------------------
# main USRNet
# deep unfolding super-resolution network
# --------------------------------------------
"""        


class USRNet(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(USRNet, self).__init__()
        '''
        n_iter: unfolding 的迭代次數
        h_nc:
        in_nc:
        out_nc:
        nc:
        nb:
        act_mode:
        downsample_mode: 先驗項的下採樣模式
        upsample_mode: 先驗項的上採樣模式
        '''
        
        # closed-form solution for the data term
        self.d = DataNet()

        # denoiser for the prior term 
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
        
        # MLP for the hyper-parameter; acts as a slide bar to control the outputs of d and p
        self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
        
        self.n = n_iter

    def forward(self, x, k, sf, sigma):
        '''
        x: 輸入張量, NxCxWxH    (batch size, channel, width, high)
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''

        # initialization & pre-calculation
        w, h = x.shape[-2:]
        FB = p2o(k, (w*sf, h*sf))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        ###
        STy = upsample(x, sf=sf)
        FBFy = FBC*torch.fft.fftn(STy, dim=(-2,-1))
        x = tf.image.resize(x, size=(w*sf, h*sf), method='nearest')

        # hyper-parameter, alpha & beta
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        # unfolding
        for i in range(self.n):
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i+1, ...], sf)
            x = self.p(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1))

        return x
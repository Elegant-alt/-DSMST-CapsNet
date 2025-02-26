import torch  
import torch.nn as nn  
from torch.nn import functional as F  
from torch.autograd import Variable  


# 通道注意力模块  
class ChannelAttention(nn.Module):  
    def __init__(self, in_planes, ratio=8):  
        super(ChannelAttention, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化  
        mid_channels = max(1, in_planes // ratio)  # 中间通道数  
        self.fc = nn.Sequential(  
            nn.Conv2d(in_planes, mid_channels, 1, bias=False),  
            nn.ReLU(),  
            nn.Conv2d(mid_channels, in_planes, 1, bias=False)  
        )  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):  
        avg_out = self.fc(self.avg_pool(x))  # 平均池化后的特征  
        max_out = self.fc(self.max_pool(x))  # 最大池化后的特征  
        out = avg_out + max_out  # 特征融合  
        return self.sigmoid(out)  


# 空间注意力模块  
class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size=7):  
        super(SpatialAttention, self).__init__()  
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):  
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化  
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接  
        x = self.conv1(x)  # 卷积操作  
        return self.sigmoid(x)  


# CBAM 模块（通道注意力 + 空间注意力）  
class CBAM(nn.Module):  
    def __init__(self, in_planes, ratio=8, kernel_size=7):  
        super(CBAM, self).__init__()  
        self.channel_attention = ChannelAttention(in_planes, ratio)  
        self.spatial_attention = SpatialAttention(kernel_size)  

    def forward(self, x):  
        x = x * self.channel_attention(x)  # 通道注意力  
        x = x * self.spatial_attention(x)  # 空间注意力  
        return x  


# Primary Capsule（初级胶囊网络）  
class PrimaryCapsule(nn.Module):  
    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0):  
        super(PrimaryCapsule, self).__init__()  
        self.dim_caps = dim_caps  
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)  

    def forward(self, x):  
        outputs = self.conv2d(x)  # 卷积操作  
        outputs = outputs.view(x.size(0), -1, self.dim_caps)  # 调整形状为胶囊形式  
        return squash(outputs)  # 激活函数  


# 胶囊网络的 squash 激活函数  
def squash(inputs, axis=-1):  
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)  # 计算向量的 L2 范数  
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)  # 缩放因子  
    return scale * inputs  


# Dense Capsule（密集胶囊网络）  
class DenseCapsule(nn.Module):  
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):  
        super(DenseCapsule, self).__init__()  
        self.in_num_caps = in_num_caps  
        self.in_dim_caps = in_dim_caps  
        self.out_num_caps = out_num_caps  
        self.out_dim_caps = out_dim_caps  
        self.routings = routings  
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))  

    def forward(self, x):  
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)  # 预测向量  
        x_hat_detached = x_hat.detach()  # 分离梯度  
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()  # 路由系数初始化  

        assert self.routings > 0, 'The \'routings\' should be > 0.'  
        for i in range(self.routings):  
            c = F.softmax(b, dim=1)  # 计算权重  
            if i == self.routings - 1:  
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))  # 最终输出  
            else:  
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))  
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)  # 更新路由系数  

        return torch.squeeze(outputs, dim=-2)  


# TSception 模型  
class MOdel(nn.Module):  
    def conv_block(self, in_chan, out_chan, kernel, step, pool):  
        return nn.Sequential(  
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=step, padding=0),  
            nn.LeakyReLU(),  
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool))  
        )  

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):  
        super(TSception, self).__init__()  
        self.inception_window = [1.0, 0.50, 0.25]  
        self.pool = 8  

        # 时间特征提取模块  
        self.TBlock1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)  
        self.TBlock2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)  
        self.TBlock3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)  

        # 空间特征提取模块  
        self.Sception1 = self.conv_block(1, num_S, (int(input_size[1]), 1), 1, int(self.pool * 0.25))  
        self.Sception2 = self.conv_block(1, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1), int(self.pool * 0.25))  
        self.Sception3 = self.conv_block(  
            1, num_S, (int(input_size[1] * 0.25), 1), (int(input_size[1] * 0.25), 1), int(self.pool * 0.25)  
        )  

        # 注意力机制  
        self.cbam_spatial = CBAM(num_S)  

        # 胶囊网络  
        self.primary = PrimaryCapsule(in_channels=15, out_channels=64, dim_caps=8, kernel_size=3, stride=1, padding=1)  
        self.dense = DenseCapsule(in_num_caps=4592, in_dim_caps=8, out_num_caps=2, out_dim_caps=16, routings=3)  

        # 批归一化  
        self.BN_t = nn.BatchNorm2d(num_T)  
        self.BN_s = nn.BatchNorm2d(num_S)  

        # 全连接层  
        self.fc = nn.Sequential(  
            nn.Linear(num_S, hidden),  
            nn.ReLU(),  
            nn.Dropout(dropout_rate),  
            nn.Linear(hidden, num_classes)  
        )  

    def forward(self, x):  
        print("x", x.shape)  

        # --- 时间特征提取 ---  
        t1 = self.TBlock1(x)  
        print("t1", t1.shape)
        t2 = self.TBlock2(x)  
        print("t2", t2.shape)
        t3 = self.TBlock3(x)  
        print("t3", t3.shape)
        time_features = torch.cat((t1, t2, t3), dim=-1)  
        time_features = self.BN_t(time_features)  
        print ("time_features",time_features.shape)


        # --- 空间特征提取 ---  
        s1 = self.Sception1(x)  
        s2 = self.Sception2(x)  
        s3 = self.Sception3(x)  
        space_features = torch.cat((s1, s2, s3), dim=2)  
        space_features = self.BN_s(space_features)  
        space_features = self.cbam_spatial(space_features)  
        print ("space_features",space_features.shape)

        # --- 融合特征 ---  
        combined_features = torch.cat((time_features, space_features), dim=1)  

        # --- 胶囊网络 ---  
        out = self.primary(combined_features)  
        out = self.dense(out)  

        # --- 平均化胶囊特征 ---  
        out = out.mean(dim=-1)  

        return out

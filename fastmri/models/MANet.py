import torch
import torch.nn as nn
import torch.nn.functional as F


class Single_level_densenet(nn.Module):
    def __init__(self,filters, num_conv = 4):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters,filters,3, padding = 1))
            self.bn_list.append(nn.BatchNorm2d(filters))
            
    def forward(self,x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final
    
class Down_sample(nn.Module):
    def __init__(self,kernel_size = 2, stride = 2):
        super(Down_sample, self).__init__()
        self.down_sample_layer = nn.MaxPool2d(kernel_size, stride)
    
    def forward(self,x):
        y = self.down_sample_layer(x)
        return y,x

class Upsample_n_Concat_1(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_1, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_2(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_2, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_3(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_3, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_4(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_4, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_T1(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_T1, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(filters,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x):
        x = self.upsample_layer(x)
        x = F.relu(self.bn(self.conv(x)))
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)    

class Dense_Unet_k(nn.Module):
    def __init__(self, in_chan, out_chan, filters, num_conv = 4):  #64   256
        super(Dense_Unet_k, self).__init__()
        self.conv1T1 = nn.Conv2d(in_chan,filters,1)
        self.conv1T2 = nn.Conv2d(4,filters,1)
        self.convdemD0 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemD1 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemD2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemD3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemU0 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemU1 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemU2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.convdemU3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        

        self.dT1_1 = Single_level_densenet(filters,num_conv )
        self.downT1_1 = Down_sample()
        self.dT1_2 = Single_level_densenet(filters,num_conv )
        self.downT1_2 = Down_sample()
        self.dT1_3 = Single_level_densenet(filters,num_conv )
        self.downT1_3 = Down_sample()
        self.dT1_4 = Single_level_densenet(filters,num_conv )
        self.downT1_4 = Down_sample()

        self.dT2_1 = Single_level_densenet(filters,num_conv )
        self.downT2_1 = Down_sample()
        self.dT2_2 = Single_level_densenet(filters,num_conv )
        self.downT2_2 = Down_sample()
        self.dT2_3 = Single_level_densenet(filters,num_conv )
        self.downT2_3 = Down_sample()
        self.dT2_4 = Single_level_densenet(filters,num_conv )
        self.downT2_4 = Down_sample()

        self.bottom_T1 = Single_level_densenet(filters,num_conv )
        self.bottom_T2 = Single_level_densenet(filters,num_conv )

        self.up4_T1 = Upsample_n_Concat_T1(filters)
        self.u4_T1 = Single_level_densenet(filters,num_conv )
        self.up3_T1 = Upsample_n_Concat_T1(filters)
        self.u3_T1 = Single_level_densenet(filters,num_conv )
        self.up2_T1 = Upsample_n_Concat_T1(filters)
        self.u2_T1 = Single_level_densenet(filters,num_conv )
        self.up1_T1 = Upsample_n_Concat_T1(filters)
        self.u1_T1 = Single_level_densenet(filters,num_conv )

        self.up4_T2 = Upsample_n_Concat_1(filters)
        self.u4_T2 = Single_level_densenet(filters,num_conv )
        self.up3_T2 = Upsample_n_Concat_2(filters)
        self.u3_T2 = Single_level_densenet(filters,num_conv )
        self.up2_T2 = Upsample_n_Concat_3(filters)
        self.u2_T2 = Single_level_densenet(filters,num_conv )
        self.up1_T2 = Upsample_n_Concat_4(filters)
        self.u1_T2 = Single_level_densenet(filters,num_conv )

        self.outconvT1 = nn.Conv2d(filters,out_chan, 1)
        self.outconvT2 = nn.Conv2d(64,out_chan, 1)

        self.atten_depth_channel_0=ChannelAttention(64)
        self.atten_depth_channel_1=ChannelAttention(64)
        self.atten_depth_channel_2=ChannelAttention(64)
        self.atten_depth_channel_3=ChannelAttention(64)

        self.atten_depth_channel_U_0=ChannelAttention(64)
        self.atten_depth_channel_U_1=ChannelAttention(64)
        self.atten_depth_channel_U_2=ChannelAttention(64)
        self.atten_depth_channel_U_3=ChannelAttention(64)

        self.atten_depth_spatial_0=SpatialAttention()
        self.atten_depth_spatial_1=SpatialAttention()
        self.atten_depth_spatial_2=SpatialAttention()
        self.atten_depth_spatial_3=SpatialAttention()

        self.atten_depth_spatial_U_0=SpatialAttention()
        self.atten_depth_spatial_U_1=SpatialAttention()
        self.atten_depth_spatial_U_2=SpatialAttention()
        self.atten_depth_spatial_U_3=SpatialAttention()



        
        
    def forward(self,T1, T2):

        T1_x1 = self.conv1T1(T1)
        T2 = torch.cat((T2,T1),dim=1)
        T2_x1 = self.conv1T2(T2)


        T1_x2,T1_y1 = self.downT1_1(self.dT1_1(T1_x1))
        T2_x2,T2_y1 = self.downT2_1(self.dT2_1(T2_x1))
        temp = T1_x2.mul(self.atten_depth_channel_0(T1_x2))
        temp = temp.mul(self.atten_depth_spatial_0(temp))
        T12_x2 = T2_x2.mul(temp)+T2_x2


        T1_x3,T1_y2 = self.downT1_1(self.dT1_2(T1_x2))
        T2_x3,T2_y2 = self.downT2_1(self.dT2_2(T12_x2))
        temp = T1_x3.mul(self.atten_depth_channel_1(T1_x3))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        T12_x3 = T2_x3.mul(temp)+T2_x3



        T1_x4,T1_y3 = self.downT1_1(self.dT1_3(T1_x3))
        T2_x4,T2_y3 = self.downT2_1(self.dT2_3(T12_x3))
        temp = T1_x4.mul(self.atten_depth_channel_2(T1_x4))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        T12_x4 = T2_x4.mul(temp)+T2_x4        




        T1_x5,T1_y4 = self.downT1_1(self.dT1_4(T1_x4))
        T2_x5,T2_y4 = self.downT2_1(self.dT2_4(T12_x4))
        temp = T1_x5.mul(self.atten_depth_channel_3(T1_x5))
        temp = temp.mul(self.atten_depth_spatial_3(temp))
        T12_x5 = T2_x5.mul(temp)+T2_x5   


        T1_x = self.bottom_T1(T1_x5)
        T2_x = self.bottom_T2(T12_x5)


        T1_1x = self.u4_T1(self.up4_T1(T1_x))
        T2_1x = self.u4_T2(self.up4_T2(T2_x,T2_y4))
        temp = T1_1x.mul(self.atten_depth_channel_U_0(T1_1x))
        temp = temp.mul(self.atten_depth_spatial_U_0(temp))
        T12_x = T2_1x.mul(temp)+T2_1x   



        T1_2x = self.u3_T1(self.up3_T1(T1_1x))
        T2_2x = self.u3_T2(self.up3_T2(T12_x,T2_y3))
        temp = T1_2x.mul(self.atten_depth_channel_U_1(T1_2x))
        temp = temp.mul(self.atten_depth_spatial_U_1(temp))
        T12_x = T2_2x.mul(temp)+T2_2x   

        T1_3x = self.u2_T1(self.up2_T1(T1_2x))
        T2_3x = self.u2_T2(self.up2_T2(T12_x,T2_y2))
        temp = T1_3x.mul(self.atten_depth_channel_U_2(T1_3x))
        temp = temp.mul(self.atten_depth_spatial_U_2(temp))
        T12_x = T2_3x.mul(temp)+T2_3x    

        T1_4x = self.u1_T1(self.up1_T1(T1_3x))
        T2_4x = self.u1_T2(self.up1_T2(T12_x,T2_y1))
        temp = T1_4x.mul(self.atten_depth_channel_U_3(T1_4x))
        temp = temp.mul(self.atten_depth_spatial_U_3(temp))
        T12_x = T2_4x.mul(temp)+T2_4x   

        T1 = self.outconvT1(T1_4x)
        T2 = self.outconvT2(T12_x)
        
        return T1,T2


class Dense_Unet_img(nn.Module):
    def __init__(self, in_chan, out_chan, filters, num_conv = 4):  
        super(Dense_Unet_img, self).__init__()
        self.conv1T1 = nn.Conv2d(in_chan,filters,1)
        self.conv1T2 = nn.Conv2d(2,filters,1)

        self.dT1_1 = Single_level_densenet_img(filters,num_conv )
        self.downT1_1 = Down_sample_img()
        self.dT1_2 = Single_level_densenet_img(filters,num_conv )
        self.downT1_2 = Down_sample_img()
        self.dT1_3 = Single_level_densenet_img(filters,num_conv )
        self.downT1_3 = Down_sample_img()
        self.dT1_4 = Single_level_densenet_img(filters,num_conv )
        self.downT1_4 = Down_sample_img()

        self.dT2_1 = Single_level_densenet_img(filters,num_conv )
        self.downT2_1 = Down_sample_img()
        self.dT2_2 = Single_level_densenet_img(filters,num_conv )
        self.downT2_2 = Down_sample_img()
        self.dT2_3 = Single_level_densenet_img(filters,num_conv )
        self.downT2_3 = Down_sample_img()
        self.dT2_4 = Single_level_densenet_img(filters,num_conv )
        self.downT2_4 = Down_sample_img()

        self.bottom_T1 = Single_level_densenet_img(filters,num_conv )
        self.bottom_T2 = Single_level_densenet_img(filters,num_conv )

        self.up4_T1 = Upsample_n_Concat_T1_img(filters)
        self.u4_T1 = Single_level_densenet_img(filters,num_conv )
        self.up3_T1 = Upsample_n_Concat_T1_img(filters)
        self.u3_T1 = Single_level_densenet_img(filters,num_conv )
        self.up2_T1 = Upsample_n_Concat_T1_img(filters)
        self.u2_T1 = Single_level_densenet_img(filters,num_conv )
        self.up1_T1 = Upsample_n_Concat_T1_img(filters)
        self.u1_T1 = Single_level_densenet_img(filters,num_conv )

        self.up4_T2 = Upsample_n_Concat_1_img(filters)
        self.u4_T2 = Single_level_densenet_img(filters,num_conv )
        self.up3_T2 = Upsample_n_Concat_2_img(filters)
        self.u3_T2 = Single_level_densenet_img(filters,num_conv )
        self.up2_T2 = Upsample_n_Concat_3_img(filters)
        self.u2_T2 = Single_level_densenet_img(filters,num_conv )
        self.up1_T2 = Upsample_n_Concat_4_img(filters)
        self.u1_T2 = Single_level_densenet_img(filters,num_conv )

        self.outconvT1 = nn.Conv2d(filters,out_chan, 1)
        self.outconvT2 = nn.Conv2d(64,out_chan, 1)
        #Components of DEM module
        self.atten_depth_channel_0=ChannelAttention_img(64)
        self.atten_depth_channel_1=ChannelAttention_img(64)
        self.atten_depth_channel_2=ChannelAttention_img(64)
        self.atten_depth_channel_3=ChannelAttention_img(64)

        self.atten_depth_channel_U_0=ChannelAttention_img(64)
        self.atten_depth_channel_U_1=ChannelAttention_img(64)
        self.atten_depth_channel_U_2=ChannelAttention_img(64)
        self.atten_depth_channel_U_3=ChannelAttention_img(64)

        self.atten_depth_spatial_0=SpatialAttention_img()
        self.atten_depth_spatial_1=SpatialAttention_img()
        self.atten_depth_spatial_2=SpatialAttention_img()
        self.atten_depth_spatial_3=SpatialAttention_img()

        self.atten_depth_spatial_U_0=SpatialAttention_img()
        self.atten_depth_spatial_U_1=SpatialAttention_img()
        self.atten_depth_spatial_U_2=SpatialAttention_img()
        self.atten_depth_spatial_U_3=SpatialAttention_img()



        
        
    def forward(self,T1, T2):

        T1_x1 = self.conv1T1(T1)
        T2 = torch.cat((T2,T1),dim=1)
        T2_x1 = self.conv1T2(T2)

        T1_x2,T1_y1 = self.downT1_1(self.dT1_1(T1_x1))
        T2_x2,T2_y1 = self.downT2_1(self.dT2_1(T2_x1))
        temp = T1_x2.mul(self.atten_depth_channel_0(T1_x2))
        temp = temp.mul(self.atten_depth_spatial_0(temp))
        T12_x2 = T2_x2.mul(temp)+T2_x2


        T1_x3,T1_y2 = self.downT1_1(self.dT1_2(T1_x2))
        T2_x3,T2_y2 = self.downT2_1(self.dT2_2(T12_x2))
        temp = T1_x3.mul(self.atten_depth_channel_1(T1_x3))
        temp = temp.mul(self.atten_depth_spatial_1(temp))
        T12_x3 = T2_x3.mul(temp)+T2_x3

        T1_x4,T1_y3 = self.downT1_1(self.dT1_3(T1_x3))
        T2_x4,T2_y3 = self.downT2_1(self.dT2_3(T12_x3))
        temp = T1_x4.mul(self.atten_depth_channel_2(T1_x4))
        temp = temp.mul(self.atten_depth_spatial_2(temp))
        T12_x4 = T2_x4.mul(temp)+T2_x4        


        T1_x5,T1_y4 = self.downT1_1(self.dT1_4(T1_x4))
        T2_x5,T2_y4 = self.downT2_1(self.dT2_4(T12_x4))
        temp = T1_x5.mul(self.atten_depth_channel_3(T1_x5))
        temp = temp.mul(self.atten_depth_spatial_3(temp))
        T12_x5 = T2_x5.mul(temp)+T2_x5 


        T1_x = self.bottom_T1(T1_x5)
        T2_x = self.bottom_T2(T12_x5)


        T1_1x = self.u4_T1(self.up4_T1(T1_x))
        T2_1x = self.u4_T2(self.up4_T2(T2_x,T2_y4))
        temp = T1_1x.mul(self.atten_depth_channel_U_0(T1_1x))
        temp = temp.mul(self.atten_depth_spatial_U_0(temp))
        T12_x = T2_1x.mul(temp)+T2_1x

        T1_2x = self.u3_T1(self.up3_T1(T1_1x))
        T2_2x = self.u3_T2(self.up3_T2(T12_x,T2_y3))
        temp = T1_2x.mul(self.atten_depth_channel_U_1(T1_2x))
        temp = temp.mul(self.atten_depth_spatial_U_1(temp))
        T12_x = T2_2x.mul(temp)+T2_2x  


        T1_3x = self.u2_T1(self.up2_T1(T1_2x))
        T2_3x = self.u2_T2(self.up2_T2(T12_x,T2_y2))
        temp = T1_3x.mul(self.atten_depth_channel_U_2(T1_3x))
        temp = temp.mul(self.atten_depth_spatial_U_2(temp))
        T12_x = T2_3x.mul(temp)+T2_3x  

        T1_4x = self.u1_T1(self.up1_T1(T1_3x))
        T2_4x = self.u1_T2(self.up1_T2(T12_x,T2_y1))
        temp = T1_4x.mul(self.atten_depth_channel_U_3(T1_4x))
        temp = temp.mul(self.atten_depth_spatial_U_3(temp))
        T12_x = T2_4x.mul(temp)+T2_4x  

        T1 = self.outconvT1(T1_4x)
        T2 = self.outconvT2(T12_x)
        
        return T1,T2

class Single_level_densenet_img(nn.Module):
    def __init__(self,filters, num_conv = 4):
        super(Single_level_densenet_img, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters,filters,3, padding = 1))
            self.bn_list.append(nn.BatchNorm2d(filters))
            
    def forward(self,x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final
    
class Down_sample_img(nn.Module):
    def __init__(self,kernel_size = 2, stride = 2):
        super(Down_sample_img, self).__init__()
        self.down_sample_layer = nn.MaxPool2d(kernel_size, stride)
    
    def forward(self,x):
        y = self.down_sample_layer(x)
        return y,x

class Upsample_n_Concat_1_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_1_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(filters*2,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_2_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_2_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_3_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_3_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_4_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_4_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(64, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(128,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x,y):
        x = self.upsample_layer(x)
        x = torch.cat([x,y],dim = 1)
        x = F.relu(self.bn(self.conv(x)))
        return x

class Upsample_n_Concat_T1_img(nn.Module):
    def __init__(self,filters):
        super(Upsample_n_Concat_T1_img, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding = 1, stride = 2)
        self.conv = nn.Conv2d(filters,filters,3, padding = 1)
        self.bn = nn.BatchNorm2d(filters)
    
    def forward(self,x):
        x = self.upsample_layer(x)
        x = F.relu(self.bn(self.conv(x)))
        return x
class ChannelAttention_img(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_img, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention_img(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_img, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)

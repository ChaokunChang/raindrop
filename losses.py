import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.models as models
class GeneratorLoss(nn.Module):

    def __init__(self):
        super(GeneratorLoss,self).__init__()
        self.vgg_model=models.vgg16(pretrained=True)#.features[:28]	# 其实就是定位到第28层，对照着上面的key看就可以理解
        self.vgg_model=model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
        self.vgg_model.cuda()	# 将模型从CPU发送到GPU,如果没有GPU则删除该行

    def forward(self, mask_list, mask_label, out_label,skip_list):
        rnn_loss = torch.zeros(1)
        n = len(mask_list)
        for id,attention_map in enumerate(mask_list):
            loss_func = nn.MSELoss()
            mse_loss = math.pow(0.8,n-id+1) * loss_func(attention_map,mask_label)
            rnn_loss = rnn_loss + mse_loss
            
        lm_loss = torch.zeros(1)
        lambda_list = [0.6,0.8,1.0]
        _,height,width,_ = out_label.size()
        label_4 = F.interpolate(out_label,size=(int(height/4),int(width/4) ))
        label_2 = F.interpolate(out_label,size=(int(height/2),int(width/2) ))
        label_list = [label_4,label_2,out_label]
        for id,skip in enumerate(skip_list):
            loss_func = nn.MSELoss()
            mse_loss = loss_func(skip,label_list[id]*lambda_list[id])
            lm_loss = lm_loss + mse_loss
        
        # compute lp_loss
        
        self.vgg_model.eval()
        src_vgg_feats = self.vgg_model(out_label)
        pred_vgg_feats = self.vgg_model(skip_list[-1])
        lp_losses = []
        for i,feats in enumerate(src_vgg_feats):
            loss_func = nn.MSELoss()
            lp_losses.append(loss_func(src_vgg_feats[id], pred_vgg_feats[id]))
        lp_loss = torch.mean(lp_losses)
        loss = torch.add(lp_loss,lm_loss,rnn_loss)
        return loss,(rnn_loss,lm_loss,lp_loss)
        

class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, mask_o,out_o,mask_r,out_r,final_mask):
        zero_mask = torch.zeros(mask_r.size())
        loss_func = nn.MSELoss()
        l_map = loss_func(final_mask,mask_o) + loss_func(mask_r,zero_mask)
        entropy_loss = -torch.log(out_r) -torch.log(-torch.sub(out_o, 1.0))
        entropy_loss = torch.mean(entropy_loss)
        loss = entropy_loss + 0.05 * l_map
        return loss,l_map
        
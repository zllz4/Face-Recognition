import math
import torch

class SoftmaxLoss(torch.nn.Module):
    def __init__(self, features, num_classes, log=True, **kargs):
        '''
            Args:
                features: features extracted by neural network
                num_classes: number of classes
                log: if true the layer will output some log data including the cos_in, cos_out and theta

        '''
        super(SoftmaxLoss, self).__init__(**kargs)
        self.features = features
        self.num_classes = num_classes
        self.log = log
        self.weight = torch.nn.Parameter(torch.zeros(num_classes, features))
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # default first dim is regarded as fan_out and second dim is fan_in
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels):

        outs = inputs @ self.weight.t()

        # if not self.train:
        #     return outs

        loss = self.loss_fn(outs, labels)

        if self.log:
            # value to trace
            inputs_norm = torch.nn.functional.normalize(inputs)
            weight_norm = torch.nn.functional.normalize(self.weight)
            cos_in = torch.clamp(inputs_norm @ weight_norm.t(), -1+1e-6, 1-1e-6) # cos: batch_size x num_classes
            
            target_cos_in = cos_in.gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_theta_in = torch.acos(cos_in).gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_cos_out = cos_in.gather(1, labels.view(-1,1)).cpu().detach().clone()

            return loss, (outs, target_cos_in, target_theta_in, target_cos_out)
        else:
            return loss
    def __repr__(self):
        return f"SoftmaxLoss(features={self.features} num_class={self.num_classes} log={self.log})"   

class ArcLoss(torch.nn.Module):
    def __init__(self, features, num_classes, s=32, m=0.5, log=True, **kargs):
        '''
            Args:
                features: features extracted by neural network
                num_classes: number of classes
                s: scale, default is 30
                m: arcface margin, default is 0.5
                log: if true the layer will output some log data including the cos_in, cos_out and theta
        '''
        super(ArcLoss, self).__init__(**kargs)
        self.margin = m
        self.scale = s
        self.log = log
        self.features = features
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.zeros(num_classes, features))
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # # value to trace
        # self.target_cos_in = None
        # self.target_cos_out = None
        # self.target_theta = None

        # self.th = math.cos(math.pi-m)
        self.mm = m*math.sin(math.pi-m)

        # default first dim is regarded as fan_out and second dim is fan_in
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels=None):
        # # l2-norm 
        inputs_norm = torch.nn.functional.normalize(inputs)
        weight_norm = torch.nn.functional.normalize(self.weight)
        
        # cos(theta)
        cos_in = torch.clamp(inputs_norm @ weight_norm.t(), -1+1e-6, 1-1e-6) # cos: batch_size x num_classes

        # cos(theta) -> cos(theta+m)
        theta_in = torch.acos(cos_in) # 0 < theta < pi, theta can't be zero, otherwise the grad will be nan
            
        # self.label_theta = theta.gather(1, self.label_debug.view(-1,1))
        label_one_hot = torch.zeros(*cos_in.size(), device=cos_in.device).scatter_(1, labels.view(-1, 1), 1)
        theta_out = theta_in + self.margin * label_one_hot
        cos_out = torch.where(theta_out < math.pi, torch.cos(theta_out), cos_in-self.mm * label_one_hot)

        

        # cos(theta+m) -> s*cos(theta+m)
        outs = self.scale * cos_out

        # if not self.train:
        #     return outs

        loss = self.loss_fn(outs, labels)

        if self.log:
            target_cos_in = cos_in.gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_theta_in = theta_in.gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_cos_out = cos_out.gather(1, labels.view(-1,1)).cpu().detach().clone()
            return loss, (outs, target_cos_in, target_theta_in, target_cos_out)
        else:
            return loss

    def __repr__(self):
        return f"ArcLoss(scale={self.scale} margin={self.margin} features={self.features} num_class={self.num_classes} log={self.log})"

class CosLoss(torch.nn.Module):
    def __init__(self, features, num_classes, s=32, m=0.35, log=True, **kargs):
        '''
            Args:
                features: features extracted by neural network
                num_classes: number of classes
                s: scale, default is 30
                m: cosface margin, default is 0.4
                log: if true the layer will output some log data including the cos_in, cos_out and theta
        '''
        super(CosLoss, self).__init__(**kargs)
        self.margin = m
        self.scale = s
        self.features = features
        self.num_classes = num_classes
        self.log = log
        self.weight = torch.nn.Parameter(torch.zeros(num_classes, features))
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # default first dim is regarded as fan_out and second dim is fan_in
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels):
        # # l2-norm 
        inputs_norm = torch.nn.functional.normalize(inputs)
        weight_norm = torch.nn.functional.normalize(self.weight)
        
        # cos(theta)
        cos_in = torch.clamp(inputs_norm @ weight_norm.t(), -1+1e-6, 1-1e-6) # cos: batch_size x num_classes
        
        # cos(theta) -> cos(theta)-m
        label_one_hot = torch.zeros(*cos_in.size(), device=cos_in.device).scatter_(1, labels.view(-1, 1), 1)
        cos_out = cos_in - self.margin * label_one_hot

        # cos(theta)-m -> s*(cos(theta)-m)
        outs = self.scale * cos_out

        # if not self.train:
        #     return outs

        loss = self.loss_fn(outs, labels)

        if self.log:
            target_cos_in = cos_in.gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_theta_in = torch.acos(cos_in).gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_cos_out = cos_out.gather(1, labels.view(-1,1)).cpu().detach().clone()
            return loss, (outs, target_cos_in, target_theta_in, target_cos_out)
        else:
            return loss

    def __repr__(self):
        return f"CosLoss(scale={self.scale} margin={self.margin} features={self.features} num_class={self.num_classes} log={self.log})"

class SphereLoss(torch.nn.Module):
    r"""
        Copy from https://github.com/ronghuaiyang/arcface-pytorch
        TODO: 实现自己的代码
        Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        log: if true the layer will output some log data including the cos_in, cos_out and theta
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4, log=True):
        super(SphereLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.log = log

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        F = torch.nn.functional
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta

        output *= NormOfFeature.view(-1, 1)

        # if not self.train:
        #     return output

        loss = self.loss_fn(output, label)

        if self.log:
            inputs_norm = torch.nn.functional.normalize(input)
            weight_norm = torch.nn.functional.normalize(self.weight)
            cos_in = torch.clamp(inputs_norm @ weight_norm.t(), -1+1e-6, 1-1e-6) # cos: batch_size x num_classes
            target_cos_in = cos_in.gather(1, label.view(-1,1)).cpu().detach().clone()
            target_theta_in = torch.acos(cos_in).gather(1, label.view(-1,1)).cpu().detach().clone()
            target_cos_out = output.gather(1, label.view(-1,1)).cpu().detach().clone()

            return loss, (output, target_cos_in, target_theta_in, target_cos_out)
        else:
            return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
                + 'in_features=' + str(self.in_features) \
                + ', out_features=' + str(self.out_features) \
                + ', m=' + str(self.m) + + ', log=' + str(self.log) + ')'

class MixLoss(torch.nn.Module):
    def __init__(self, features, num_classes, s=32, m1=1, m2=0, m3=0, log=True, **kargs):
        '''
            Args:
                features: features extracted by neural network
                num_classes: number of classes
                s: scale, default is 30
                m1,m2,m3: cos(θ) -> cos(m1θ+m2)-m3
                log: if true the layer will output some log data including the cos_in, cos_out and theta
        '''
        super(MixLoss, self).__init__(**kargs)
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.scale = s
        self.features = features
        self.num_classes = num_classes
        self.log = log
        self.weight = torch.nn.Parameter(torch.zeros(num_classes, features))
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # self.th = math.cos(math.pi-m)
        self.mm = m2*math.sin(math.pi-m2)

        # default first dim is regarded as fan_out and second dim is fan_in
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels=None):
        # # l2-norm 
        inputs_norm = torch.nn.functional.normalize(inputs)
        weight_norm = torch.nn.functional.normalize(self.weight)
        
        # cos(theta)
        cos_in = torch.clamp(inputs_norm @ weight_norm.t(), -1+1e-6, 1-1e-6) # cos: batch_size x num_classes

        # cos(theta) -> cos(theta+m)
        theta_in = torch.acos(cos_in) # 0 < theta < pi, theta can't be zero, otherwise the grad will be nan
        
        # self.label_theta = theta.gather(1, self.label_debug.view(-1,1))
        label_one_hot = torch.zeros(*cos_in.size(), device=cos_in.device).scatter_(1, labels.view(-1, 1), 1)
        theta_out = theta_in + (theta_in * (self.m1-1) + self.m2) * label_one_hot
        cos_out = torch.where(theta_out < math.pi, torch.cos(theta_out)-self.m3 * label_one_hot, cos_in-self.mm * label_one_hot-self.m3 * label_one_hot)

        # cos(theta+m) -> s*cos(theta+m)
        outs = self.scale * cos_out

        # if not self.train:
        #     return outs

        loss = self.loss_fn(outs, labels)

        if self.log:  
            target_cos_in = cos_in.gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_theta_in = theta_in.gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_cos_out = cos_out.gather(1, labels.view(-1,1)).cpu().detach().clone()
            return loss, (outs, target_cos_in, target_theta_in, target_cos_out)
        else:
            return loss

    def __repr__(self):
        return f"MixLoss(scale={self.scale} m1={self.m1} m2={self.m2} m3={self.m3} features={self.features} num_class={self.num_classes} los={self.log})"

class NormSoftmaxLoss(torch.nn.Module):
    def __init__(self, features, num_classes, s=30, log=True, **kargs):
        '''
            Args:
                features: features extracted by neural network
                num_classes: number of classes
                s: norm output -> s * norm output
                log: if true the layer will output some log data including the cos_in, cos_out and theta
        '''
        super(NormSoftmaxLoss, self).__init__(**kargs)
        self.scale = s
        self.features = features
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.zeros(num_classes, features))
        self.log = log
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # default first dim is regarded as fan_out and second dim is fan_in
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, inputs, labels):
        # # l2-norm 
        inputs_norm = torch.nn.functional.normalize(inputs)
        weight_norm = torch.nn.functional.normalize(self.weight)
        
        # cos(theta)
        cos = torch.clamp(inputs_norm @ weight_norm.t(), -1+1e-6, 1-1e-6) # cos: batch_size x num_classes

        # s*cos(theta)
        outs = self.scale * cos

        # if not self.train:
        #     return outs
            
        loss = self.loss_fn(outs, labels)

        if self.log:
            target_cos_in = cos.gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_theta_in = torch.acos(cos).gather(1, labels.view(-1,1)).cpu().detach().clone()
            target_cos_out = cos.gather(1, labels.view(-1,1)).cpu().detach().clone()
            return loss, (outs, target_cos_in, target_theta_in, target_cos_out)
        else:
            return loss

    def __repr__(self):
        return f"NormSoftmaxLoss(scale={self.scale} features={self.features} num_class={self.num_classes} log={self.log})"
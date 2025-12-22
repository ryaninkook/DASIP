
# t = t * (upper_bound - lower_bound) + lower_bound
t=t.unsqueeze(-1).unsqueeze(-1)
s=s.unsqueeze(-1).unsqueeze(-1)
# if self.path_type=='direct_distill':
xs=x0*(1-s) + nactions*(s)
x_st= (1-t+s)*xs +(t-s)*self.model(xs,s,global_cond,t)

self.DT=0.001

dt = torch.tensor(self.DT)

gradt=1/2*(self.model(xs,s,global_cond,t+dt)-self.model(xs,s,global_cond,t-dt))/self.DT


dXst_dt=-xs+self.model(xs,s,global_cond,t)+(t-s)*gradt
# global pretrained_ema_policy
with torch.no_grad():
    dxst_dt_hat=pretrained_ema_policy.model(x_st,t*int(self.T-1),global_cond)

mse_loss=F.mse_loss(dxst_dt_hat,dXst_dt)
mse_loss=mse_loss.mean()







model = Diffusion()

x0= rand
t=
xt= (1-t)*x0+t*x1

# dxdt_true = -x0 +x1
dxdt_true = model(xt,t)

dxdt_hat=model2(xt,t)

loss=mse(dxdt_true,dxdt_hat)



x=rand
for t in range(100):
    dxdt_hat=model2(x,t/100)
    x=x+dxdt_hat*(1/100)




## 

teacher=model
new_model=model



batch ={
'img1'=3,128,128
'img2'=3,128,128

}

batch {
    'reg'= 256,256,3
}
import h5py
import numpy as np

f= h5py.File('/blue/joel.harley/zhihui.tian/MF_new/data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(10,-1)_numnei(64)_cut(0).h5', 'r')

ims_id=np.load('./700to900.npy')

b=[]
for i in range(100):   # use 200 rather than 250 in case of overflow
    a=[]
    for j in range(1,6):
        img=ims_id[i+j-1]
        a.append(img)
    b.append(a)
b=np.array(b)
b=np.expand_dims(b,axis=2)

with h5py.File('/blue/joel.harley/zhihui.tian/MF_new/data/mf_sz(1024x1024)_ng(4096)_nsteps(1000)_cov(10,-1)_numnei(64)_cut(0).h5', 'r') as hdf:
    ea = hdf['sim0']['euler_angles'][:112, :]
    miso_1d=hdf['sim0']['miso_array'][:6216]



ea_all=np.zeros((100,112,3))
for i in range(100):
    noise=np.random.normal(loc=0, scale=0.1, size=ea.shape)
    ea_all[i]=ea+noise
ea_all=np.array(ea_all)


miso_all=np.zeros((100,6216))
for i in range(100):
    noise=np.random.normal(loc=0, scale=0.1, size=len(miso_1d))
    miso_all[i]=miso_1d+noise
miso_all=np.array(miso_all)


with h5py.File('./Unet_dataset2.h5','w') as hdf:
    hdf.create_dataset('ims_id',data=b)
    hdf.create_dataset('euler_angles', data=ea_all)
    hdf.create_dataset('miso_array', data=miso_all)
import numpy as np
import torch
import math

def get_H(kernel, img_dim1, img_dim2):
    p = math.floor(kernel.shape[0] / 2)
    x_dim1 = img_dim1 + 2 * p
    x_dim2 = img_dim2 + 2 * p
    H = torch.zeros(img_dim1*img_dim2, x_dim1*x_dim2)
    t = 0
    for i in range(H.shape[0]):
        for s in range(kernel.shape[0]):
            H[i, t+s*x_dim2:t+s*x_dim2+kernel.shape[1]] = kernel[s,:]
        if np.mod(t+kernel.shape[1], x_dim2) == 0:
            t = t + kernel.shape[1]
        else:
            t = t + 1
    return H

def get_H_nopad(kernel, img_dim1, img_dim2):
    H = get_H(kernel, img_dim1, img_dim2)
    p = math.floor(kernel.shape[0]/2)
    x_dim1 = img_dim1 + 2*p
    x_dim2 = img_dim2 + 2 * p
    H_ = torch.zeros(x_dim1, x_dim2, H.shape[0])
    for i in range(H_.shape[2]):
        H_[:,:,i] = H[i,:].reshape(x_dim1, x_dim2)
    H_nopad_ = H_[p:x_dim1-p,p:x_dim2-p, :]
    H_nopad = torch.zeros(H_nopad_.shape[2], H_nopad_.shape[0]*H_nopad_.shape[1])
    for i in range(H_nopad.shape[0]):
        H_nopad[i,:] = H_nopad_[:,:,i].reshape(1, H_nopad_.shape[0]*H_nopad_.shape[1])
    return H_nopad


def A(k_1D, img_dim2):
    p = math.floor(k_1D.shape[0] / 2)
    k_1D = k_1D.transpose(0, 1)
    x_dim2 = img_dim2 + 2 * p
    A_k = torch.zeros(x_dim2-k_1D.shape[1]+1,x_dim2)
    for i in range(A_k.shape[0]):
        A_k[i,i:i+k_1D.shape[1]] = k_1D
    return A_k

def A_nopad(k_1D, img_dim2):
    A_k = A(k_1D, img_dim2)
    p = math.floor(k_1D.shape[0]/2)
    A_nopad = A_k[:,p:A_k.shape[1]-p]
    return A_nopad


def get_ss(sr, sc):
    sr = sr.reshape(sr.shape[0], 1)
    sc = sc.reshape(sc.shape[0], 1)
    ss = torch.matmul(sr, sc.transpose(0,1))
    return ss


if __name__=='__main__':
    img = np.arange(36).reshape(6, 6)
    img = torch.tensor(img, dtype=torch.float32)
    kernel = np.array([[1., 3., 2.], [2, 6, 4], [3, 9, 6]])
    kernel = torch.tensor(kernel)
    X = np.zeros((8, 8))
    X[1:7, 1:7] = img
    X = torch.tensor(X)
    X_nopad = img.reshape(36, 1)

    H = get_H_nopad(kernel, img.shape[0], img.shape[1])
    result1 = torch.matmul(H, X_nopad)
    print(result1.reshape(6, 6))

    r = torch.tensor([[1.], [2.], [3.]])
    # r = torch.tensor([[0.11], [0.11], [0.11],[0.11], [0.11], [0.11],[0.11], [0.11], [0.11]])
    c = torch.tensor([[1.], [3.], [2.]])
    Ar = A_nopad(r, img.shape[1])
    # Ar = A_nopad(r, 256)
    Ac = A_nopad(c, img.shape[1])

    Ar_Ac = torch.kron(Ar, Ac)
    result2 = torch.matmul(Ar_Ac, X_nopad)
    print(result2.reshape(6, 6))

    Ur, sr, Vr = torch.svd(Ar, some=False)
    Uc, sc, Vc = torch.svd(Ac, some=False)

    temp1 = torch.matmul(torch.matmul(Vr.transpose(0, 1), img), Vc)
    ss = get_ss(sr, sc)
    # temp1_ = temp1[0:6,0:6]
    # temp2 = ss*temp1_
    temp2 = ss * temp1
    result3 = torch.matmul(torch.matmul(Ur, temp2), Uc.transpose(0, 1)).reshape(36, 1)
    print(result3.reshape(6, 6))


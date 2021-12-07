import torch
from torch.autograd import Variable
import numpy as np
from numpy.linalg import norm

def test(test_loader, cuda, model):
    '''
    Forward pass through the CNN for one epoch
    '''
    model.eval()
    # placeholder for all predictions
    predictions = np.array([], dtype=np.int64).reshape(0) 
    
    # placeholder for all scores
    scores = np.array([], dtype=np.float32).reshape(0,4) 
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if cuda:
                data = Variable(data[0].cuda())
            else:
                data = Variable(data[0])
            output = model(data) # forward pass through the CNN
            # get the index of the max log-probability
            pred = output.data.max(1)[1] 
            predictions = np.concatenate((predictions, pred.cpu().numpy()))
            scores = np.concatenate((scores, output.data.cpu().numpy()),axis=0)
    return predictions, scores

def load_weights(model, cuda, checkpoint):
    '''
    Load trained parameters to model
    '''
    model_dict = model.state_dict()
    if cuda:
        pretrained_dict = torch.load(checkpoint)
    else:
        pretrained_dict = torch.load(checkpoint, 
                                        map_location=torch.device('cpu'))
    pretrained_dict = \
    {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size() }
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
def L0_1D(x, thresh):

    x_abs = np.abs(x)
    x_large = np.where(x_abs>thresh)[0]
    
    y = 0
    for i in range(x_large.shape[0]):
        if i == 1:
            if x_large[i]==1:
                y = y + 1
        else:
            if x_large[i]==1 and x_large[i-1]==0:
                y = y + 1
    return y

def L0_2D(x):
    x_abs = np.abs(x)
    thresh = np.amax(x_abs)*0.5
    y = 0
    for i in range(x_abs.shape[0]):
        for j in range(x_abs.shape[1]):
        
            if x_abs[i,j]>thresh:
                if i == 0 and j == 0:
                    y = y + 1
                elif i == 0:
                    if x_abs[i,j-1]<=thresh:
                        y = y + 1
                elif j == 0:
                    if x_abs[i-1,j]<=thresh and x_abs[i-1,j+1]<=thresh:
                        y = y + 1
                elif j == x_abs.shape[1]-1:
                    if x_abs[i-1,j]<=thresh and x_abs[i,j-1]<=thresh \
                                        and x_abs[i-1,j-1]<=thresh:
                        y = y + 1                    
                else:
                    if x_abs[i-1,j]<=thresh and x_abs[i,j-1]<=thresh \
                            and x_abs[i-1,j-1]<=thresh \
                            and x_abs[i-1,j+1]<=thresh:
                        y = y + 1
    return y

def entropy1D(X):
    e = 0
    for i in range(X.shape[0]):
        if X[i]>0:
            e = e - (X[i] * np.log(X[i]))
    return e
    
def deterministic_dist(n,mesh_v):

    targets = np.zeros((n, (mesh_v.shape[0])**n))
    
    if n==1:
        c = 0
        for i1 in mesh_v:
            targets[:,c]=i1
            targets[:,c] = targets[:,c]/norm(targets[:,c],1)
            c = c + 1
    elif n==2:
        c = 0
        for i1 in mesh_v:
            for i2 in mesh_v:
                targets[:,c]=[i1, i2]
                targets[:,c] = targets[:,c]/norm(targets[:,c],1)
                c = c + 1
    elif n==3:
        c = 0
        for i1 in mesh_v:
            for i2 in mesh_v:
                for i3 in mesh_v:
                    targets[:,c]=[i1, i2, i3]
                    targets[:,c] = targets[:,c]/norm(targets[:,c],1)
                    c = c + 1
    elif n==4:
        c = 0
        for i1 in mesh_v:
            for i2 in mesh_v:
                for i3 in mesh_v:
                    for i4 in mesh_v:
                        targets[:,c]=[i1, i2, i3, i4]
                        targets[:,c] = targets[:,c]/norm(targets[:,c],1)
                        c = c + 1
    elif n==5:
        c = 0
        for i1 in mesh_v:
            for i2 in mesh_v:
                for i3 in mesh_v:
                    for i4 in mesh_v:
                        for i5 in mesh_v:
                            targets[:,c]=[i1, i2, i3, i4, i5]
                            targets[:,c] = targets[:,c]/norm(targets[:,c],1)
                            c = c + 1
    else:
        for i in range(targets.shape[1]):
            ind = np.array(np.unravel_index(i, (mesh_v.shape[0] for j in n)))
            for k in range(targets.shape[0]):
                targets[k,i] = mesh_v[ind[k]]
            targets[:,i] = targets[:,i]/norm(targets[:,i],1)
            
    return targets

def get_rand_peak(image_width, image_height):
    (X, Y) = np.meshgrid(np.linspace(-1,1,image_width), 
                                np.linspace(-1,1,image_height))
    sumXY = np.abs(X)+np.abs(Y)
    target_2d = np.maximum(1.- np.random.rand() * sumXY, 0.)
    x_shift = np.random.randint(0,image_width)
    y_shift = np.random.randint(0,image_height)
    target_2d = np.roll(target_2d, x_shift, axis=0)
    target_2d = np.roll(target_2d, y_shift, axis=1)
    if x_shift<image_width/2.:
        target_2d[0:x_shift, :] = 0
    else:
        target_2d[x_shift:image_width, :] = 0
    if y_shift<image_height/2.:
        target_2d[:, 0:y_shift] = 0
    else:
        target_2d[:, y_shift:image_height] = 0
    return target_2d

def predict_OT(image, lambda_v, channel):

    image_width = image.shape[1]
    image_height = image.shape[2]
    source = image[channel,:,:].flatten()
    source = source/norm(source,1)

    n = source.shape[0]
    epsilon = .01

    C = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            (ix, iy) = np.unravel_index(i,(image_width, image_height))
            (jx, jy) = np.unravel_index(j,(image_width, image_height))
            C[i,j] = norm([ix-jx, iy-jy])

    K = np.exp(-C/epsilon)

#     mesh_size = 3
#     mesh_v = np.linspace(0,1,num=mesh_size)
#     targets_size = (mesh_v.shape[0])**n
    targets_size = 20
    
    dist_W = np.zeros((targets_size,lambda_v.shape[0]))
    dist_l1 = np.zeros((targets_size,lambda_v.shape[0]))
    dist_l2 = np.zeros((targets_size,lambda_v.shape[0]))
    target_v = np.zeros((targets_size,lambda_v.shape[0],n))
    target_W_entropy = np.zeros((lambda_v.shape[0]))
    target_W_points = np.zeros((lambda_v.shape[0]))

    for lambda_ind in range(lambda_v.shape[0]):
#         targets = deterministic_dist(n,mesh_v)
#         for rand_ind in range(targets_size):
#             target = targets[:,rand_ind]
        for rand_ind in range(targets_size):
            #print(str(rand_ind)+',', end = '')
            
            if np.random.rand() > .5:
                target_2d = get_rand_peak(image_width, image_height)
            else:
                target_2d = get_rand_peak(image_width, image_height) \
                            + get_rand_peak(image_width, image_height)
                
            target = target_2d.flatten()
            target = target/norm(target,1)

            u = np.ones(n)
            v = np.ones(n)
            T = np.diag(u) @ K @ np.diag(v)

            for opt_ind in range(100):
                u = source/(K @ v)
                v = target/(K.T @ u)
                T = np.diag(u) @ K @ np.diag(v)

            dist_W[rand_ind,lambda_ind] = norm(C * T, 'fro') \
                + lambda_v[lambda_ind]*entropy1D(target)
            dist_l1[rand_ind,lambda_ind] = norm(source-target,1) \
                + lambda_v[lambda_ind]*entropy1D(target)
            dist_l2[rand_ind,lambda_ind] = norm(source-target,2) \
                + lambda_v[lambda_ind]*entropy1D(target)
            target_v[rand_ind,lambda_ind,:] = target.flatten()

        #print('')            
        I = np.argmin(dist_W[:,lambda_ind])
        optimal_target = target_v[I,lambda_ind,:]
        target_W_entropy[lambda_ind] = entropy1D(optimal_target)
        target_W_points[lambda_ind] = L0_2D(np.reshape(optimal_target,
                                                (image_width, image_height)))
    #print(target_W_entropy)
    print('target_W_points: %d'%target_W_points)
    return target_W_points[0]
    
def test_OT(test_loader, predictions_shape, scores_shape, lambda_v, 
                                predictions_nnet, data_subsample, data_np):
    prediction_class = 3+np.zeros(predictions_shape, dtype=np.int64)
    
    for i in data_subsample:
        p_channel = np.zeros(5)
        for channel in range(0,5):
            print('channel: %d'%channel)
            p_channel[channel] = predict_OT(data_np[i,:,:,:], lambda_v, channel)
        p = np.amax(p_channel)
        print('data point %d, true class label %d, prediction %f'%(i,
                                                predictions_nnet[i], p))
        if p==1:
            prediction_class[i] = 1
        else:
            prediction_class[i] = 2

    return prediction_class, None



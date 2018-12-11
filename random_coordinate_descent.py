import numpy as np
import time

def random_coordinate_descent_primal_subgradient(init, steps, imgs, labels, t_bias, proj=lambda x: x):
    """coordinate_descent_primal_subgradient.
    
    Inputs:
        initial: starting point
        steps: list of scalar step sizes
        imgs: images
        labels: labels
        proj (optional): function mapping points to points
        
    Returns:
        List of all points computed by projected gradient descent.
    """

    print("RCD training starts...")
    t_start = time.time()
    num_examples, num_features = imgs.shape
    xs = [init]
    running_time = [0.0]      
    t = 1
    # iterate for steps
    for i, _lambda in enumerate(steps):        
        # randomly sample from all examples

        for j in range(0, num_features):
            w_ = xs[-1].copy()
            ftr_idx = np.random.choice(num_features,1, replace=True)  
#            ftr_idx = j
            grad_ftr = 0
#             margin = (labels.reshape(labels.shape[0],1)*(imgs@w_))
#             margin_mask = np.squeeze(margin<1)
#             grad_ftr = np.sum(labels[margin_mask]*(imgs[:,ftr_idx][margin_mask]*w_[ftr_idx]))
#             print("anchor1",j, time.time() - t_start)
            count = 0
            for n in range(0, num_examples):
                x_i = imgs[n]
                y_i = labels[n]
                # calculate the margin
                margin = y_i*(w_.T @ x_i)
                # update the w_ based on the margin and w_i
                if margin<1:
                    count += 1
                    grad_ftr += y_i*x_i[ftr_idx]
            grad_ftr /= count
            w_[ftr_idx] = (1-1/t)*w_[ftr_idx] + (1/(_lambda*t))*grad_ftr
        
            # project w_ to the constrained set and store results
            xs.append(proj(w_, _lambda))
            running_time.append(time.time()-t_start)

        t = t_bias*(i+1)
        print("epoch:",i,"  ", "total_time:", time.time() - t_start)
    print("RCD training ends...")

    return xs, running_time

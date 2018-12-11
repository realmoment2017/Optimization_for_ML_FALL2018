import numpy as np
import time

def pegasos_gradient_descent(init, steps, imgs, labels, proj=lambda x: x):
    """Projected gradient descent.
    
    Inputs:
        initial: starting point
        steps: list of scalar step sizes
        imgs: images
        labels: labels
        proj (optional): function mapping points to points
        
    Returns:
        List of all points computed by projected gradient descent.
    """

    print("Pegasos training starts...")
    t_start = time.time()
    num_examples, num_features = imgs.shape
    xs = [init]
    running_time = [0.0]
    
    # iterate for steps
    for i, _lambda in enumerate(steps):
        # randomly sample from all examples
        sample_idx = np.random.choice(num_examples,1, replace=False)
        x_i = imgs[sample_idx]
        y_i = labels[sample_idx]
        w_i = xs[-1]
        
        # calculate the margin
        margin = y_i*(w_i.T @ x_i.T)
        t = i+1
        
        # update the w_ based on the margin and w_i
        if margin<1:
#             print(sample_idx, margin, "margin<1")
            indicator = 1
            w_ = (1-1/t)*w_i + (1/(_lambda*t))*indicator*y_i*x_i.T
        else:
#             print(sample_idx, margin, "margin>1")
            indicator = 0
            w_ = (1-1/t)*w_i + (1/(_lambda*t))*indicator*y_i*x_i.T
        
        # project w_ to the constrained set and store results
        xs.append(proj(w_, _lambda))
        running_time.append(time.time()-t_start)
    print("Pegasos training ends...")
    return xs, running_time

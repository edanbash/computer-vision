from xxlimited import new
import numpy as np
import cv2
import argparse
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix

from helpers import *

### Part 2.1 Toy Problem ###
def create_Ab(src, im2var):
    h, w = src.shape

    e = 0
    x_eq = h * (w-1)
    y_eq = w * (h-1)
    A = np.zeros((x_eq + y_eq + 1, h * w))
    b = np.zeros((x_eq + y_eq + 1, 1))

    # Set up the x equations
    for y in range(h):
        for x in range(w-1):
            A[e][im2var[y][x+1]] = 1
            A[e][im2var[y][x]] = -1
            b[e] = src[y][x+1] - src[y][x]
            e += 1
    
    # Set up the y equations
    for x in range(w):
        for y in range(h-1):
            A[e][im2var[y+1][x]] = 1
            A[e][im2var[y][x]] = -1
            b[e] = src[y+1][x] - src[y][x]
            e += 1
    
    # Assert top left corners are same color
    A[e][0] = 1
    b[e][0] = src[0][0]
    
    return csr_matrix(A), b


def solve_linear_system(source):
    # Create the indexing matrix
    im_var = im2var(source)

    print('Setting up matrix')
    A, b = create_Ab(source, im_var)
    
    print(f'Solving linear system of {A.shape[0]} equations')
    v = lsqr(A, b)[0]

    return v.reshape(source.shape).astype(np.int)

def section1():
    im = cv2.imread('./samples/toy_problem.png', 0)
    im = im.astype(np.float32)

    result = solve_linear_system(im)

    cv2.imshow('image', result/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('./output/reconstructed_toy.jpg', result)
        

def create_A(im2var):
    print('Creating sparse matrix')
    
    h, w = im2var.shape
    A = np.zeros((4 * h * w + 1, h * w))
    e = 0

    # Set up the x equations
    for y in range(h):
        for x in range(w):
            # Get gradient from right neigbor
            if x+1 < w:
                A[e][im2var[y][x+1]] = 1
                A[e][im2var[y][x]] = -1
            else:
                A[e][im2var[y][x]] = 1
            e += 1

            # Get gradient from left neigbor
            if x-1 > 0:
                A[e][im2var[y][x]] = 1
                A[e][im2var[y][x-1]] = -1
            else:
                A[e][im2var[y][x]] = 1
            e += 1
    
    # Set up the y equations
    for x in range(w):
        for y in range(h):
            # Get gradient from top neigbor
            if y+1 < h:
                A[e][im2var[y+1][x]] = 1
                A[e][im2var[y][x]] = -1
            else:
                A[e][im2var[y][x]] = 1
            e += 1

            # Get gradient from bottom neigbor
            if y-1 > 0:
                A[e][im2var[y][x]] = 1
                A[e][im2var[y-1][x]] = -1
            else:
                A[e][im2var[y][x]] = 1
            e += 1
    
    return csr_matrix(A)


def create_b(source, target, im2var, top_left, mixed=True):
    h, w = im2var.shape
    b = np.zeros((4 * h * w + 1, 1))
    x_start, y_start = top_left
    e = 0

    # Set up the x equations
    for y in range(h):
        for x in range(w):
            # Get gradient from right neigbor
            source_grad = None
            if x+1 < w:
                b[e] = source[y][x+1] - source[y][x]
                if mixed:
                    target_grad = target[y_start + y][x_start + (x+1)] - target[y_start + y][x_start + x]
                    b[e] = get_max_grad(b[e], target_grad)
            else:
                source_grad = source[y][x] - source[y][x-1]
                b[e] = source_grad + target[y_start + y][x_start + (x+1)] 
            e += 1

            # Get gradient from left neigbor
            if x-1 > 0:
                b[e] = source[y][x] - source[y][x-1]
                if mixed:
                    target_grad = target[y_start + y][x_start + x] - target[y_start + y][x_start + (x-1)]
                    b[e] = get_max_grad(b[e], target_grad)
            else:
                source_grad = source[y][x+1] - source[y][x]
                b[e] = source_grad + target[y_start + y][x_start + (x-1)]
            e += 1
    
    # Set up the y equations
    for x in range(w):
        for y in range(h):
            # Get gradient from top neigbor
            if y+1 < h:
                b[e] = source[y+1][x] - source[y][x]
                if mixed:
                    target_grad = target[y_start + (y+1)][x_start + x] - target[y_start + y][x_start + x]
                    b[e] = get_max_grad(b[e], target_grad)
            else:
                source_grad = source[y][x] - source[y-1][x]
                b[e] = source_grad + target[y_start + (y+1)][x_start + x]
            e += 1

            # Get gradient from bottom neigbor
            if y-1 > 0:
                b[e] = source[y][x] - source[y-1][x]
                if mixed:
                    target_grad = target[y_start + y][x_start + x] - target[y_start + (y-1)][x_start + x]
                    b[e] = get_max_grad(b[e], target_grad)
            else:
                source_grad = source[y+1][x] - source[y][x]
                b[e] = source_grad + target[y_start + (y-1)][x_start + x] 
            e += 1

    return b
    

def section2():
    target_path = './samples/wood.jpg'
    source_path = './samples/rose.png'
    output_path = './output/mixed.jpg'

    target = cv2.imread(target_path, 1).astype(np.float32)
    source = cv2.imread(source_path, 1).astype(np.float32)

    top_left = get_top_left_point(target)

    im_var = im2var(source)
    A = create_A(im_var)
    
    v_channels = []
    for c in range(3):
        # Create the constant b vector channel c
        b = create_b(source[:,:,c], target[:,:,c], im_var, top_left)
       
        print(f'Solving channel {c}')
        # Solve for v using least squares
        v = lsqr(A, b)[0]

        # Unflatten v into source shape
        v = v.reshape(im_var.shape).astype(np.int32)
        v_channels.append(v)

    # Combine v into an rgb image
    v = np.dstack(v_channels)
    
    # Add course to target image
    result = add_source(target, v, top_left)

    cv2.imshow('image', result/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cv2.imwrite(output_path, result)

   

def extra():
    path = './samples/rose.png'
    im = cv2.imread(path, 1).astype(np.float32)
    resize_img(im, path, scale=0.9)
    

def main():
    # Create argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--section", required=False, help="first operand")
    args = vars(ap.parse_args())

    if args['section'] == '1':
        section1()  
    if args['section'] == '2':
        section2()
    if args['section'] == 'e':
        extra()  
    else:
        print('No section selected')



# driver function
if __name__=="__main__":
    main()
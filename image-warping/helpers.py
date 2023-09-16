
import scipy.signal as scisig
import numpy as np
import cv2
import skimage.transform as sktr
import csv

## UTILITY HELPER FUNCTIONS

def extra():
    fname = 'kresge' 

    im1 = cv2.imread(f'./data/{fname}1.jpg')
    im2 = cv2.imread(f'./data/{fname}2.jpg')

    im1 = resize_img(im1, scale=20)
    im2 = resize_img(im2, scale=20)

    cv2.imwrite(f'./data/{fname}_1.jpg', im1)
    cv2.imwrite(f'./data/{fname}_2.jpg', im2)


def csv_to_list(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        return np.array(list(reader)).astype(float)

def resize_img(img, scale=20):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
  
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def add_ones(pts):
    new_pts = np.zeros((pts.shape[0], 3))
    for i, pt in enumerate(pts):
       new_pts[i]  = np.array([pt[0], pt[1], 1])
    return new_pts

def ssd(A, B):
    return np.sqrt(np.sum(((A-B)**2), axis=1))


## MANUAL POINT SELECT HELPER FUNCTIONS

# Handles click event, draws circle, and saves coordinates
def click_event(event, x, y, flags, params):
    coords = params['coords']
    img = params['image']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x,y))
        cv2.circle(img, (x,y), 2, (255, 255, 255), 3)
        cv2.imshow('image', img)

# Iniaties interactive creation of face mesh
def get_pts(img, coords):
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event, {'image': img, 'coords': coords})
    cv2.waitKey(0)
    cv2.destroyAllWindows()


## BLENDING HELPER FUNCTIONS

def gaussian_stack(im, levels, size=18, sigma=3):
    gauss_stack = [im]
    gauss = cv2.getGaussianKernel(size, sigma) 
    gauss = np.outer(gauss, gauss.T)
    for i in range(1, levels):
        curr = scisig.convolve2d(gauss_stack[i-1], gauss, mode = 'same')
        gauss_stack.append(curr)
    return gauss_stack

def lap_stack(im, levels, size=18, sigma=3):
    gauss_stack = gaussian_stack(im, levels, sigma, size)
    lap_stack = [[[]]]
    for i in range(levels - 1):
        curr = gauss_stack[i] - gauss_stack[i + 1]
        lap_stack.append(curr)
    lap_stack.append(gauss_stack[-1])
    return np.array(lap_stack[1:])

def blend(lap1, lap2, gr, levels):
    res = [[[]]]
    for i in range(levels):
        res.append((gr[i] * lap2[i]) + (1 - gr[i]) * lap1[i])
    return np.sum(res[1:], axis=0)

def blend_img(im1, im2, mask, levels=2):
    b1, g1, r1 = im1.T[0].T, im1.T[1].T, im1.T[2].T
    b2, g2, r2 = im2.T[0].T, im2.T[1].T, im2.T[2].T

    la_b = lap_stack(b1, levels)
    la_g = lap_stack(g1, levels)
    la_r = lap_stack(r1, levels)

    lb_b = lap_stack(b2, levels)
    lb_g = lap_stack(g2, levels)
    lb_r = lap_stack(r2, levels)

    gauss = cv2.getGaussianKernel(25, 5) 
    kernel = np.outer(gauss, gauss.T)
    mask_blurred = scisig.convolve2d(mask, kernel, mode='same')

    gr = gaussian_stack(mask_blurred, levels, size=15, sigma=2)
    combined_b = blend(la_b, lb_b, gr, levels)
    combined_g = blend(la_g, lb_g, gr, levels)
    combined_r = blend(la_r, lb_r, gr, levels)

    blended = np.dstack([combined_b, combined_g, combined_r])
    return blended


## STORAGE HELPER FUNCTIONS

def store_best_pts(im, best_pts, fname):
    for pt in best_pts:
        cv2.circle(im, [pt[1], pt[0]], 2, (255, 255, 255), 2) 

    cv2.imwrite(f'./output/{fname}-corners.jpg', im)


def store_fds(im, best_pts, box_pts, feature_descriptors, fname, indices):
    count = 0
    for i, pt in enumerate(best_pts[:6]):
        if i in indices:
            begin_x, begin_y, end_x, end_y = box_pts[i]
            top_left, bottom_right = [begin_y, begin_x], [end_y, end_x]
            cv2.rectangle(im, top_left,bottom_right, (255,255,255), 2)
            cv2.putText(im, f'F{count}', (begin_y, begin_x-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.imwrite(f'./output/{fname}-feature-boxes.jpg', im)
        
            f = resize_img(feature_descriptors[i], scale=2000)
            cv2.imwrite(f'./output/{fname}-fd-{count}.jpg', f)
            count += 1

def store_matches(all_images, all_matches, fnames, match_name):
    for i, matching_pts in enumerate(all_matches):
        for j, pt in enumerate(matching_pts):
            x, y = pt[0], pt[1]
            cv2.circle(all_images[i], [x,y], 2, (255, 255, 255), 2) 
            cv2.putText(all_images[i], f'F{j}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imwrite(f'./output/{fnames[i]}-{match_name}.jpg', all_images[i])
    
        cv2.imshow('image', all_images[i]/255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


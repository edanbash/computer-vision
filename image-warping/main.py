from xxlimited import new
import numpy as np
import cv2
import argparse
import math
from scipy.interpolate import griddata
from helpers import *
from harris import *

#############################
## PROJ 4A FUNCTIONS BEGIN ##
#############################

# Compute the A matrix for homography computation (A = Hb)
def compute_point_matrix(im1_pts, im2_pts):
    P, b = [], []
    for pt1, pt2 in zip(im1_pts, im2_pts):
        x1, x2 = pt1[0], pt2[0]
        y1, y2 = pt1[1], pt2[1]
        p_i = np.array([[x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2],
                        [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2]])
        P.append(p_i)
        b.extend([x2, y2])
    return np.vstack(P), b

# Compute homography matrix
def compute_H(im1_pts, im2_pts):
    A, b = compute_point_matrix(im1_pts, im2_pts)
    h, _, _, _= np.linalg.lstsq(A, b, rcond=None)
    h = np.append(h, 1)
    H = np.reshape(h, (3,3))
    return H

# Fill in missinng values in an image channel
def interpolate(channel):
    pixles_x, pixles_y = np.where((channel[:,:]!=0))
    cords_system = np.dstack((pixles_x, pixles_y))[0]
    nx, ny = channel.shape[1], channel.shape[0]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    samples = channel[pixles_x, pixles_y]
  
    return griddata(cords_system, samples.flatten(), (Y, X))

# Creates coordinate matrix based on image size
def create_coords(im):
    # Create a coordinate matrix
    rows, cols = im.shape[0], im.shape[1]
    old_coords = np.indices((cols, rows))
    old_coords = old_coords.reshape(2, -1)
    old_coords = np.vstack((old_coords, np.ones(rows*cols)))
    old_coords = old_coords.astype(int)
    return old_coords

# Creates new coordinate system based on homography transformation
def transform_coords(old_coords, H):
    new_coords = H @ old_coords
    w = new_coords[2]
    new_coords = (new_coords / w).astype(int)
    return new_coords

# Warp src img into the homography space
def warp_image(src_img, H):
    # Create a coordinate matrix
    old_coords = create_coords(src_img)
    new_coords = transform_coords(old_coords, H)

    # Get bounding shape
    min_x, max_x = np.min(new_coords[1]), np.max(new_coords[1])
    min_y, max_y = np.min(new_coords[0]), np.max(new_coords[0])

    # Shift the coordinates
    new_coords[1] += abs(min_x)
    new_coords[0] += abs(min_y)

    bound_x = abs(max_x - min_x)
    bound_y = abs(max_y - min_y)

    print(min_x, max_x, min_y, max_y)

    # Only keep indices that are valid
    indices = np.all((new_coords[1] < bound_x, new_coords[1]>=0, new_coords[0] < bound_y, new_coords[0]>=0), axis=0)

    # Intialize resulting image
    result = np.zeros((bound_x, bound_y, 3))

    # Set result pixels in new coordinate system based on pixel values of old coordinate system
    result[new_coords[1][indices], new_coords[0][indices]] = src_img[old_coords[1][indices], old_coords[0][indices]]

    # Interpolate the image
    print("Interpolating image...")
    for i in range(3):
        result[:,:,i] = interpolate(result[:,:,i])

    cv2.imshow('warped image', result/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

# Create box to recitify image
def make_box(pts):
    x = [pt[0] for pt in pts]
    y = [pt[1] for pt in pts]
    bound_x = max(x) - min(x)
    bound_y = max(y) - min(y)
    return [[0,0],[bound_x, 0],[0, bound_y], [bound_x, bound_y]]

# Change perspective of image
def rectify(filename, src_pts):
    src_img = cv2.imread(filename, 1)
    dst_pts = make_box(src_pts)
    H = compute_H(src_pts, dst_pts)
    warped_im = warp_image(src_img, H)
    return warped_im

# Calculate new coordinates for correspondence pts
def get_warped_pts(src_img, src_pts, H):
    # Create transformed coordinate space
    old_coords = create_coords(src_img)
    new_coords = transform_coords(old_coords, H)
    
    # Get bounding shape
    min_x = np.min(new_coords[1])
    min_y = np.min(new_coords[0])

    # Get new correspondence pts
    warped_pts = []
    for pt in src_pts:
        p = H @ np.array([pt[0], pt[1], 1])
        p = (p[:2] / p[2]).astype(int)
        p[1] += abs(min_x)
        p[0] += abs(min_y)
        warped_pts.append(p)
    return warped_pts

# Align images for mosaic
def align_images(warped, fixed, warped_pts, fixed_pts):
    # Get difference between warped pts and fixed pts
    n = len(fixed_pts)

    x_diff = np.abs(np.array([warped_pts[i][0] - fixed_pts[i][0] for i in range(n)]))
    y_diff = np.abs(np.array([warped_pts[i][1] - fixed_pts[i][1] for i in range(n)]))

    # Set the offsets to be avergae of the diffs
    col_offset = np.mean(x_diff).astype(int)
    row_offset = np.mean(y_diff).astype(int)

    # Set boundaries for mosaic image
    r = max(warped.shape[0], fixed.shape[0])
    c = col_offset + fixed.shape[1]
    result = np.zeros((r, c, 3)) 

    # Add warped and fixed image to result
    result[:warped.shape[0], :warped.shape[1]] = warped
    result[row_offset:row_offset + fixed.shape[0], col_offset:] = fixed

    print("Blending Image")
    #result = resize_img(result, scale=scale)

    # Create mask 
    mask = np.zeros((result.shape[0], result.shape[1]))
    intersection = int(col_offset)
    mask[:, intersection-5:intersection+5] = 1

    result = blend_img(result, result, mask)
    return result

# Create mosaic from two images
def create_mosaic(moving, fixed, moving_pts, fixed_pts):
    # Warp image into homography space
    H = compute_H(moving_pts, fixed_pts)
    warped = warp_image(moving, H)
    cv2.imwrite(f'./output/room-warped.jpg', warped)
    warped = cv2.imread(f'./output/room-warped.jpg')

    # Create mosaic
    warped_pts = get_warped_pts(moving, moving_pts, H)
    result = align_images(warped, fixed, warped_pts, fixed_pts)

    cv2.imshow('mosaic', result/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

###########################
## PROJ 4A FUNCTIONS END ##
###########################

def partA():
    # Rectify images
    a_pts = [(397, 379), (641, 154), (833, 707), (1070, 432)]
    #rectified = rectify('./data/menu.jpeg', a_pts)
    #cv2.imwrite('./output/rectified_menu.jpg', rectified)

    b_pts = [(109, 255), (623, 96), (104, 450), (632, 345)]
    #rectified = rectify('./data/chik.jpeg', b_pts)
    #cv2.imwrite('./output/rectified_chik.jpg', rectified)
    
    # Create mosaic
    name = 'room' # 'regent', 'campus
    fixed_name, fixed_pts_name = f'{name}_2', f'{name}_fixed_pts.csv'
    moving_name, moving_pts_name = f'{name}_1', f'{name}_moving_pts.csv'

    fixed = cv2.imread(f'./data/{fixed_name}.jpg')
    moving = cv2.imread(f'./data/{moving_name}.jpg')

    fixed_pts = csv_to_list(f'./points/{fixed_pts_name}')
    moving_pts = csv_to_list(f'./points/{moving_pts_name}')

    result = create_mosaic(moving, fixed, moving_pts, fixed_pts)
    #cv2.imwrite(f'./output/{name}_mosaic_blend.jpg', result)


#############################
## PROJ 4B FUNCTIONS BEGIN ##
#############################

# Adaptive non-maximal suppresion alogrithm. Returns good spread of corner points.
def anms(im, coords, c=0.9, max_pts=100):
    print('Computing ANMS')
    
    # Create array of interest pts
    interest_pts = np.array([[coords[0][i], coords[1][i]] for i in range(coords.shape[1])])

    # Create array for corner strengths
    corner_strengths = np.array([im[pt[0], pt[1]] for pt in interest_pts])
    
    # Find minimum suppression radius per interest point
    radiuses = {}
    for i, pt in enumerate(interest_pts):
        if i%1000==0:
            print(f'Iteration {i}/{len(interest_pts)}')

        indices = np.where(corner_strengths > 1/c * corner_strengths[i])[0]
        if len(indices) == 0:
            radiuses[np.array(pt).tobytes()] = float('inf')
            continue

        dist = dist2(np.array([pt]), interest_pts[indices])
        radiuses[np.array(pt).tobytes()] = np.min(dist)
  
    # Sort points by radius and return the largest ones
    sorted_r = sorted(radiuses.items(), key=lambda x: x[1], reverse=True)
    sorted_pts = np.array([np.frombuffer(k, dtype=int) for k, _ in sorted_r])
    return sorted_pts[:max_pts]


# Get the pixels surrounding interest points in an image
def get_feature_descriptors(im, best_pts, window_size=40, s=5):
    print('Fetching Feature Descriptors')

    # Crete feature descriptor for each interest point
    box_pts = []
    feature_descriptors = []
    for i, pt in enumerate(best_pts):
        # Define the box for feature descriptor
        x, y = pt[0], pt[1]
        begin_x = max(x - window_size//2, 0)
        end_x = min(x + window_size//2, im.shape[0])
        begin_y = max(y - window_size//2, 0)
        end_y = min(y + window_size//2, im.shape[1])

        # Extract and rescale the feature descriptor
        feature_desc = im[begin_x:end_x, begin_y:end_y]
        scale = (window_size//s)/window_size * 100
        feature_desc = resize_img(feature_desc, scale=scale)
        
        feature_descriptors.append(feature_desc)
        box_pts.append([begin_x, begin_y, end_x, end_y])

    return feature_descriptors, box_pts

# Find the best feature matches given two sets of feature descriptors
def feature_matching(fds1, fds2, thresh=0.7):
    print('Doing Feature Matching')

    matches = []
    for i, fd1 in enumerate(fds1):
        nn1_idx = -1
        nn1_err = float('inf')
        nn_ratio = float('inf')

        for j, fd2 in enumerate(fds2):
            mse = np.mean(np.square(fd1 - fd2))
            if mse < nn1_err:
                nn_ratio = nn_ratio if math.isinf(nn1_err) else mse/nn1_err 
                nn1_err, nn1_idx = mse, j             
        matches.append((i, nn1_idx, nn_ratio))

    filtered_matches = [(match[0], match[1]) for match in matches if match[2] <= thresh]
    return filtered_matches

# Ransac algorithm to remove outliers frmo our feature matches.
def ransac(im, matches1, matches2, eps=10):
    print("Doing RANSAC")
    n = len(matches1)
    max_inlier_indices = []

    for i in range(10):
        indices = np.random.choice(list(range(n)), size=4, replace=False)
        m1, m2 = matches1[indices], matches2[indices]
        H = compute_H(m1, m2)

        warped_matches = []
        matches1 = add_ones(matches1)
        for match in matches1:
            warped_pt = H @ match
            warped_pt = warped_pt[:2]//warped_pt[2]
            warped_matches.append(warped_pt)
        
        dist = ssd(warped_matches, matches2)
        inlier_indices = np.where(dist < eps)[0]
        if len(inlier_indices) > len(max_inlier_indices):
            max_inlier_indices = inlier_indices


    m1, m2 = matches1[max_inlier_indices], matches2[max_inlier_indices]
    best_H = compute_H(m1, m2) 
    return best_H, max_inlier_indices

###########################
## PROJ 4B FUNCTIONS END ##
###########################

def partB():
    fname = 'kresge' # haas, room, regent, campus, music, doe
    im1 = cv2.imread(f'./data/{fname}_1.jpg', 0)
    im2 = cv2.imread(f'./data/{fname}_2.jpg', 0)

    # Extract the Harris corners
    h1, coords1 = get_harris_corners(im1)
    #cv2.imwrite(f'./output/{fname}_1-harris.jpg', h1*255)
    h2, coords2 = get_harris_corners(im2)
    #cv2.imwrite(f'./output/{fname}_2-harris.jpg', h2*255)

    # Extract the best spread of Harris corners with ANMS
    best_pts1 = anms(h1, coords1)
    #store_best_pts(im1, best_pts1, f'{fname}_1')
    best_pts2 = anms(h2, coords2)
    #store_best_pts(im2, best_pts2, f'{fname}_2')

    # Extract the feature descriptors
    feature_descriptors1, box_pts1 = get_feature_descriptors(im1, best_pts1)
    #store_fds(im1, best_pts1, box_pts1, feature_descriptors1, f'{fname}_1', indices=[0,1,2,3])
    feature_descriptors2, box_pts2 = get_feature_descriptors(im2, best_pts2)
    #store_fds(im2, best_pts2, box_pts2, feature_descriptors2, f'{fname}_2', indices=[0,1,4,5])

    # Perform feature matching
    matching_idxs = feature_matching(feature_descriptors1, feature_descriptors2)
    print(f'Matching Features Length: {len(matching_idxs)}')

    # Extract the matching pts
    matches1 = best_pts1[[m[0] for m in matching_idxs]]
    matches2 = best_pts2[[m[1] for m in matching_idxs]]
    print(matches1, matches2)

    # Swap x and y coordinate to compute homography
    matches1 = np.array([[pt[1], pt[0]] for pt in matches1])
    matches2 = np.array([[pt[1], pt[0]] for pt in matches2])

    im1 = cv2.imread(f'./data/{fname}_1.jpg', 0)
    im2 = cv2.imread(f'./data/{fname}_2.jpg', 0)
    #store_matches([im1, im2], [matches1, matches2], [f'{fname}_1', f'{fname}_2'], 'matches')

    H, inlier_indices = ransac(im1, matches1, matches2)
    print(inlier_indices)
    
    im1 = cv2.imread(f'./data/{fname}_1.jpg', 0)
    im2 = cv2.imread(f'./data/{fname}_2.jpg', 0)
    #store_matches([im1, im2], [matches1[inlier_indices], matches2[inlier_indices]], [f'{fname}_1', f'{fname}_2'], 'ransac')
    
    # Read in the tow images and their correspondence points
    moving = cv2.imread(f'./data/{fname}_1.jpg')
    fixed = cv2.imread(f'./data/{fname}_2.jpg')
    moving_pts = matches1[inlier_indices]
    fixed_pts = matches2[inlier_indices]

    # Create mosaic (from part A)
    H = compute_H(moving_pts, fixed_pts)
    warped = warp_image(moving, H)
    warped_pts = get_warped_pts(moving, moving_pts, H)
    result = align_images(warped, fixed, warped_pts, fixed_pts)

    cv2.imshow('mosaic', result/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cv2.imwrite(f'./output/{fname}-auto-mosaic.jpg', result)


def main():
    # Create argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--section", required=False, help="first operand")
    args = vars(ap.parse_args())

    if args['section'] == 'a':
        partA()  
    elif args['section'] == 'b':
        partB()  
    elif args['section'] == 'e':
        extra()
    else:
        print('No section selected')



# driver function
if __name__=="__main__":
    main()
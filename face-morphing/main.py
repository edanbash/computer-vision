from email.policy import default
import numpy as np
import argparse
import imageio
import os
import cv2
from scipy.spatial import Delaunay
from skimage.draw import polygon
from scipy.interpolate import griddata
from helpers import *

# Order of points: left eye, right eye, left mouth, right mouth, left bottom head, left top head, middle top head, right top head, right bottom head, left jaw, right jaw, nose
#IMA_PTS = [(287, 376), (474, 377), (389, 464), (314, 572), (455, 570), (384, 714), (276, 687), (194, 580), (162, 405), (181, 272), (262, 184), (391, 143), (513, 206), (567, 291), (606, 406), (570, 594), (506, 685)]
#IMB_PTS = [(305, 376), (483, 368), (398, 467), (332, 560), (464, 555), (405, 693), (314, 667), (212, 545), (170, 374), (180, 233), (252, 152), (382, 122), (514, 157), (580, 230), (607, 345), (583, 534), (519, 650)]
IMA_PTS = [(7, 8), (778, 10), (784, 902), (6, 898), (246, 382), (286, 360), (330, 380), (287, 389), (422, 387), (478, 361), (522, 385), (474, 397), (353, 412), (349, 457), (324, 490), (390, 500), (446, 481), (420, 456), (413, 417), (312, 568), (388, 596), (458, 566), (384, 547), (179, 428), (200, 568), (243, 654), (322, 706), (385, 710), (448, 703), (520, 659), (566, 578), (591, 437), (212, 333), (281, 296), (351, 327), (418, 328), (498, 307), (551, 338)]
IMB_PTS = [(6, 8), (786, 8), (781, 900), (16, 887), (262, 382), (303, 360), (336, 376), (301, 388), (443, 374), (483, 350), (524, 376), (488, 381), (370, 396), (369, 444), (345, 477), (400, 494), (448, 478), (425, 438), (419, 394), (329, 558), (402, 585), (471, 552), (400, 543), (197, 386), (220, 535), (276, 635), (343, 685), (402, 694), (463, 681), (526, 633), (572, 528), (588, 397), (225, 353), (294, 321), (346, 336), (438, 330), (504, 315), (557, 339)]

EDAN_KEY_PTS = [(106, 179), (136, 178), (87, 223), (107, 211), (118, 213), (131, 209), (154, 222), 
(134, 235), (110, 235), (53, 122), (74, 110), (96, 122), (74, 129), (138, 125), (162, 111), (184, 123), 
(162, 132), (26, 120), (26, 165), (33, 218), (43, 255), (59, 275), (83, 292), (121, 296), (156, 293), 
(179, 277), (199, 253), (207, 219), (213, 164), (214, 126), (39, 97), (50, 84), (93, 81), (103, 93), 
(133, 98), (143, 84), (186, 84), (200, 103), (109, 132), (107, 156), (92, 170), (91, 185), (146, 186), 
(146, 168), (134, 156), (131, 133), (0,0), (249, 0), (0, 299), (249, 299)]

edan_k12_dict = {
    'KEY_PTS_0': [(94, 173), (113, 159), (137, 173), (112, 183), (184, 178), (203, 163), (228, 174), (213, 187), (148, 190), (145, 215), (133, 232), (162, 237), (189, 234), (177, 218), (174, 192), (121, 273), (147, 259), (170, 258), (197, 266), (177, 295), (145, 295), (55, 171), (64, 221), (74, 277), (103, 314), (132, 335), (154, 342), (179, 336), (211, 313), (240, 272), (253, 223), (261, 177), (80, 145), (106, 133), (133, 137), (181, 147), (213, 136), (241, 148), (51, 126), (69, 66), (146, 28), (230, 64), (256, 119), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_1': [(82, 177), (104, 162), (130, 178), (108, 188), (175, 184), (197, 165), (219, 181), (201, 190), (138, 194), (136, 219), (125, 237), (154, 237), (180, 235), (169, 219), (164, 194), (108, 265), (140, 262), (163, 260), (196, 264), (168, 285), (138, 286), (53, 179), (59, 222), (77, 295), (102, 322), (122, 336), (147, 341), (174, 337), (192, 325), (221, 295), (244, 221), (251, 179), (68, 151), (98, 136), (131, 148), (180, 147), (210, 136), (237, 152), (51, 128), (73, 73), (144, 37), (221, 72), (247, 120), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_2': [(84, 179), (107, 167), (129, 180), (107, 190), (179, 183), (201, 170), (229, 181), (209, 192), (142, 198), (142, 223), (126, 238), (159, 243), (184, 238), (171, 222), (169, 199), (106, 267), (141, 263), (166, 261), (202, 264), (173, 293), (140, 293), (47, 178), (53, 217), (72, 289), (100, 315), (128, 336), (153, 339), (179, 338), (204, 316), (232, 284), (250, 222), (252, 181), (68, 153), (93, 137), (126, 142), (178, 146), (212, 135), (239, 151), (43, 128), (68, 74), (143, 40), (223, 72), (248, 117), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_3': [(90, 177), (111, 162), (141, 174), (116, 185), (186, 176), (210, 159), (233, 172), (211, 185), (151, 190), (151, 211), (139, 232), (170, 236), (198, 228), (185, 211), (180, 189), (119, 267), (150, 259), (181, 256), (214, 259), (191, 287), (150, 286), (53, 180), (57, 225), (70, 284), (99, 318), (142, 339), (168, 344), (195, 340), (223, 314), (244, 276), (254, 219), (261, 171), (79, 149), (109, 130), (144, 138), (184, 139), (211, 127), (245, 144), (48, 133), (72, 69), (144, 25), (228, 65), (257, 113), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_4': [(87, 170), (115, 157), (139, 173), (113, 183), (187, 169), (209, 154), (243, 166), (216, 175), (154, 189), (157, 215), (146, 232), (179, 237), (203, 229), (191, 213), (186, 189), (130, 272), (162, 260), (188, 256), (219, 261), (198, 290), (163, 291), (42, 170), (51, 221), (74, 294), (113, 321), (150, 336), (182, 340), (211, 331), (232, 306), (251, 273), (262, 210), (269, 163), (71, 145), (98, 128), (139, 134), (189, 138), (221, 122), (253, 133), (34, 122), (54, 67), (127, 29), (211, 50), (243, 94), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_5': [(88, 168), (112, 155), (132, 174), (108, 180), (178, 180), (205, 165), (230, 181), (207, 189), (141, 187), (139, 219), (122, 234), (154, 244), (181, 238), (173, 223), (171, 192), (104, 266), (134, 262), (161, 262), (192, 269), (165, 292), (132, 291), (47, 162), (50, 210), (58, 276), (85, 310), (112, 333), (141, 340), (172, 335), (199, 315), (225, 279), (241, 224), (254, 180), (77, 139), (107, 128), (137, 141), (190, 153), (220, 144), (244, 159), (51, 107), (77, 57), (152, 24), (236, 70), (262, 124), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_6': [(68, 175), (92, 164), (113, 179), (92, 186), (163, 184), (187, 170), (216, 180), (192, 192), (124, 197), (120, 222), (103, 239), (134, 247), (167, 237), (154, 221), (152, 198), (90, 269), (120, 263), (144, 261), (184, 266), (152, 296), (117, 294), (38, 172), (46, 212), (52, 271), (74, 303), (108, 335), (133, 337), (157, 336), (195, 314), (228, 273), (246, 217), (251, 184), (53, 151), (82, 140), (116, 151), (165, 156), (196, 145), (228, 151), (42, 120), (69, 60), (145, 24), (227, 66), (253, 125), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_7': [(94, 169), (119, 158), (141, 177), (117, 181), (189, 185), (210, 171), (234, 189), (215, 195), (153, 195), (152, 222), (138, 237), (165, 244), (194, 242), (185, 224), (183, 199), (115, 267), (147, 262), (172, 262), (197, 274), (175, 296), (140, 290), (47, 164), (53, 215), (60, 276), (86, 311), (129, 337), (151, 343), (179, 340), (212, 315), (230, 284), (244, 232), (253, 189), (84, 144), (117, 132), (147, 145), (192, 161), (222, 149), (246, 167), (50, 128), (77, 75), (161, 38), (236, 77), (258, 136), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_8': [(65, 175), (86, 158), (112, 176), (88, 184), (162, 178), (188, 159), (215, 172), (194, 179), (123, 191), (123, 213), (108, 236), (137, 243), (169, 235), (158, 217), (152, 192), (93, 272), (123, 265), (150, 262), (184, 268), (159, 292), (123, 292), (39, 174), (49, 228), (61, 285), (83, 317), (115, 339), (138, 341), (165, 338), (205, 313), (233, 278), (249, 219), (250, 175), (52, 147), (78, 131), (110, 140), (167, 142), (203, 130), (231, 144), (38, 123), (56, 70), (134, 35), (232, 68), (251, 122), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_9': [(62, 167), (86, 155), (109, 172), (84, 177), (160, 177), (183, 162), (211, 172), (193, 186), (117, 193), (113, 218), (101, 232), (129, 245), (163, 237), (148, 216), (146, 193), (77, 259), (112, 256), (141, 260), (181, 260), (144, 294), (108, 292), (33, 163), (36, 204), (39, 260), (63, 300), (98, 337), (126, 343), (156, 341), (197, 315), (235, 265), (246, 217), (251, 177), (47, 140), (75, 128), (106, 140), (162, 151), (195, 140), (227, 152), (33, 114), (57, 66), (146, 40), (232, 82), (256, 135), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_10': [(49, 168), (69, 155), (92, 171), (69, 174), (149, 178), (171, 165), (198, 177), (176, 181), (100, 192), (96, 213), (83, 233), (113, 242), (147, 233), (130, 217), (130, 193), (70, 263), (99, 256), (129, 256), (167, 261), (136, 294), (101, 292), (32, 167), (33, 213), (38, 264), (58, 298), (85, 331), (113, 340), (145, 334), (190, 305), (225, 268), (240, 221), (237, 181), (35, 143), (63, 130), (97, 143), (142, 146), (177, 135), (208, 146), (29, 116), (48, 73), (130, 33), (227, 89), (247, 142), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_11': [(58, 175), (79, 161), (105, 174), (81, 184), (149, 175), (176, 157), (206, 169), (187, 175), (113, 196), (113, 220), (100, 241), (135, 247), (167, 233), (151, 218), (144, 196), (89, 268), (120, 263), (149, 260), (190, 254), (157, 293), (120, 293), (33, 175), (40, 214), (51, 267), (80, 306), (113, 340), (141, 345), (174, 338), (212, 303), (239, 259), (247, 210), (243, 167), (37, 155), (61, 138), (94, 146), (150, 146), (183, 129), (218, 146), (24, 134), (38, 83), (117, 36), (218, 82), (244, 129), (0,0), (299,0), (0,349), (299,349)],
    'KEY_PTS_12': [(79, 143), (106, 129), (130, 148), (106, 155), (177, 154), (205, 139), (233, 158), (208, 165), (139, 176), (137, 205), (120, 228), (151, 235), (185, 229), (168, 205), (168, 175), (105, 264), (132, 258), (155, 258), (189, 266), (163, 281), (133, 280), (40, 144), (43, 190), (53, 264), (77, 298), (108, 330), (139, 341), (168, 337), (205, 315), (235, 274), (248, 216), (258, 165), (65, 109), (94, 99), (135, 118), (180, 124), (212, 114), (244, 130), (41, 83), (61, 29), (163, 11), (246, 59), (263, 115), (0,0), (299,0), (0,349), (299,349)],
}

# Handles click event, draws circle, and saves coordinates
def click_event(event, x, y, flags, params):
    coords = params['coords']
    img = params['image']
    
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x,y))
        cv2.circle(img, (x,y), 2, (255, 255, 255), 3)
        cv2.imshow('image', img)

# Iniaties interactive creation of face mesh
def get_face_mesh(img, coords):
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event, {'image': img, 'coords': coords})
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Computes the mean between sets of points
def compute_mean(all_points, warp_frac=-1):
    mean_coords = []
    if warp_frac == -1:
        for i in range(len(all_points[0])):
            x_mean = np.mean([points[i][0] for points in all_points])
            y_mean = np.mean([points[i][1] for points in all_points])
            mean_coords.append((x_mean, y_mean))
    else:
        assert len(all_points) == 2
        for i,j in zip(all_points[0], all_points[1]):
            weighted_mean_x = warp_frac * i[0] + (1 - warp_frac) * j[0]
            weighted_mean_y = warp_frac * i[1] + (1 - warp_frac) * j[1]
            mean_coords.append((weighted_mean_x, weighted_mean_y))

    print(f'MEAN_PTS = {mean_coords}')
    return mean_coords

# Draws the face mesh between mean coordinates
def plot_lines(im, mean_coords, indices):
    for tri in indices:
        pts = [mean_coords[tri[0]], mean_coords[tri[1]], mean_coords[tri[2]]]
        pts = np.array(pts, np.int32).reshape((-1,1,2))
        cv2.polylines(im, [pts], True, (255,255,255), 1)
    
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return im

# Part 1. Defining Correspondences
def section1():
    a_coords = []
    im1 = cv2.imread('./data/edan1.jpg', 1)
    get_face_mesh(im1, a_coords)
    print(f'IMA_PTS = {a_coords}')
   
    b_coords = []
    im2 = cv2.imread('./data/tom1.jpg', 1)
    get_face_mesh(im2, b_coords)
    print(f'IMB_PTS = {b_coords}')
    
    im1 = cv2.imread('./data/edan1.jpg', 1)
    im2 = cv2.imread('./data/tom1.jpg', 1)
    
    # Compute the mean coordinate btwn two meshes
    all_points = [a_coords, b_coords]
    mean_pts = compute_mean(all_points)
    mean_tri_indcies = get_triangle_indices(mean_pts)
    
    im1_mesh = plot_lines(im1, mean_pts, mean_tri_indcies)
    #cv2.imwrite("output/edan-mesh.jpg", im1_mesh)
    im2_mesh = plot_lines(im2, mean_pts, mean_tri_indcies)
    #cv2.imwrite("output/tom-mesh.jpg", im2_mesh)

# Generate the affine transformation matrix 
def compute_affine(tri1, tri2):
    A = np.vstack([np.array(tri1).T, np.ones(3)])
    X = np.vstack([np.array(tri2).T, np.ones(3)])
    return X.dot(np.linalg.inv(A))

# Get all Affine Transfomation from image to mean
def get_affine_transfomations(im_tri, mean_tri):
    matrices = []
    for i in range(len(mean_tri)):
        A = compute_affine(im_tri[i], mean_tri[i])
        if np.all(np.float32(im_tri[i]) == mean_tri[i]):
            A = np.eye(3)
        matrices.append(A)
    return matrices

# Creates an array of the triangle vertices
def create_triangles(pts, indices):
    triangles = []
    for tri in indices:
        triangles.append([pts[tri[0]], pts[tri[1]], pts[tri[2]]])
    return triangles

# Return indices of triangle vertices
def get_triangle_indices(pts):
    return Delaunay(pts).simplices

# A full image mask of the mean face mesh
'''
def create_mean_mask(im, mean_tri):
    full_mask = np.zeros(im.shape)
    for i, tri in enumerate(mean_tri):
        mask = np.zeros(im.shape)
        r = np.array([vert[1] for vert in tri])
        c = np.array([vert[0] for vert in tri])
        rr, cc = polygon(r, c)
        mask[rr, cc] = 1
        full_mask += mask
    return full_mask
'''

# Transforms every triangle of the image to new coordinate space
def warp_affine(im, im_tri, im_matrices):
    # Create a coordinate matrix
    rows, cols = im.shape[0], im.shape[1]
    old_coords = np.indices((cols, rows))
    old_coords = old_coords.reshape(2, -1)
    old_coords = np.vstack((old_coords, np.ones(rows*cols)))
    old_coords = old_coords.astype(int)

    # Initialize resulting image and face mask
    result = np.zeros(im.shape)

    # Iterate through each triangle
    for i, tri in enumerate(im_tri):
        # Creates a mask of the src triangle shape
        mask = np.zeros(im.shape)
        r = np.array([vert[1] for vert in tri])
        c = np.array([vert[0] for vert in tri])
        rr, cc = polygon(r, c)
        mask[rr, cc] = 1

        # Applies mask to original image
        masked_im = im * mask

        # Creates new coordinate system
        M = im_matrices[i]
        new_coords = np.dot(M, old_coords).astype(int)

        # Only keep indices that are valid
        indices = np.all((new_coords[1]<rows, new_coords[1]>=0, new_coords[0]<cols, new_coords[0]>=0), axis=0)

        # Set result pixels in new coordinate system based on pixel values of old coordinate system
        result[new_coords[1][indices], new_coords[0][indices]] += masked_im[old_coords[1][indices], old_coords[0][indices]]
    
    # Intialize coordinate grid for image
    X, Y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1))
    
    # Get coordinates that are non-black and non-white
    color_coords = np.column_stack(np.where(result > 0))
    ix = color_coords[:,0]
    iy = color_coords[:,1]

    # Interpolate missing data for each color channel 
    print(f'Interpolating image...')
    for i in range(3):
        values = result[ix, iy, i]
        result[:,:,i] = griddata((ix, iy), values, (Y, X), method='nearest')

    return result

# Warp image into the given shpa
def warp_image(im, im_pts, tri, tri_indices):
    # Create triangle mesh for image
    im_tri = create_triangles(im_pts, tri_indices)
    
    # Get transformation matrix for each triangle
    im_m = get_affine_transfomations(im_tri, tri)

    # Warp the images into mean mesh
    result = warp_affine(im, im_tri, im_m)
    
    return result * 255

# Cross dissolve two images based on weight
def cross_disolve(images, dissolve_frac):
    if dissolve_frac == -1:
        # Take the average color of all images in list
        result = np.zeros(images[0].shape)
        for image in images:
            result += np.array(image) * 1/len(images)
        return result
    else:
        # Take the weighted average color of two images
        assert len(images) == 2
        return dissolve_frac * images[0] + (1 - dissolve_frac) * images[1]

# Warp images and then cross-dissolve colors
def morph(images, points, warp_frac=-1, dissolve_frac=-1):
    # Compute the triangle mesh for images to be warped into
    mean_pts = compute_mean(points, warp_frac=warp_frac)
    tri_indices = get_triangle_indices(mean_pts)
    tri = create_triangles(mean_pts, tri_indices)
    
    # Warp each image into the given triangle mesh
    results = []
    for i in range(len(images)):
        print(f"Warping image {i+1}")
        result = warp_image(images[i], points[i], tri, tri_indices)
        results.append(result)
        #cv2.imwrite(f'output/warped-{i}'.jpg', result)

    # Cross dissolve colors
    morphed_im = cross_disolve(results, dissolve_frac=dissolve_frac)
    
    return morphed_im

# Part 2. Computing the "Mid-way Face"
def section2():
    # Read in the two faces
    im1 = cv2.imread('./data/edan1.jpg', 1).astype(np.float32)/255.0
    im2 = cv2.imread('./data/tom1.jpg', 1).astype(np.float32)/255.0

    # Set up the image and point lists
    images = [im1, im2]
    points = [IMA_PTS, IMB_PTS]
    
    # Generate the mean image (shape and color have equal weightss)
    mean_im = morph(images, points, warp_frac=0.5, dissolve_frac=0.5)

    cv2.imshow('image', mean_im/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #cv2.imwrite('output/mean-img.jpg', mean_im)

# Part 3. The Morph Sequence
def section3(morph_dir='./morph'):
    # Read in start and end images
    im1 = cv2.imread('./data/edan1.jpg', 1).astype(np.float32)/255.0
    im2 = cv2.imread('./data/tom1.jpg', 1).astype(np.float32)/255.0

    # Set up the image and point lists
    images = [im1, im2]
    points = [IMA_PTS, IMB_PTS]

    # Generate each intermediate frame of morph sequence
    for t in np.round(np.linspace(0, 1, 45), 5):
        print(f'Morph-{t}')
        morph_im = morph(images, points, warp_frac=t, dissolve_frac=t)
        #cv2.imwrite(f'{morph_dir}/{t}.jpg', morph_im)

    # Create the GIF for morph sequence
    filenames = list(filter(lambda f: not f.startswith('.'), os.listdir(morph_dir)))
    filenames = list(reversed(sorted(filenames)))
    #imageio.mimsave('movie.gif', [imageio.imread(f'{morph_dir}/{f}') for f in filenames])

# Read a .pts file from population dataset
def read_pt_file(filepath):
    text = open(filepath).read()
    points = text[text.find("{")+1:text.find("}")].strip()
    points = [(float((pt.split(' ')[0])), float(pt.split(' ')[1])) for pt in points.split('\n')]
    return points

# Read a .jpg file from population dataset
def read_img_file(filepath):
    return cv2.imread(filepath).astype(np.float32)/255.0

# Returns all the #a.pts files as a list of points
def read_files(dir, is_image=True, file_suffix='a'):
    filenames = os.listdir(dir)
    filenames = list(sorted(filenames, key=lambda f: int(f[:f.find('.') - 1])))
    
    all_data = []
    for filename in filenames:
        data = None
        if filename.find(file_suffix) > 0:
            if is_image:
                data = read_img_file(f'{dir}/{filename}')
            else:
                data = read_pt_file(f'{dir}/{filename}')
            all_data.append(data)
    return all_data

def interpolate_helper(im, cols=250, rows=300, thresh=0.6):
    # Intialize coordinate grid for image
    X, Y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1))
    
    # Get coordinates that are non-black and non-white
    color_coords = np.column_stack(np.where(im < thresh))
    ix = color_coords[:,0]
    iy = color_coords[:,1]

    # Interpolate missing data for each color channel 
    print(f'Interpolating image...')
    for i in range(3):
        values = im[ix, iy, i]
        im[:,:,i] = griddata((ix, iy), values, (Y, X), method='nearest')
    return im

def get_population_mean():
    # Read in image files
    all_images = read_files('./sec4_images', is_image=True)
    # Read in point files
    all_points = read_files('./sec4_points', is_image=False)[:100]

    # Make sure all vertices are within bounds of the image
    for points in all_points:
        for i, coords in enumerate(points):
            if coords[0] >= 250:
                points[i] = (249.99, coords[1])
            if coords[1] >= 300:
                points[i] = (coords[0], 299.99)
        points.extend([(0,0), (249,0), (0,299), (249,299)])

    # Compute triangle mesh for mean face of population
    mean_pts = compute_mean(all_points)
    mean_tri_indices = get_triangle_indices(mean_pts)
    mean_tri = create_triangles(mean_pts, mean_tri_indices)
    return mean_tri, mean_tri_indices

# Part 4. The "Mean face" of a population
def section4():
    mean_tri, mean_tri_indices = get_population_mean()

    # 4.1 Compute the average face shape of the whole population or some subset of the population
    '''
    pop_mean_mesh = plot_lines(np.zeros(all_images[0].shape), mean_pts, mean_tri_indices)
    cv2.imwrite('./output/pop_mean_mesh.jpg', pop_mean_mesh)
    '''

    # 4.2 Morph each of the faces in the dataset into the average shape. Show us some examples.
    '''
    for i in range(4):
        warped = warp_image(all_images[i], all_points[i], mean_tri, mean_tri_indices)

        cv2.imshow('image', warped/255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imwrite(f'./output/warped-ex-{i}.jpg', warped)
    '''
    
    # 4.3 Compute the average face of the population
    '''
    print("Creating mean face")
    mean_face = morph(all_images, all_points)

    cv2.imshow('image', mean_face/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'mean_face.jpg', mean_face)
    '''

    # Extra: My Face
    edan_face = cv2.imread('./output/edan_pop.jpg', 1).astype(np.float32)/255.0

    face_coords = []
    #get_face_mesh(edan_face, face_coords)
    #print(f'KEY_PTS = {face_coords}')

    assert(len(EDAN_KEY_PTS)) == 50

    # Warp my face into the mean face
    '''
    edan_warped = warp_image(edan_face, EDAN_KEY_PTS, mean_tri, mean_tri_indices)
    edan_warped = interpolate_helper(edan_warped, thresh=170)

    cv2.imshow('image', edan_warped/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite(f'./output/warped_edan.jpg', edan_warped)
    '''

    # Warp the mean face into my face
    '''
    mean_face = cv2.imread('./output/mean_face.jpg', 1).astype(np.float32)/255.0
    edan_tri = create_triangles(EDAN_KEY_PTS, mean_tri_indices)

    mean_warped = warp_image(mean_face, mean_pts, edan_tri, mean_tri_indices)
    mean_warped = interpolate_helper(mean_warped, thresh=110)

    cv2.imshow('image', mean_warped/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite(f'./output/warped_mean.jpg', mean_warped)
    '''
    
def combine_triangles(tri1, tri2, f):
    new_tri = []
    for i in range(len(tri1)):
        new_verts = [None] * 3
        for j in range(3):
            new_verts[j] = tuple(map(f, tri1[i][j], tri2[i][j]))
        new_tri.append(new_verts)
    return new_tri

# Part 5. Caricatures: Extrapolating from the mean
def section5():
    # Read in my face
    edan_face = cv2.imread('./output/edan_pop.jpg').astype(np.float32)/255.0

    # Get mean face triangles and my face triangles
    mean_tri, tri_indices = get_population_mean()
    edan_tri = create_triangles(EDAN_KEY_PTS, tri_indices)

    # Compute the difference between my face and mean
    diff_tri = combine_triangles(edan_tri, mean_tri, lambda i, j: int(i - j))

    # Create a caricature mesh by adding differences to my face
    caricature_tri = combine_triangles(edan_tri, diff_tri, lambda i, j: int(i + j))

    # Warp my face into caricature mesh
    caricature = warp_image(edan_face, EDAN_KEY_PTS, caricature_tri, tri_indices)
    caricature = interpolate_helper(caricature, thresh=200)

    cv2.imshow('image', caricature/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite(f'./output/caricature.jpg', caricature)

# Bells and Whistles. Create morph sequence for K-12 Edan
def bells_and_whistles():
    all_points = [points for points in edan_k12_dict.values()]
    mean_pts = compute_mean(all_points)
    mean_tri_indices = get_triangle_indices(mean_pts)
    mean_tri = create_triangles(mean_pts, mean_tri_indices)

    images = []
    for i in range(13):
        # Resize all the headshots
        '''
        face = cv2.imread(f'./data/edan-{i}-src.jpg')
        face_resized = resize_img(face, dim=(300,350))
        cv2.imwrite(f'./data/edan-{i}-resized.jpg', face_resized)
        '''
        
        # Get all key points
        '''
        face = cv2.imread(f'./data/edan-{i}-resized.jpg')
        face_coords = []
        get_face_mesh(face, face_coords)
        print(f'KEY_PTS_{i} = {face_coords}')
        '''

        # Read in all the K-12 images
        im = cv2.imread(f'./data/edan-{i}-resized.jpg').astype(np.float32)/255.0
        images.append(im)
        
        '''
        # Visualize face meshes
        im = plot_lines(im, mean_pts, mean_tri_indices)
        cv2.imshow('image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

    # Generate each intermediate frame of morph sequence
    '''
    for i in range(12):
        for t in np.round(np.linspace(0, 1, 11), 2):
            print(f'Morph-{t}')
            curr_images = [images[i+1], images[i]]
            curr_points = [all_points[i+1], all_points[i]]
            morph_im = morph(curr_images, curr_points, warp_frac=t, dissolve_frac=t)
            cv2.imwrite(f'./k12_morph/edan-{i}-{i+1}-{t}.jpg', morph_im)
    '''

    # Create the GIF for morph sequence
    filenames = list(filter(lambda f: not f.startswith('.'), os.listdir('./k12_morph/')))
    filenames = list((sorted(filenames)))
    filenames = filenames[:22] + filenames[44:] + filenames[22:44]
    filenames = filenames + [filenames[-1] for _ in range(3)]
    #imageio.mimsave('./output/k12_movie.gif', [imageio.imread(f'./k12_morph/{f}') for f in filenames]) 

def main():
    # Create argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--section", required=False, help="first operand")
    args = vars(ap.parse_args())
    
    if args['section'] == '1':
        section1()
    elif args['section'] == '2':
        section2()
    elif args['section'] == '3':
        section3()
    elif args['section'] == '4':
        section4()
    elif args['section'] == '5':
        section5()
    elif args['section'] == 'b':
        bells_and_whistles()
    else:
        print("No section selected")

# driver function
if __name__=="__main__":
    main()
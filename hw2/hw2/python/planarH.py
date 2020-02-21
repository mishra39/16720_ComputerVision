import numpy as np
import cv2
import pdb
import scipy
import skimage.io
import skimage.color
from PIL import Image, ImageDraw, ImageFilter

from opts import get_opts
from matplotlib import pyplot as plt
from matchPics import matchPics
from statistics import stdev
from skimage import img_as_float64
opts = get_opts()


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    # x1 = x1.T
    # x2 = x2.T

    A = np.empty([2*x1.shape[0],9])

    for ind in range(x1.shape[0]):
        u_1=  x1[ind,0] # Extract x and y from each from of x1
        v_1 = x1[ind,1]
        u_2=  x2[ind,0] # Extract x and y from each from of x2
        v_2 = x2[ind,1]
        # u_2, v_2 =  x2[ind,:] # Extract x and y from each from of x2
        A[2*ind] = [-u_1, -v_1, -1, 0,0,0, u_1*u_2, u_2*v_1, u_2]    #[0,0,0, -u_2,-v_2,-1, v_1*u_2, v_1*v_2, v_1]
        A[2*ind+1] = [0,0,0,-u_1,-v_1,-1,v_2*u_1,v_2*v_1,v_2]#[u_2, v_2, 1, 0,0,0, -u_2*u_1, -v_2*u_1, -u_1]#[0,0,0, -u_2,-v_2,-1, v_1*u_2, v_1*v_2, v_2]
    # pdb.set_trace()
    # print(x1.shape)
    # print(A.shape)
    U,S,V_t = np.linalg.svd(A)
    eig_val = S[-1] # the smallest eignvalue
    eig_vect = V_t[-1,:] / V_t[-1,-1] # the eigenvector corresponding to smallest eignvalue
    # print(V_t.shape)

    H2to1  = eig_vect
    H2to1 = H2to1.reshape(3,3)
    # H2to1 = H2to1/np.linalg.norm(H2to1)
    return H2to1

def computeH_norm(x1, x2):
	#Q2.2.2
    #mean of the points
    mean_x1 = np.mean(x1[:,0])
    mean_y1 = np.mean(x1[:,1])

    mean_x2 = np.mean(x2[:,0])
    mean_y2 = np.mean(x2[:,1])
    sum_s_den_x1 = 0
    sum_s_den_x2 = 0
    s_x1 = np.empty((x1.shape[0]))
    s_x2 = np.empty((x2.shape[0]))

    # pdb.set_trace()
    for i in range(x1.shape[0]):
        s_x1[i] =  np.sqrt((x1[i,0]-mean_x1)**2 + (x1[i,1]-mean_y1)**2)

    for i in range(x2.shape[0]):
        s_x2[i] = np.sqrt((x2[i,0]-mean_x2)**2 + (x2[i,1]-mean_y2)**2)

    s1 = np.sqrt(2)/ np.max(s_x1)
    # t1_x = -s1*mean_x1
    # t1_y = -s1*mean_y1
    s1_mat = np.array([[s1,0,0],[0,s1,0],[0,0,1]])
    trans1_mat = np.array([[1,0,-mean_x1],[0,1,-mean_y1],[0,0,1]])
    T1 = np.dot(s1_mat,trans1_mat)

    s2 = np.sqrt(2) / np.max(s_x2)
    # t2_x = -s2*mean_x2
    # t2_y = -s2*mean_y2

    s2_mat = np.array([[s2,0,0],[0,s2,0],[0,0,1]])
    trans2_mat = np.array([[1,0,-mean_x2],[0,1,-mean_y2],[0,0,1]])
    T2 = np.dot(s2_mat,trans2_mat)

	#Similarity transform 1n
    x1_hom = np.hstack((x1,np.ones((x1.shape[0],1))))
    # x2_hom = np.hstack((x2,np.ones((x2.shape[0],1))))
    # x1 = np.transpose(x1)
    # x1 = np.vstack(x1,np.ones((1,x1.shape[1])))
    x1_hom = T1@x1_hom.T

	#Similarity transform 2
    x2_hom = np.hstack((x2,np.ones((x2.shape[0],1))))
    # x2 = np.transpose(x2)
    # x2 = np.vstack(x2,np.ones((1,x2.shape[1])))
    x2_hom = T2@x2_hom.T

	#Compute homography
    H2to1_hom = computeH(x1_hom,x2_hom)
	#Denormalization
    H2to1 = np.dot(np.linalg.inv(T2),H2to1_hom)
    H2to1 = np.dot(H2to1,T1)

    return H2to1


def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points

    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    bestH2to1 = np.empty([3,3]) #3 by 3 empty matrix for the best homograph
    # error = np.empty([2,1])
    rand_1 = np.empty([2,4])
    rand_2 = np.empty([2,4])
    max_inliers = -1

    x1 = locs1
    x2 = locs2
    # x1 = np.transpose(x1)
    # x2 = np.transpose(x2)
    x1_hom = np.hstack((x1,np.ones((x1.shape[0],1))))
    x2_hom = np.hstack((x2,np.ones((x2.shape[0],1))))


    for ind in range(max_iters):
        tot_inliers = 0
        ind_rand = np.random.choice(locs1.shape[0],4)#,replace=False)

        # print(chosen_matches)
        rand_1 = locs1[ind_rand,:]
        rand_2 = locs2[ind_rand,:]

        H_norm = computeH_norm(rand_1,rand_2) # Homography between locs1 and locs2
        # H_norm, status = cv2.findHomography(rand_1,rand_2)


        for i in range(x2_hom.shape[0]):

            pred_x2 = np.dot(H_norm,x1_hom[i].T) #Predicted Values. Does it matter if you predict x1 or x2?
            pred_x2[0] = pred_x2[0]/pred_x2[2]
            pred_x2[1] = pred_x2[1]/pred_x2[2]

            error_1 = (x2_hom[i][0] - pred_x2[0])
            error_2 = (x2_hom[i][1] - pred_x2[1])
            error = [error_1, error_2]
            error = np.linalg.norm(error)
            if error <= inlier_tol:
                tot_inliers += 1

        if tot_inliers > max_inliers:
            bestH2to1 = H_norm
            max_inliers = tot_inliers

            #     for i in range(matches.shape[0]):
            # pred_x1 = np.dot(H_norm,x2_hom[i].T)
            # print(pred_x1)
            # pred_x1 = pred_x1 / pred_x1[2]
            # pred_no1 = np.transpose([pred_x1[0], pred_x1[1]])
            # error = np.linalg.norm(x1[i]-pred_no1)
            # print(pred_no1)
            # print(x1[i])
            # print(error)
            # bestH2to1 = H_norm

        # print(dist)

        # for indx in range(matches.shape[0]):
        #     err_dist = np.sqrt(error[0,indx]**2 + error[1,indx]**2)
        #     # print(err_dist)
            # print(err_dist)


        # inliers[one_val] = 1 #Setting valid locations to 1
        # zero_val = np.where(error_x > inlier_tol) #Locations where matches are not a part of the consensus
        # inliers[zero_val] = 0 #Setting valid locations to 0
        # tot_inliers = np.sum(inliers)
        # if tot_inliers > max_inliers:
        #     max_inliers = tot_inliers

    inliers = max_inliers
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    # template is the hp image resized to cv_cover
    # img is the desk
 	#Create a composite image after warping the template image on top
 	#of the image using the homography
 	#Note that the homography we compute is from the image to the template;
 	#For warping the template to the image, we need to invert it.
    mask_ones = np.ones(template.shape)
    mask_ones = cv2.transpose(mask_ones)
    warp_mask = cv2.warpPerspective(mask_ones, H2to1, (img.shape[0],img.shape[1]))
    template = cv2.transpose(template)

    warp_mask = cv2.transpose(warp_mask)
    non_zero_ind = np.nonzero(warp_mask)

    warp_template = cv2.warpPerspective(template, H2to1, (img.shape[0],img.shape[1]))
    warp_template = cv2.transpose(warp_template)
    img[non_zero_ind] = warp_template[non_zero_ind]
    composite_img = img

    # warped_hp = cv2.warpPerspective(template,(H2to1) ,(img.shape[1], img.shape[0]))

    # img[non_zero_ind] = 0
    # composite_img = warped_hp + img
    # plt.imshow(composite_img)

 	#Warp mask by appropriate homography

 	#Warp template by appropriate homography

 	#Use mask to combine the warped template and the image


    return composite_img


def centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = (points.shape[0])

    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return [centroid_x, centroid_y]

if __name__ == '__main__':
    # print('hi')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    matches, locs1, locs2 = matchPics(cv_cover,cv_cover ,opts)
    # a,b = computeH_ransac(matches, locs1, locs2, opts)
    # H_py, status = cv2.findHomography(locs1, locs2)
    H = computeH_norm(locs1, locs2)
    # pdb.set_trace()
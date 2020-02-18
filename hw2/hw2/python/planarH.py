import numpy as np
import cv2
import pdb
import scipy
import skimage.io
import skimage.color
from opts import get_opts
from matplotlib import pyplot as plt
from matchPics import matchPics

opts = get_opts()


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    A = np.empty([2*x1.shape[0],9])

    for ind in range(x1.shape[0]):
        u_1=  x1[ind,0] # Extract x and y from each from of x1
        v_1 = x1[ind,1]
        u_2=  x2[ind,0] # Extract x and y from each from of x2
        v_2 = x2[ind,1]
        # u_2, v_2 =  x2[ind,:] # Extract x and y from each from of x2


        A[2*ind] = [-u_1, -v_1, -1, 0,0,0, u_1*u_2, u_2*v_1, u_2]    #[0,0,0, -u_2,-v_2,-1, v_1*u_2, v_1*v_2, v_1]
        A[2*ind+1] = [0,0,0,-u_1,-v_1,-1,v_2*u_1,v_2*v_1,v_2]#[u_2, v_2, 1, 0,0,0, -u_2*u_1, -v_2*u_1, -u_1]#[0,0,0, -u_2,-v_2,-1, v_1*u_2, v_1*v_2, v_2]

    U,S,V_t = np.linalg.svd(A)
    eig_val = S[-1] # the smallest eignvalue
    eig_vect = V_t[-1,:] / V_t[-1,-1] # the eigenvector corresponding to smallest eignvalue
    H2to1  = eig_vect
    H2to1 = H2to1.reshape(3,3)
    # H2to1 = H2to1/np.linalg.norm(H2to1)
    return H2to1

def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
    centroid_x1_x, centroid_x1_y = centroid(x1)
    centroid_x2_x, centroid_x2_y = centroid(x2)

	#Shift the origin of the points to the centroid
    x1[:,0] = x1[:,0] - centroid_x1_x
    x1[:,1] = x1[:,1] - centroid_x1_y

    x2[:,0] = x2[:,0] - centroid_x2_x
    x2[:,1] = x2[:,1] - centroid_x2_y

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1[:,0] = x1[:,0] / np.max(x1[:,0])
    x1[:,1] = x1[:,1] / np.max(x1[:,1])

    x2[:,0] = x2[:,0] / np.max(x2[:,0])
    x2[:,1] = x2[:,1] / np.max(x2[:,1])

	#Similarity transform 1


	#Similarity transform 2



	#Compute homography
    H2to1 = computeH(x1,x2)
	#Denormalization

    return H2to1


def computeH_ransac(matches, locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points

    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    bestH2to1 = np.empty([3,3]) #3 by 3 empty matrix for the best homograph
    # error = np.empty([2,1])
    rand_1 = np.empty([2,4])
    rand_2 = np.empty([2,4])
    max_inliers = -1


    x1 = locs1[matches[:,0]]
    x2 = locs2[matches[:,1]]

    # x1 = np.transpose(x1)
    # x2 = np.transpose(x2)
    x1_hom = np.hstack((x1,np.ones((x1.shape[0],1))))
    x2_hom = np.hstack((x2,np.ones((x2.shape[0],1))))


    for ind in range(max_iters*10):
        tot_inliers = 0
        ind_rand = np.random.choice(matches.shape[0],4,replace=False)
        chosen_matches = matches[ind_rand]
        # print(chosen_matches)
        rand_1 = locs1[chosen_matches[:,0]]
        rand_2 = locs2[chosen_matches[:,1]]

        H_norm = computeH(rand_1,rand_2) # Homography between locs1 and locs2
        # H_norm, status = cv2.findHomography(rand_1,rand_2)


        for i in range(x2_hom.shape[0]):

            pred_x2 = np.dot(H_norm,x1_hom[i].T) #Predicted Values
            pred_x2[0] = pred_x2[0]/pred_x2[2]
            pred_x2[1] = pred_x2[1]/pred_x2[2]

            error_1 = (x2_hom[i][0] - pred_x2[0])
            error_2 = (x2_hom[i][1] - pred_x2[1])
            error = [error_1, error_2]
            error = np.linalg.norm(error)
            print(error)
            if error <= inlier_tol:
                tot_inliers += 1
                print(tot_inliers)

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

	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
    x_img = np.dot(np.linalg.inv(H2to1),

	#Create mask of same size as template
    mask = np.zeros(template.shape[1], template.shape[0])

	#Warp mask by appropriate homography
    warp_mask = np.dot(H2to1, mask)

	#Warp template by appropriate homography
    warp_mask = np.dot(H2to1, template)

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
    print('hi')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    matches, locs1, locs2 = matchPics(cv_cover,cv_desk,opts)
    a,b = computeH_ransac(matches, locs1, locs2, opts)
    # H_py, status = cv2.findHomography(locs1, locs2)
    # H = computeH_norm(locs1, locs2)
    # pdb.set_trace()
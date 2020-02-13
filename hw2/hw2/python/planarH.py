import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    A = np.empty(x1.shape[0],9)

    for ind in range(x1.shape[0]):
        u_1, v_1, =  x1[ind,:] # Extract x and y from each from of x1
        u_2, v_2, =  x2[ind,:] # Extract x and y from each from of x2
        A[ind,:] = [-u_2, -v_2, -1, 0,0,0, u_2*u_1, v_2*u_1, u_1]
        A[ind+1,:] = [0,0,0, -u_2,-v_2,-1, v_1*u_2, v_1*v_2, v_1]

    U,S,V_t = np.linalg.svd(A)
    eig_val = S[-1] # the smallest eignvalue
    print(V_t.shape)
    eig_vect = V[-1,:] # the eigenvector corresponding to smallest eignvalue
    H2to1  = np.dot(A,eig_vect)
    print(H2to1.shape)
    H2to1 = H2to1.reshape(3,3)

	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points


	#Shift the origin of the points to the centroid


	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)


	#Similarity transform 1


	#Similarity transform 2


	#Compute homography


	#Denormalization


	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier




	return bestH2to1, inliers



def compositeH(H2to1, template, img):

	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.


	#Create mask of same size as template

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image

	return composite_img



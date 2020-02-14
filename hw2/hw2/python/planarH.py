import numpy as np
import cv2
import pdb
import scipy

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
    A = np.empty([x1.shape[0],9])

    for ind in range(x1.shape[0]-1):
        u_1, v_1, =  x1[ind,:] # Extract x and y from each from of x1
        u_2, v_2, =  x2[ind,:] # Extract x and y from each from of x2
        A[ind,:] = [-u_2, -v_2, -1, 0,0,0, u_2*u_1, v_2*u_1, u_1]
        A[ind+1,:] = [0,0,0, -u_2,-v_2,-1, v_1*u_2, v_1*v_2, v_1]

    U,S,V_t = np.linalg.svd(A)
    eig_val = S[-1] # the smallest eignvalue
    print(V_t.shape)
    eig_vect = V_t[-1,:] # the eigenvector corresponding to smallest eignvalue
    H2to1  = eig_vect
    print(H2to1)
    H2to1 = H2to1.reshape(3,3)

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
    inliers = np.empty([locs1.shape[0],1],dtype=int)  #Empty matrix of size N by 2 for inliers
    bestH2to1 = np.empty([3,3]) #3 by 3 empty matrix for the best homograph
    max_inliers = 0

    for ind in range(max_iters):
        # print("Iteration: ",ind)
        # print(locs1.shape)
        # print(locs2.shape)
        x_match = locs1[matches[:,0],0:]
        print(x_match.shape)
        rand_1 = np.random.choice(matches[:,0],4)
        rand_2 = np.random.choice(matches[:,1],4)
        x1_rand = locs1[rand_1]
        x2_rand = locs2[rand_2]
        H_norm = computeH_norm(x1_rand, x2_rand) # Homography between locs1 and locs2

        locs1 = np.transpose(locs1)
        locs2 = np.transpose(locs2)
        x1_hom = np.vstack((locs1,np.ones((1,locs1.shape[1]))))
        x2_hom = np.vstack((locs2,np.ones((1,locs2.shape[1]))))

        pred_x1 = np.dot(H_norm,x2_hom) #Predicted Values
        print(pred_x1.shape)
        print(x2_hom.shape)
        # error_x = scipy.spatial.distance.cdist(pred_x1, x1_hom, metric='euclidean') # error for x coordinates
        error_x = np.sqrt((pred_x1[0,:]**2 - x1_hom[0,:]**2) + (pred_x1[1,:]**2 - x1_hom[1,:]**2))
        print(error_x.shape)
        one_val = np.where(error_x <= inlier_tol) #Locations where matches are part of the consensus
        inliers[one_val] = 1 #Setting valid locations to 1
        zero_val = np.where(error_x > inlier_tol) #Locations where matches are not a part of the consensus
        inliers[zero_val] = 0 #Setting valid locations to 0
        tot_inliers = np.sum(inliers)
        if tot_inliers > max_inliers:
            max_inliers = tot_inliers
            bestH2to1 = H_norm

    # print(inliers)
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


def centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = (points.shape[0])

    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return [centroid_x, centroid_y]
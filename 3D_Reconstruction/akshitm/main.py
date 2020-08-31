# Insert your package here
import numpy as np
import helper
import pdb
import scipy.optimize
import submission
import matplotlib.pyplot as plt
import cv2

connections_3d = [[0,1], [1,3], [2,3], [2,0], [4,5], [6,7], [8,9], [9,11], [10,11], [10,8], [0,4], [4,8], [1,5], [5,9], [2,6], [6,10], [3,7], [7,11]]
color_links = [(255,0,0),(255,0,0),(255,0,0),(255,0,0),(0,0,255),(255,0,255),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,0,255),(0,0,255),(0,0,255),(0,0,255),(255,0,255),(255,0,255),(255,0,255),(255,0,255)]
colors = ['blue','blue','blue','blue','red','magenta','green','green','green','green','red','red','red','red','magenta','magenta','magenta','magenta']

# 2.1
pts = np.load('../data/some_corresp.npz')
pts1 = pts['pts1']
pts2 = pts['pts2']
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
M = np.max(im1.shape)

F = submission.eightpoint(pts1, pts2, M) # EightPoint algrithm to find F
# F7 = submission.sevenpoint(pts1, pts2, M) # EightPoint algrithm to find F

# np.savez('q2_1.npz', F=F, M=M)
# # helper.displayEpipolarF(im1, im2, F) # Visualize result

# # # 3.1
# # import camera instrinsics
K = np.load('../data/intrinsics.npz')
K1 = K['K1']
K2 = K['K2']
E = submission.essentialMatrix(F, K1, K2)

# # # 4.1
# np.savez('q4_1.npz', F = F, pts1 = pts1, pts2 = pts2)
# sel_pts1 , sel_pts2 = helper.epipolarMatchGUI(im1, im2, F)

# # 5.1
pts = np.load('../data/some_corresp_noisy.npz')
pts1 = pts['pts1']
pts2 = pts['pts2']

# # Without RANSAC
F = submission.eightpoint(pts1, pts2, M)
# # helper.displayEpipolarF(im1,im2,F)

# # RANSAC
nIters = 110
tol = 0.85
F,inliers = submission.ransacF(pts1, pts2, M, nIters, tol)
# print("Acccuracy of Ransac: ", (np.count_nonzero(inliers)/len(inliers)))

F = submission.eightpoint(pts1[inliers,:], pts2[inliers,:], M)
# np.savez('F_ransac.npz',F=F, inliers = inliers)

E = submission.essentialMatrix(F, K1, K2)

# # helper.displayEpipolarF(im1, im2, F)
# # 5.3
M1 = np.eye(3)
M1 = np.hstack((M1, np.zeros([3,1])))

M2_all = helper.camera2(E)

C1 = np.dot(K1 , M1)
err_val = np.inf

for i in range(M2_all.shape[2]):

    C2 = np.dot(K2 , M2_all[:,:,i])
    w,err = submission.triangulate(C1, pts1, C2, pts2)

    if err < err_val:
        err_val = err
        M2 = M2_all[:,:,i]
        C2_best = C2
        w_best = w

P_init,err = submission.triangulate(C1, pts1[inliers,:], C2_best, pts2[inliers,:])
M2_opt, P2 = submission.bundleAdjustment(K1, M1, pts1[inliers,:], K2, M2, pts2[inliers,:], P_init)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.set_xlim3d(np.min(P_init[:,0]),np.max(P_init[:,0]))
# ax.set_ylim3d(np.min(P_init[:,1]),np.max(P_init[:,1]))
# ax.set_zlim3d(np.min(P_init[:,2]),np.max(P_init[:,2]))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.scatter(P_init[:,0],P_init[:,1],P_init[:,2])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.set_xlim3d(np.min(P2[:,0]),np.max(P2[:,0]))
# ax.set_ylim3d(np.min(P2[:,1]),np.max(P2[:,1]))
# ax.set_zlim3d(np.min(P2[:,2]),np.max(P2[:,2]))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.scatter(P2[:,0],P2[:,1],P2[:,2])
# plt.show()

# 6.1
i=0
time_0 = np.load('../data/q6/time'+str(i)+'.npz')
print('Time: ', (i))
pts1 = time_0['pts1'] # Nx3 matrix
pts2 = time_0['pts2'] # Nx3 matrix
pts3 = time_0['pts3'] # Nx3 matrix
M1_0 = time_0['M1']
M2_0 = time_0['M2']
M3_0 = time_0['M3']
K1_0 = time_0['K1']
K2_0 = time_0['K2']
K3_0 = time_0['K3']
C1_0 = np.dot(K1_0,M1_0)
C2_0 = np.dot(K1_0,M2_0)
C3_0 = np.dot(K1_0,M3_0)
Thres = (np.average(pts1[:,2]) + np.average(pts2[:,2])  + np.average(pts3[:,2]))/3
M3, err = submission.MultiviewReconstruction(C1_0, pts1, C2_0, pts2, C3_0, pts3, Thres)

# for i in range(3):

#     time_0 = np.load('../data/q6/time'+str(i+4)+'.npz')
#     print('time: ',i+4)
#     pts1 = time_0['pts1'] # Nx3 matrix
#     pts2 = time_0['pts2'] # Nx3 matrix
#     pts3 = time_0['pts3'] # Nx3 matrix
#     M1_0 = time_0['M1']
#     M2_0 = time_0['M2']
#     M3_0 = time_0['M3']
#     K1_0 = time_0['K1']
#     K2_0 = time_0['K2']
#     K3_0 = time_0['K3']
#     C1_0 = np.dot(K1_0,M1_0)
#     C2_0 = np.dot(K1_0,M2_0)
#     C3_0 = np.dot(K1_0,M3_0)
#     Thres = (np.average(pts1[:,2]) + np.average(pts2[:,2])  + np.average(pts3[:,2]))/3
#     # print('Threshold: ', Thres)

#     P_mv, err_mv = submission.MultiviewReconstruction(C1_0, pts1, C2_0, pts2, C3_0, pts3, Thres)

# # 6.2
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(10):
#     time_0 = np.load('../data/q6/time'+str(i)+'.npz')
#     pts1 = time_0['pts1'] # Nx3 matrix
#     pts2 = time_0['pts2'] # Nx3 matrix
#     pts3 = time_0['pts3'] # Nx3 matrix
#     M1_0 = time_0['M1']
#     M2_0 = time_0['M2']
#     M3_0 = time_0['M3']
#     K1_0 = time_0['K1']
#     K2_0 = time_0['K2']
#     K3_0 = time_0['K3']
#     C1_0 = np.dot(K1_0,M1_0)
#     C2_0 = np.dot(K1_0,M2_0)
#     C3_0 = np.dot(K1_0,M3_0)
#     Thres = (np.average(pts1[:,2]) + np.average(pts2[:,2])  + np.average(pts3[:,2]))/3
#     print('Threshold: ', Thres)

#     P2_opt, err_mv = submission.MultiviewReconstruction(C1_0, pts1, C2_0, pts2, C3_0, pts3, Thres)

#     num_points = P2_opt.shape[0]
#     np.set_printoptions(threshold=1e6, suppress=True)
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     for j in range(len(connections_3d)):
#         index0, index1 = connections_3d[j]
#         xline = [P2_opt[index0,0], P2_opt[index1,0]]
#         yline = [P2_opt[index0,1], P2_opt[index1,1]]
#         zline = [P2_opt[index0,2], P2_opt[index1,2]]
#         ax.plot(xline, yline, zline, color=colors[j])
# # plt.show()

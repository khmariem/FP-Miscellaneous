import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def extract_features(img_path, sil_path=None):
    img = cv.imread(img_path)
    if sil_path!=None:
        sil = cv.imread(sil_path)
        img = cv.multiply(img, sil)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #surf = cv.xfeatures2d.SURF_create()
    #correspondences, descriptor = surf.detectAndCompute(img,None)
    sift = cv.xfeatures2d.SIFT_create()
    correspondences, descriptor = sift.detectAndCompute(gray,None)

    return correspondences, descriptor

def match(correspondences1, correspondences2, descriptor1, descriptor2):
    #match the found features
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 50)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptor1,descriptor2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    good=[]
    pts1=[]
    pts2=[]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
            pts2.append(correspondences2[m.trainIdx].pt)
            pts1.append(correspondences1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2, matches, matchesMask

def triangulate_point(P1, P2, x1, x2):

    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[:2, 4] = -x1
    M[3:,:4] = P2
    M[3:5, 5] = -x2
    M[5,5] = -1

    U, S, V = np.linalg.svd(M)
    X = V[-1,:4]

    return X/X[3]

def triangulate(P1, P2, X1, X2):
    X=[]
    for i in range(X1.shape[0]):
        X.append(triangulate_point(P1, P2, X1[i,:], X2[i,:]))
    
    return np.array(X).T


def choose(P1, P2, pts1, pts2):

    X = triangulate(P1[:3],P2[:3],pts1,pts2)
    depth1 = np.dot(P1,X)[2]
    depth2 = np.dot(P2,X)[2]
    infront = (depth1>0) & (depth2>0)
    
    return infront

def plot_it(X):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(-X[0], X[1], X[2], 'k.')
    plt.axis('off')
    plt.show()

def main(path1, path2, P1, P2, path11=None, path21=None):
    # Feature extraction
    img1 = cv.imread(path1)
    correspondences1, descriptor1 = extract_features(path1, path11)

    img2 = cv.imread(path2)
    correspondences2, descriptor2 = extract_features(path2, path21)

    # Match the found feature points
    pts1, pts2, matches, matchesMask = match(correspondences1, correspondences2, descriptor1, descriptor2)

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1,correspondences1,img2,correspondences2,matches,None,**draw_params)
    plt.imshow(img3)
    plt.show()
    
    infront = choose(P1, P2, pts1, pts2)
    
    X = triangulate(P1[:3], P2[:3], pts1, pts2)
    X=X[:, infront]

    return X

if __name__ == "__main__":


    #Next.... read all text files and create the projection matrices for every two images
    X=None
    for i in range(32):
        if i<9:
            n1="/home/mariem/Downloads/beethoven_data/calib/000"+str(i)+".txt"
            path1 = "/home/mariem/Downloads/beethoven_data/images/000"+str(i)+".ppm"
            path11 = "/home/mariem/Downloads/beethoven_data/silhouettes/000"+str(i)+".pgm"
            n2="/home/mariem/Downloads/beethoven_data/calib/000"+str(i+1)+".txt"
            path2 = "/home/mariem/Downloads/beethoven_data/images/000"+str(i+1)+".ppm"
            path21 = "/home/mariem/Downloads/beethoven_data/silhouettes/000"+str(i+1)+".pgm"
        elif i==9:
            n1="/home/mariem/Downloads/beethoven_data/calib/000"+str(i)+".txt"
            path1 = "/home/mariem/Downloads/beethoven_data/images/000"+str(i)+".ppm"
            path11 = "/home/mariem/Downloads/beethoven_data/silhouettes/000"+str(i)+".pgm"
            n2="/home/mariem/Downloads/beethoven_data/calib/00"+str(i+1)+".txt"
            path2 = "/home/mariem/Downloads/beethoven_data/images/00"+str(i+1)+".ppm"
            path21 = "/home/mariem/Downloads/beethoven_data/silhouettes/00"+str(i+1)+".pgm"
        else:
            n1="/home/mariem/Downloads/beethoven_data/calib/00"+str(i)+".txt"
            path1 = "/home/mariem/Downloads/beethoven_data/images/00"+str(i)+".ppm"
            path11 = "/home/mariem/Downloads/beethoven_data/silhouettes/00"+str(i)+".pgm"
            n2="/home/mariem/Downloads/beethoven_data/calib/00"+str(i+1)+".txt"
            path2 = "/home/mariem/Downloads/beethoven_data/images/00"+str(i+1)+".ppm"
            path21 = "/home/mariem/Downloads/beethoven_data/silhouettes/00"+str(i+1)+".pgm"

        m1 = np.zeros((3,4))
        m2 = np.zeros((3,4))

        f1 = open(n1,"r")
        f1.readline()
        k=0
        for line in f1:
            val = line.split()
            for l in range(len(val)):
                m1[k,l] = float(val[l])
            k+=1
        f1.close()

        f2 = open(n2,"r")
        f2.readline()
        k=0
        for line in f2:
            val = line.split()
            for l in range(len(val)):
                m2[k,l] = float(val[l])
            k+=1
        f2.close()

        X2=main(path1, path2, m1, m2,path11,path21)
        if X is None:
            X = X2
        else:
            X = np.hstack([X, X2])
        plot_it(X2)
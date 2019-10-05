import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import cv2 as cv
from math import sqrt
from random import *

def from_quaternion(quaternion):

    "Generate the rotation matrix from the quaternion"

    eps = random()/1000.0
    q = np.array(quaternion)
    sqq = np.dot(q,q)

    if sqq<eps:
        return np.identity(4)
    
    q *= sqrt(2.0 / sqq)
    q = np.outer(q,q)

    R = np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

    return R

def calculate_transform(pose):

    M = from_quaternion(pose[1])
    M[:3, 3] = pose[0]

    return M

def read_pose(posefile,obj_ind):
    with open(posefile,'r') as f:
        for i in range(4*obj_ind+3):
            l = f.readline()
        l=f.readline().strip('\n').split(': ')[1].strip(']').strip('[').strip(', ').split('], [')
        t = l[0].split(', ')
        t=[float(el) for el in t]
        r = fl[1].split(', ')
        r=[float(el) for el in r]

        p=[t,r]
    return p


def apply_transform(filename, M):

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_poly_data= reader.GetOutput()
    bounds = vtk_poly_data.GetBounds()
    points = np.ones((8,4))
    points[:4,0]=bounds[0]
    points[4:,0]=bounds[1]
    points[:2,1]=bounds[2]
    points[2:4,1]=bounds[3]
    points[4:6,1]=bounds[2]
    points[6:,1]=bounds[3]
    points[:,2] = bounds[5]
    points[0:2:,2] =bounds[4]

    proj_points = np.dot(M, points.T)

    bbox = [np.max(proj_points[:,0]),np.max(proj_points[:,1]), np.min(proj_points[:,0]), np.min(proj_points[:,1])]

    return bbox

if __name__ == "__main__":
    img = cv.imread("logs/my6  1/images/0000000001_rgb.png")
    posefile="logs/my6  1/images/0000000001_poses.yaml"
    pose = read_pose(posefile,1)
    M = calculate_transform(pose)
    filename = "sd.vtp"
    bbx = apply_transform(filename, M)
    print(bbx)
    cv.rectangle(img,(bbx[2], bbx[3]),(bbx[0],bbx[1]),(0,255,0),2)
    cv.imshow("Show",img)
    cv.waitKey()  
    cv.destroyAllWindows()

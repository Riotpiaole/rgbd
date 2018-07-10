

import os
import math

import cv2
import numpy as np

from utils import *
from utils_camera import *

def depthimg_bilateral(depthImg, maxdiff=3.0):
    """ Smooth depth image using a bilateral filter ignoring large differences """
    depthImg_bilateral = cv2.bilateralFilter(np.float32(depthImg), 15, 20.0, 20.0)
    h, w = depthImg.shape
    finalDepth = np.zeros([h,w])
    for y in range(0,h):
        for x in range(0, w):
            cur = depthImg[y,x]
            smo = depthImg_bilateral[y,x]
            if abs(smo-cur) < maxdiff: finalDepth[y,x] = smo
            else: finalDepth[y,x] = cur
    return finalDepth
    

def read_ColorImage(strFilePath, imgResize=False, resizeSize=(240, 320)):
    """ Given a path to an image, read it into a numpy array. Resize as required. """
    # clrImg = misc.imread(strFilePath)
    # if imgResize: clrImg = misc.imresize(clrImg, resizeSize, interp='bicubic')
    clrImg = cv2.imread(strFilePath) # any old color image loading
    if imgResize: clrImg =  cv2.resize(clrImg, (resizeSize[1], resizeSize[0]), interpolation=cv2.INTER_AREA)
    return clrImg

def read_depth16bit_image(strFilePath, imgResize=False, resizeSize=(240, 320)):
    """ Read the a 16bit depth image data into a numpy array. """
    depthImg =  cv2.imread(strFilePath,-1) # regular 16-bit PNG loading
    if imgResize: depthImg = cv2.resize(depthImg, (resizeSize[1], resizeSize[0]), interpolation=cv2.INTER_NEAREST)
    return depthImg


def image_fusion(camera_params, depthData, clrImg=None, normals=None,threshold=False):
    """
        Given a depth image and its corresponding color image, return a colored point cloud as a vector of (x, y, z, r, g, b).
        Assume only depth and color, and if provided with normals, fuse those too.
        The output format is a PLY (required to view it in color in MeshLab).
    """
    # nanLocationsDepth = np.isnan(depthData)
    # numberOfVertices = depthData.size - np.count_nonzero(nanLocationsDepth)
    # depthData[nanLocationsDepth] = -1 # replace all NaN values with -1
    

    bHasColors = clrImg is not None
    bHasNormals = normals is not None
    if bHasNormals:
        nanLocationsNormals = np.isnan(normals)
        normals[nanLocationsNormals] = 0

    h, w = depthData.shape

    # generate point cloud via numpy array functions
    coords = np.indices((h, w))
    
    # geometry
    xcoords = (((coords[1] - camera_params.cx)/camera_params.fx)*depthData).flatten()
    ycoords = (((coords[0] - camera_params.cy)/camera_params.fy)*depthData).flatten()
    zcoords = depthData.flatten()
    
    # color
    chan_red = chan_blue = chan_green = None
    if bHasColors:
        # chan_red = clrImg[..., 0].flatten()
        # chan_blue = clrImg[..., 1].flatten()
        # chan_green = clrImg[..., 2].flatten()

        chan_red = clrImg[..., 2].flatten()
        chan_blue = clrImg[..., 1].flatten()
        chan_green = clrImg[..., 0].flatten()

    ptcloud = None

    # normals
    normalsX = normalsY = normalsZ = None
    if bHasNormals:
        normalsX = normals[..., 0].flatten()
        normalsY = normals[..., 1].flatten()
        normalsZ = normals[..., 2].flatten()

    

    if bHasColors and bHasNormals: ptcloud = np.dstack((xcoords, ycoords, zcoords, normalsX, normalsY, normalsZ, chan_red, chan_blue, chan_green))[0]
    elif bHasColors and not bHasNormals: ptcloud = np.dstack((xcoords, ycoords, zcoords, chan_red, chan_blue, chan_green))[0]
    elif not bHasColors and bHasNormals:  ptcloud = np.dstack((xcoords, ycoords, zcoords, normalsX, normalsY, normalsZ))[0]
    else: ptcloud = np.dstack((xcoords, ycoords, zcoords))[0]
    # depth filtering
    if threshold:
        filters =np.where(ptcloud[:,2] >= 2800 )
        ptcloud = np.delete(ptcloud[:] ,filters ,0 )
        
    return ptcloud, ptcloud.size//6

def output_pointcloud(nVertices, ptcloud, strOutputPath, bHasNormals=False):
    """
        Given a point cloud produced from image_fusion, output it to a PLY file.
        TODO: Consider having a separate flag to allow for outputing just the depth and not colors.
    """
    # open the file and write out the standard ply header
    outputFile = open(strOutputPath + ".ply", "w")
    outputFile.write("ply\n")
    outputFile.write("format ascii 1.0\n")
    outputFile.write("comment generated via python script Process3DImage\n")
    outputFile.write("element vertex %d\n" %(nVertices))
    outputFile.write("property float x\n")
    outputFile.write("property float y\n")
    outputFile.write("property float z\n")

    if bHasNormals:
        outputFile.write("property float nx\n")
        outputFile.write("property float ny\n")
        outputFile.write("property float nz\n")

    outputFile.write("property uchar red\n")
    outputFile.write("property uchar green\n")
    outputFile.write("property uchar blue\n")
    outputFile.write("element face 0\n")
    outputFile.write("property list uchar int vertex_indices\n")
    outputFile.write("end_header\n")

    # output the actual points
    for pt in ptcloud:
        dx, dy, dz = pt[0:3]

        dx *= 0.001
        dy *= 0.001
        dz *= 0.001

        if bHasNormals:
            nx, ny, nz, r, g, b = pt[3:]
            outputFile.write("%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %d %d %d\n" %(-dx, dy, dz, nx, ny, nz, r, g, b))
        else:
            r, g, b = pt[3:]
            outputFile.write("%10.6f %10.6f %10.6f %d %d %d\n" %(dx, dy, dz, r, g, b))

    outputFile.close()


def images_to_textured_mesh(img_color, img_depth, camera_params, tri_dist_thres=0.75):
    """ Given an rgb-d image pair, return a textured mesh using the spherical model. """
    ptcloud = []
    triangles = []
    texcoords = []

    img_h, img_w = img_depth.shape
    n_points_total = img_w*img_h
    n_points_processed = 0

    for h in range(0, img_h):
        for w in range(0, img_w):
            depth_val = img_depth[h, w]
            r, g, b = img_color[h, w]


            xcoord = ( (w - camera_params.cx)/camera_params.fx )*depth_val
            ycoord = ( (h - camera_params.cy)/camera_params.fy )*depth_val
            zcoord = depth_val

            ptcloud.append([xcoord, ycoord, zcoord, r, g, b])

            # texcoords.append([1.0 - w_ratio, 1.0 - h_ratio])
            texcoords.append([w/img_w, 1.0-h/img_h])

            n_points_processed+=1
            print_text_progress_bar(n_points_processed/n_points_total, bar_name='Images to textured mesh   ')
    print()
 
    skip_val = 2
    n_triangles_processed = 0
    for h in range(0, img_h,2):
        for w in range(0, img_w,2):
            idx_00 = h*img_w + w         ## top left
            idx_10 = h*img_w + w + skip_val     ## top right
            idx_01 = (h+skip_val)*img_w + w     ## bottom left
            idx_11 = (h+skip_val)*img_w + w + skip_val ## bottom right

            if (idx_00 < len(ptcloud)) and (idx_10 < len(ptcloud)) and (idx_01 < len(ptcloud)) and (idx_11 < len(ptcloud)):
                pt_tl = np.array(ptcloud[idx_00][0:3])
                pt_tr = np.array(ptcloud[idx_10][0:3])
                pt_bl = np.array(ptcloud[idx_01][0:3])
                pt_br = np.array(ptcloud[idx_11][0:3])

                ## compute pair-wise distances and exclude triangles who have edges which are too long
                d_tltr = np.linalg.norm(pt_tl-pt_tr)
                d_trbr = np.linalg.norm(pt_tr-pt_br)
                d_brbl = np.linalg.norm(pt_br-pt_bl)
                d_bltl = np.linalg.norm(pt_bl-pt_tl)
                d_trbl = np.linalg.norm(pt_tr-pt_bl)

                if (d_tltr < tri_dist_thres) and (d_trbr < tri_dist_thres) and (d_brbl < tri_dist_thres) and (d_bltl < tri_dist_thres) and (d_trbl < tri_dist_thres):
                    triangles.append([idx_00+1, idx_01+1, idx_10+1]) ## triangle 1
                    triangles.append([idx_10+1, idx_01+1, idx_11+1]) ## triangle 2


            n_triangles_processed += 2
            print_text_progress_bar(n_triangles_processed/(img_h*img_w*2), bar_name='Triangulating....   ')
    print()

    return np.array(ptcloud), np.array(triangles), np.array(texcoords), n_points_total

def output_textured_mesh(ptcloud, texcoords, triangles, fname_image_texture, fname_obj, strOutputPath):
    texture_file = open(os.path.join(strOutputPath, fname_image_texture) + ".mtl", "w")
    texture_file.write("newmtl material0\n")
    texture_file.write("Ka 1.000000 1.000000 1.000000\n")
    texture_file.write("Kd 1.000000 1.000000 1.000000\n")
    texture_file.write("Ks 0.000000 0.000000 0.000000\n")
    texture_file.write("Tr 1.000000\n")
    texture_file.write("illum 1\n")
    texture_file.write("Ns 0.000000\n")
    texture_file.write("map_Kd "+fname_image_texture+".png\n")
    texture_file.close()

    print("Writing point cloud to '" + str(strOutputPath + fname_obj + ".obj") + "' ... ", end="", flush=True)
    outputFile = open(os.path.join(strOutputPath, fname_obj) + ".obj", "w")
    outputFile.write("mtllib ./"+fname_image_texture+".mtl\n")

    # output the actual points
    for pt in ptcloud:
        dx, dy, dz = pt[0:3]
        outputFile.write("v %.6f %.6f %.6f\n" %(dx, dy, dz))
        outputFile.write("vn 0.0 0.0 0.0\n")

    for vt in texcoords:
        tex_u, tex_v = vt
        outputFile.write("vt %.6f %.6f\n" %(tex_u, tex_v))

    outputFile.write("usemtl material0\n")
    for t in triangles:
        idx_1, idx_2, idx_3 = t
        outputFile.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" %(idx_1, idx_1, idx_1, idx_2, idx_2, idx_2, idx_3, idx_3, idx_3))

    outputFile.close()

    print("Finished!")


if __name__ == "__main__":
    # strInputPath = "/home/po/Documents/data/body_reconstruction_deep_learning/00_testdataset_01/snaps01/"
    # strOutputPath = strInputPath
    # process_multiview(strInputPath, strOutputPath, 3)

    for i in range(1,4):
        strInputPath = "F:/data/deep-learning/additive_depth/98_vr_calib/test01_img_seq/cam0"+str(i)+"/"
        strColorFolder = os.path.join(strInputPath, "color")
        strDepthFolder= os.path.join(strInputPath, "depth")

        strColorFilename = os.path.join(strColorFolder, "1519264284736.jpg")
        strDepthFilename = os.path.join(strDepthFolder, "1519264284736.png")

        img_color = cv2.imread(strColorFilename)
        img_depth = cv2.imread(strDepthFilename, -1)

        params = GetCameraParameters("OrbbecAstra", 0.5)

        ## output textured mesh
        strOutputPath = "F:/data/deep-learning/additive_depth/98_vr_calib/test01_calib_frames/"
        if not os.path.exists(strOutputPath): os.makedirs(strOutputPath)
        texture_img_name = "cam0"+str(i)+"-image-texture"

        cv2.imwrite(os.path.join(strOutputPath, texture_img_name + '.png'), img_color)
        # plt.imsave(, cv2.flip( (imgC_clr/255), 1 ))
        # ptcloud, triangles, texcoords, n_points_total = images_to_textured_mesh(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB), pred_00)

        ptcloud, triangles, texcoords, n_points_total = images_to_textured_mesh(img_color, img_depth/1000, params, tri_dist_thres=0.25)
        output_textured_mesh(ptcloud, texcoords, triangles, texture_img_name, "cam0"+str(i), strOutputPath)
        print()


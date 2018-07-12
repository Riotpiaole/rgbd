import os 

# Folder of all the images
strVideoFolder = "/Users/rockliang/Documents/Research/VISION/RGBD/unity-multiview/data"
strFolderName = "test01"
strFolderNameBlack = "test01black"


# Poses of all the cameras 
strVideoFullPath = os.path.join(strVideoFolder, strFolderName)
strPoseLocation = os.path.join(strVideoFullPath, "unity3d_poses.txt")

# folder storing filtered images
strFilterFolder="/Users/rockliang/Documents/Research/VISION/RGBD/rgbd_prediction/data"
strFilterFullPath = os.path.join(strFilterFolder,strFolderName)
strFilterFullPathBlack = os.path.join(strFilterFolder,strFolderNameBlack)


# Parameters for filtering the image depth
cam1_calib={ 'maxDep':3000 , 'lx':40,'rx':250,'yd':None   ,'open':(2,2),'close':(2,2), 'ksize':(5,5) ,'dsize':10, 'bilater':(30,50) , 'grad':500 , 'connectivity':520}
cam2_calib={ 'maxDep':3500 , 'lx':140,'rx':None,'yd':215  ,'open':(2,2),'close':(2,2), 'ksize':(5,5) ,'dsize':8,  'bilater':(10,65) , 'grad':60  , 'connectivity':8}
cam3_calib={ 'maxDep':3500 , 'lx':50,'rx': 220 ,'yd':160  ,'open':(3,3),'close':(5,5), 'ksize':(3,3) ,'dsize':7,  'bilater':(10,30) , 'grad':2000  , 'connectivity':20}

filter_param={ 
    strFolderName:[cam1_calib,cam2_calib,cam3_calib]
}
    
input_shape = (240,320,3)

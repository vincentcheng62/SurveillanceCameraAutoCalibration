
"""
Module to calibrate the extrinsic matrix between chessboard on the ground and camera coordinate
"""

import numpy as np
import cv2 as cv
import cv2
import time, datetime
from matplotlib import pyplot as plt
import operator
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linspace
import argparse
from imutils.object_detection import non_max_suppression
from numpy.linalg import inv
from math import log10, floor
import pyproj


def draw_axis_and_ptgrid(img, corners, imgpts, ptgrid, IsPlotPtGrid):
    """
    function to draw the World XYZ axis and point grid(optional) after calibration is done
    """
    count=0
    for con in corners:
        cv.drawChessboardCorners(img, pattern_size, con, True)

        #draw axis
        corner = tuple(con[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[count][0].ravel()), (0,0,255), 5)
        img = cv.line(img, corner, tuple(imgpts[count][1].ravel()), (0,255,0), 5)
        img = cv.line(img, corner, tuple(imgpts[count][2].ravel()), (255,0,0), 5)

        count=count+1

    # Also draw a point grid in the z-plane for debug purpose
    if IsPlotPtGrid:
        for pt in ptgrid:
            img = cv.circle(img, tuple(pt.ravel()), 4, (0,255,0), thickness=2, lineType=8, shift=0) 

    return img


if __name__ == '__main__':

    np.set_printoptions(suppress=True)

    # read parameters from extrinsic_calibration_input.yml
    print("Reading input from extrinsic_calibration_input.yml...")
    fs_read = cv2.FileStorage("extrinsic_calibration_input.yml", cv2.FILE_STORAGE_READ)

    IsSaveToVideo = eval(fs_read.getNode("IsSaveToVideo").string())
    print("IsSaveToVideo: ", IsSaveToVideo)  
    IsBgProcessing = eval(fs_read.getNode("IsBgProcessing").string())
    print("IsBgProcessing: ", IsBgProcessing)  
    IsSaveIntermediateImageToDebug = eval(fs_read.getNode("IsSaveIntermediateImageToDebug").string())
    print("IsSaveIntermediateImageToDebug: ", IsSaveIntermediateImageToDebug)  
    IsPlottingPtGrid = eval(fs_read.getNode("IsPlottingPtGrid").string())
    print("IsPlottingPtGrid: ", IsPlottingPtGrid)  
    IsDebugCBCanFound = eval(fs_read.getNode("IsDebugCBCanFound").string())
    print("IsDebugCBCanFound: ", IsDebugCBCanFound)  
    IsLiveWindowDisplay = eval(fs_read.getNode("IsLiveWindowDisplay").string())
    print("IsLiveWindowDisplay: ", IsLiveWindowDisplay)  
    IsUsingPredefinedMaskForCB = eval(fs_read.getNode("IsUsingPredefinedMaskForCB").string())
    print("IsUsingPredefinedMaskForCB: ", IsUsingPredefinedMaskForCB)  


    videopath = fs_read.getNode("videopath").string()
    print("videopath: ", videopath)  
    square_size_in_m = fs_read.getNode("square_size_in_m").real()
    print("square_size_in_m: ", square_size_in_m)
    pattern_size_x = int(fs_read.getNode("pattern_size_x").real())
    print("pattern_size_x: ", pattern_size_x)
    pattern_size_y = int(fs_read.getNode("pattern_size_y").real())
    print("pattern_size_y: ", pattern_size_y)   
    grid_pt_size = int(fs_read.getNode("grid_pt_size").real())
    print("grid_pt_size: ", grid_pt_size)        
    video_save_fps = int(fs_read.getNode("video_save_fps").real())
    print("video_save_fps: ", video_save_fps)   
    skip_sec = int(fs_read.getNode("skip_sec").real())
    print("skip_sec: ", skip_sec)       
    SupposeCBNumber = int(fs_read.getNode("SupposeCBNumber").real())
    print("SupposeCBNumber: ", SupposeCBNumber)               
    current_fps_for_skip_sec = int(fs_read.getNode("current_fps_for_skip_sec").real())
    print("current_fps_for_skip_sec: ", current_fps_for_skip_sec)        
    display_axis_length_in_m = fs_read.getNode("display_axis_length_in_m").real()
    print("display_axis_length_in_m: ", display_axis_length_in_m)        
    grid_pt_seperation = fs_read.getNode("grid_pt_seperation").real()
    print("grid_pt_seperation: ", grid_pt_seperation)         
    margin_when_masking_out_founded_chessboard = int(fs_read.getNode("margin_when_masking_out_founded_chessboard").real())
    print("margin_when_masking_out_founded_chessboard: ", margin_when_masking_out_founded_chessboard)
    thrhd_grayvalue_for_whitepart_of_chessboard = int(fs_read.getNode("thrhd_grayvalue_for_whitepart_of_chessboard").real())
    print("thrhd_grayvalue_for_whitepart_of_chessboard: ", thrhd_grayvalue_for_whitepart_of_chessboard)
    dilate_kernel_for_cb_mask_radius = int(fs_read.getNode("dilate_kernel_for_cb_mask_radius").real())
    print("dilate_kernel_for_cb_mask_radius: ", dilate_kernel_for_cb_mask_radius)
    morph_open_kernel_radius = int(fs_read.getNode("morph_open_kernel_radius").real())
    print("morph_open_kernel_radius: ", morph_open_kernel_radius)    
    margin_when_masking_out_founded_chessboard = int(fs_read.getNode("margin_when_masking_out_founded_chessboard").real())
    print("margin_when_masking_out_founded_chessboard: ", margin_when_masking_out_founded_chessboard)

    predefined_mask = fs_read.getNode("predefined_mask").mat()
    print('predefined_mask: ', predefined_mask)

    fs_read.release()


    fs_read = cv2.FileStorage("intrinsic_and_distortion_coeff.yml", cv2.FILE_STORAGE_READ)
    camera_matrix_manual = fs_read.getNode("intrinsic").mat()
    print('intrinsic: ', camera_matrix_manual)
    dist_coefs_manual = fs_read.getNode("distortion_coeff").mat()
    print('distortion_coeff: ', dist_coefs_manual)
    fs_read.release()  




    pattern_size = (pattern_size_x, pattern_size_y)
    cap = cv.VideoCapture(videopath)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size_in_m
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
    FinishCalibration = False
    ref_rvec=None
    rot=None
    ref_tvec=None
    ptgrid = None
    CamCenterInCBCoord=None
    dilate_kernel_for_cb_mask = cv.getStructuringElement(cv.MORPH_ELLIPSE,(dilate_kernel_for_cb_mask_radius, dilate_kernel_for_cb_mask_radius))    
    morph_open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(morph_open_kernel_radius,morph_open_kernel_radius))   
    axis = np.float32([[display_axis_length_in_m,0,0], [0,display_axis_length_in_m,0], [0,0,-display_axis_length_in_m]]).reshape(-1,3)
    OtherCBCorners=[]
    OtherCBImgpt=[]
    calib_corners = None
    calib_imgpt = None
    frame_num=0
    fitting_error=0
    video_writer=None
    now = datetime.datetime.now()
    name = "output" + str(now.strftime("%Y-%m-%d-%H-%M-%S")) + ".avi"

    if IsSaveToVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(name, fourcc, video_save_fps, (1280, 720))
        print("Saving the frames to video output -> ", name)

    CurrentdetectedChessBoardnum=0  
    largest_cb_square_dist=0 # in case SupposeCBNumber>1, select the largest chessboard for calibration   

    skip_frame=skip_sec*current_fps_for_skip_sec
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame) # skip frame

    #Construct the point grid for display
    grid=[]
    grid_raw=[]
    if IsPlottingPtGrid:
        for ii in range(0, grid_pt_size):
            for jj in range(0, grid_pt_size):
                grid_raw.append([(ii-grid_pt_size*0.5)*grid_pt_seperation, (jj-grid_pt_size*0.5)*grid_pt_seperation, 0])

        grid = np.float32(grid_raw).reshape(-1,3)    


    # Main loop
    print("")
    print("Calibration start!")
    print("====================================================================================")
    while(cap.isOpened()):

        frame_num = frame_num+1
        print("Frame no: ", frame_num)

        ret, frame = cap.read()
        if ret == False:
            break
        elif IsSaveToVideo:
            video_writer.write(frame)


        framecopy=frame.copy()

        if frame.any() and not FinishCalibration:

            if IsSaveIntermediateImageToDebug:
                cv.imwrite("frame_" + str(frame_num) + "_raw.jpg", frame)

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            print("CurrentdetectedChessBoardnum: ", CurrentdetectedChessBoardnum)

            #hide the data and camera no
            if IsUsingPredefinedMaskForCB:
                for q in range(0, predefined_mask.shape[0]):
                    cv.rectangle(frame_gray,(int(predefined_mask[q][0]),int(predefined_mask[q][1])),(int(predefined_mask[q][2]),int(predefined_mask[q][3])),0,-1)

            print("largest_cb_square_dist: ", largest_cb_square_dist)

            #Mask out the last detected chessboards, so wont detected again
            if largest_cb_square_dist>0.0 and CurrentdetectedChessBoardnum > 0:
                for cornerss in OtherCBCorners:
                    x,y,w,h = cv.boundingRect(cornerss)
                    cv.rectangle(frame_gray,(x-margin_when_masking_out_founded_chessboard,y-margin_when_masking_out_founded_chessboard),(x+w+margin_when_masking_out_founded_chessboard,y+h+margin_when_masking_out_founded_chessboard),128,-1)


            frame_for_detect_cb = frame_gray.copy()

            # Since chessboard white is close to 255, use this to threshold out the real chessboard and do a mask
            # Mask out unrelvant area can speed up the chessboard discovery
            if IsBgProcessing:
                bw_ret,frame_bw = cv.threshold(frame_gray,thrhd_grayvalue_for_whitepart_of_chessboard,255,cv.THRESH_BINARY)
                frame_bw = cv.morphologyEx(frame_bw, cv.MORPH_OPEN, morph_open_kernel)
                frame_bw = cv.dilate(frame_bw,dilate_kernel_for_cb_mask,iterations = 1)

                # a mask for the left view
                frame_for_detect_cb = cv2.bitwise_and(frame_gray, frame_gray, mask=frame_bw)    

                if IsLiveWindowDisplay:
                    frame_for_detect_cb_half = cv.resize(frame_for_detect_cb, (0,0), fx=0.5, fy=0.5)
                    cv.imshow('chessboard detection region', frame_for_detect_cb_half)     

            found, corners = cv.findChessboardCorners(frame_for_detect_cb, pattern_size,  flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)
            if found:

                print("====================================================================================")
                print("a chessboard is found!")

                if IsSaveIntermediateImageToDebug:       
                    cv.imwrite("frame_" + str(frame_num) + "_gray.jpg", frame_gray)
                    cv.imwrite("frame_" + str(frame_num) + "_for_detect_cb.jpg", frame_for_detect_cb)
                    cv.imwrite("frame_" + str(frame_num) + "_mask.jpg", frame_bw)

                CurrentdetectedChessBoardnum = CurrentdetectedChessBoardnum + 1


                # Prepare the 2d corners pt and 3d world points for cv.solvePnP()
                obj_points = []
                img_points = []   
                cv.cornerSubPix(frame_gray, corners, (5, 5), (-1, -1), term)
                print("corners:", corners)                
                cb_square_dist = (corners[0][0][0]-corners[1][0][0])*(corners[0][0][0]-corners[1][0][0])+(corners[0][0][1]-corners[1][0][1])*(corners[0][0][1]-corners[1][0][1])
                print("cb_square_dist:", math.sqrt(cb_square_dist), "px")

                # If it is the only chessboard or it is a larger chessboard than the previous one
                # Since we use the largest chessboard for calibration, other as test chessboard
                if SupposeCBNumber==1 or cb_square_dist > largest_cb_square_dist:

                    chessboards = [(corners.reshape(-1, 2), pattern_points)]

                    chessboards = [x for x in chessboards if x is not None]
                    for (corners, pattern_points) in chessboards:
                        img_points.append(corners)
                        obj_points.append(pattern_points)
                    
                    returnval, rvecs, tvecs = cv.solvePnP(np.array(obj_points), np.array(img_points),camera_matrix_manual, dist_coefs_manual )

                    print("This chessboard is large enough, use it as calibration chessboard!")
                    largest_cb_square_dist = cb_square_dist
                    ref_rvec = rvecs
                    ref_tvec = tvecs
                    
                    calib_corners = corners

                    projected_grid_pt, _ = cv.projectPoints(grid, rvecs, tvecs, camera_matrix_manual, dist_coefs_manual)
                    ptgrid = projected_grid_pt

                    projected_axis_pt, _ = cv.projectPoints(axis, rvecs, tvecs, camera_matrix_manual, dist_coefs_manual)       
                    OtherCBImgpt.append(projected_axis_pt)
                    calib_imgpt = projected_axis_pt

                    OtherCBCorners.append(corners)

                    # calculate the reprojection error
                    reprojectimgpts, _ = cv.projectPoints(np.array(obj_points), rvecs, tvecs, camera_matrix_manual, dist_coefs_manual)

                    totalfittingerror=0
                    for zz in range(len(reprojectimgpts)):
                        totalfittingerror = totalfittingerror + math.sqrt(math.pow(reprojectimgpts[zz][0][0]-img_points[0][zz][0], 2)+math.pow(reprojectimgpts[zz][0][1]-img_points[0][zz][1], 2))
                    fitting_error = totalfittingerror
                    print("fittingerror of the projected grid points: ", totalfittingerror)


                    # brings the calibration pattern from the model coordinate space (in which object points are specified)
                    # to the world coordinate space, that is, a real position of the calibration pattern
                    # from chessboard (0, 0, 0) to 
                    print("rotation: ",  [x* 180.0 / math.pi for x in rvecs])
                    print("translation: ", cv.transpose(tvecs))     

                    ext = cv.hconcat([np.array(cv.transpose(rvecs)), np.array(cv.transpose(tvecs))])
                    print("extrinsic: ", ext)

            
            else:
                print("Cannot find any chessboard! break!")

                if CurrentdetectedChessBoardnum > 0:
                    idx=1 # print index of different chessboard for display purpose

                    # Draw chessboard corners and index for display purpose
                    for cornerss in OtherCBCorners:
                        x,y,w,h = cv.boundingRect(cornerss)
                        cv.drawChessboardCorners(frame, pattern_size, cornerss, True)
                        cv.putText(frame, str(idx), (x, y), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 5) 
                        idx=idx+1
                    
                if CurrentdetectedChessBoardnum > 1:
                    dilate_kernel_for_cb_mask = cv.getStructuringElement(cv.MORPH_ELLIPSE,(40,40))
                
                if IsLiveWindowDisplay:
                    frame_half = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
                    cv.imshow('live frame', frame_half)     


            # Can stop since we detect all the existing chessboard we have in the FOV
            if CurrentdetectedChessBoardnum == SupposeCBNumber:
                print("====================================================================================")         
                print("CurrentdetectedChessBoardnum == SupposeCBNumber, for loop end")   
                print("Calibration result is: ", cv.transpose(ref_rvec), cv.transpose(ref_tvec))

                rot, _ = cv.Rodrigues(ref_rvec)
                CamCenterInCBCoord=np.matmul(cv.transpose(rot), -ref_tvec)
                print("CamCenterInCBCoord(-RT*Tvec) is: ", cv.transpose(CamCenterInCBCoord))

                print("fitting error of final calibration answer: ", fitting_error)
        
                FinishCalibration = True

            # for testing purpose
            elif CurrentdetectedChessBoardnum > 0:
                print("Found at least one, but not meeting SupposeCBNumber=", SupposeCBNumber)


        #Finish calibration and display
        if frame.any() and FinishCalibration:
            print("====================================================================================")            
            frame_display = frame.copy()
            frame_display = draw_axis_and_ptgrid(frame_display,OtherCBCorners,OtherCBImgpt, ptgrid, IsPlottingPtGrid)  

            fs_write = cv.FileStorage('video_calibrate_extrinsic_result.yml', cv.FILE_STORAGE_WRITE)
            fs_write.write("fitting_error", fitting_error)   
            fs_write.write("rotation", rot)   
            fs_write.write("ref_rvec", ref_rvec)   
            fs_write.write("ref_tvec", ref_tvec)   
            fs_write.write("CamCenterInCBCoord", CamCenterInCBCoord)   
            fs_write.release()

            if IsSaveIntermediateImageToDebug:
                cv.imwrite("extrinsic_calibration_result_with_axis.jpg", frame_display)         

            if IsLiveWindowDisplay:
                half_frame = cv.resize(frame_display, (0,0), fx=(0.5), fy=(0.5)) 
                cv.imshow('result', half_frame)    
            else:
                print("Finish calibration, program exit!")
                break 

           

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if IsSaveToVideo:
        video_writer.release()
    cv.destroyAllWindows()
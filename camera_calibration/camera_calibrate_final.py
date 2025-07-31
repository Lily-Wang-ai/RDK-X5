#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [-w <width>] [-h <height>] [-t <pattern type>] [--square_size=<square size>]
    [--marker_size=<aruco marker size>] [--aruco_dict=<aruco dictionary name>] [<image mask>]

usage example:
    calibrate.py -w 4 -h 6 -t chessboard --square_size=50 ../data/left*.jpg

default values:
    --debug:    ./output/
    -w: 4
    -h: 6
    -t: chessboard
    --square_size: 10
    --marker_size: 5
    --aruco_dict: DICT_4X4_50
    --threads: 4
    <image mask> defaults to ../data/left*.jpg

NOTE: Chessboard size is defined in inner corners. Charuco board size is defined in units.
'''

'''
============================wl:  python calibration_test.py -w 11 -h 8 -t chessboard --square_size=30 ./calibration_images/*.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# # local modules
# from common import splitfn#======wl:直接在下面定义函数代替

# built-in modules
import os

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def main():
    import sys
    import getopt
    from glob import glob

    args, img_names = getopt.getopt(sys.argv[1:], 'w:h:t:', ['debug=','square_size=', 'marker_size=',
                                                      'aruco_dict=', 'threads=', ])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('-w', 4)
    args.setdefault('-h', 6)
    args.setdefault('-t', 'chessboard')
    args.setdefault('--square_size', 10)
    args.setdefault('--marker_size', 5)
    args.setdefault('--aruco_dict', 'DICT_4X4_50')
    args.setdefault('--threads', 4)

    if not img_names:
        img_mask = '../data/left??.jpg'  # default
        img_names = glob(img_mask)

    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)

    height = int(args.get('-h'))
    width = int(args.get('-w'))
    pattern_type = str(args.get('-t'))
    square_size = float(args.get('--square_size'))
    marker_size = float(args.get('--marker_size'))
    aruco_dict_name = str(args.get('--aruco_dict'))

    pattern_size = (width, height)
    if pattern_type == 'chessboard':
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results

    aruco_dicts = {
        'DICT_4X4_50': cv.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv.aruco.DICT_4X4_1000,
        'DICT_5X5_50': cv.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv.aruco.DICT_5X5_1000,
        'DICT_6X6_50': cv.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv.aruco.DICT_6X6_1000,
        'DICT_7X7_50': cv.aruco.DICT_7X7_50,
        'DICT_7X7_100': cv.aruco.DICT_7X7_100,
        'DICT_7X7_250': cv.aruco.DICT_7X7_250,
        'DICT_7X7_1000': cv.aruco.DICT_7X7_1000,
        'DICT_ARUCO_ORIGINAL': cv.aruco.DICT_ARUCO_ORIGINAL,
        'DICT_APRILTAG_16h5': cv.aruco.DICT_APRILTAG_16h5,
        'DICT_APRILTAG_25h9': cv.aruco.DICT_APRILTAG_25h9,
        'DICT_APRILTAG_36h10': cv.aruco.DICT_APRILTAG_36h10,
        'DICT_APRILTAG_36h11': cv.aruco.DICT_APRILTAG_36h11
    }

    if (aruco_dict_name not in set(aruco_dicts.keys())):
        print("unknown aruco dictionary name")
        return None
    aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dicts[aruco_dict_name])
    board = cv.aruco.CharucoBoard(pattern_size, square_size, marker_size, aruco_dict)
    charuco_detector = cv.aruco.CharucoDetector(board)

    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found = False
        corners = 0
        if pattern_type == 'chessboard':
            found, corners = cv.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
                frame_img_points = corners.reshape(-1, 2)
                frame_obj_points = pattern_points
        elif pattern_type == 'charucoboard':
            corners, charucoIds, _, _ = charuco_detector.detectBoard(img)
            if (len(corners) > 0):
                frame_obj_points, frame_img_points = board.matchImagePoints(corners, charucoIds)
                found = True
            else:
                found = False
        else:
            print("unknown pattern type", pattern_type)
            return None

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            if pattern_type == 'chessboard':
                cv.drawChessboardCorners(vis, pattern_size, corners, found)
            elif pattern_type == 'charucoboard':
                cv.aruco.drawDetectedCornersCharuco(vis, corners, charucoIds=charucoIds)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_board.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('pattern not found')
            return None

        print('           %s... OK' % fn)
        return (frame_img_points, frame_obj_points)

    threads_num = int(args.get('--threads'))
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

 # 计算重投影误差
    mean_error = 0
    total_squared_error = 0
    total_points = 0
    
    for i in range(len(obj_points)):
        imgpoints2, _ = cv.projectPoints(obj_points[i], _rvecs[i], _tvecs[i], camera_matrix, dist_coefs)
        # 确保两个数组的形状一致 - 都reshape为(N, 2)
        img_pts = img_points[i].reshape(-1, 2)
        proj_pts = imgpoints2.reshape(-1, 2)
        
        # 方法1: 平均L2距离 (原始方法)
        error = cv.norm(img_pts, proj_pts, cv.NORM_L2)/len(proj_pts)
        mean_error += error
        
        # 方法2: 计算每个点的误差平方和 (类似RMS的计算方式)
        diff = img_pts - proj_pts
        squared_errors = np.sum(diff * diff, axis=1)  # 每个点的误差平方
        total_squared_error += np.sum(squared_errors)
        total_points += len(proj_pts)
    
    # 输出两种计算方法的结果
    mean_reprojection_error = mean_error / len(obj_points)
    rms_style_error = np.sqrt(total_squared_error / total_points)
    
    print("平均重投影误差 (L2距离平均): {:.6f}".format(mean_reprojection_error))
    print("RMS风格重投影误差 (误差平方根): {:.6f}".format(rms_style_error))
    print("OpenCV标定RMS: {:.6f}".format(rms))
    
    # 标定质量评估
    print("\n=== 标定质量评估 ===")
    if rms < 0.3:
        print("标定质量: 非常优秀 ⭐⭐⭐⭐⭐")
        print("适用场景: 高精度测量、3D重建、工业检测")
    elif rms < 0.5:
        print("标定质量: 优秀 ⭐⭐⭐⭐")
        print("适用场景: 精密视觉应用、机器人导航")
    elif rms < 1.0:
        print("标定质量: 良好 ⭐⭐⭐")
        print("适用场景: 目标检测、跟踪、一般测量")
    elif rms < 2.0:
        print("标定质量: 可接受 ⭐⭐")
        print("适用场景: 监控、粗略定位")
        print("建议: 考虑改进标定条件")
    else:
        print("标定质量: 需要改进 ⭐")
        print("建议: 重新标定，改善图像质量和拍摄条件")
    
    print("使用图像数量: {}".format(len(obj_points)))
    print("总角点数: {}".format(total_points))

    # undistort the image with the calibration
    print('')
    for fn in img_names if debug_dir else []:
        _path, name, _ext = splitfn(fn)
        img_found = os.path.join(debug_dir, name + '_board.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
import os 
import cv2 as cv

rgb_dir = "/home/ryo/Dataset/scene_01/rgbs"
photo_dir = "/home/ryo/workspace/rcf/RCF-pytorch/tmp/RCF/initial-testing-record"
geo_dir = "/home/ryo/workspace/maskGeneration/build/tmp"

rgbfiles = os.listdir(rgb_dir)
rgbfiles.sort()
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('test.avi', fourcc, 20.0, (640*3, 480))

for rgb_file in rgbfiles[1:]:
    rgb = cv.imread(os.path.join(rgb_dir, rgb_file))
    photo = cv.imread(os.path.join(photo_dir, rgb_file), 1)
    geo = cv.imread(os.path.join(geo_dir, rgb_file), 1)
    res = cv.hconcat([rgb, photo])
    res = cv.hconcat([res, geo])
    out.write(res)

out.release()
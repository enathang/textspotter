import getdeps
deps = getdeps.getdeps()

import sys
import os
sys.path.insert(0, os.environ['DATA_REPOSITORY']+'/pylayer')
sys.path.insert(0, os.environ['CAFFEROOT']+'/python')
sys.path.insert(0, os.environ['DATA_REPOSITORY'])
from tool import is_image, write2txt_icdar15_e2e, contain_num, \
    non_max_suppression
import matplotlib.pyplot as plt
import argparse
import caffe
import numpy as np
import cv2


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='textspotter')
    parser.add_argument(
        '--weight',
        dest='weight',
        default='./models/text_detection.caffemodel',
        help='the weight file (caffemodel)',
        type=str)
    parser.add_argument(
        '--prototxt',
        dest='prototxt',
        default='./models/text_detection.pt',
        help='prototxt file for detection',
        type=str)
    parser.add_argument(
        '--img',
        dest='img',
        default='./imgs/img_105.jpg',
        help='img file or folder',
        type=str
    )
    parser.add_argument(
        '--thresholds-ms',
        dest='thresholds',
        default='0.95, 0.95, 0.95, 0.95',
        help='multiscale thresholds for text region prediction',
        type=str
    )
    parser.add_argument(
        '--scales-ms',
        dest='scales',
        default='2240, 1920, 1792, 2080',
        help='multiscales for testing',
        type=str
    )
    parser.add_argument(
        '--tile-size',
        dest='tile_size',
        default='1024',
        help='size of tiles',
        type=int
    )
    parser.add_argument(
        '--nms',
        dest='nms',
        default=0.2,
        help='nms threshold',
        type=float
    )

    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        default='../data',
        type=str
    )

    args = parser.parse_args()
    return args


def forward_pass(network, tile, mask_th):
    network.blobs['data'].reshape(*tile.shape)
    network.blobs['data'].data[...] = tile

    fcn_th_blob = np.zeros((1, 1), dtype=np.float32)
    fcn_th_blob[0, 0] = mask_th
    network.blobs['fcn_th'].reshape(*fcn_th_blob.shape)
    network.blobs['fcn_th'].data[...] = fcn_th_blob

    ### determine bboxes
    network.forward()
    bboxes = network.blobs['rois'].data[:, 1:].copy()

    ### determine probability of bboxes
    bboxes_prob = bboxes[:, 8] # bboxes 1-8 contains x or y coords and 9 contains prob
    return bboxes, bboxes_prob


def getTileCoords(im, image_resize_width, image_resize_height, tile_size):
    x_coords = list()
    y_coords = list()
    for n in range(int(np.floor(image_resize_width/tile_size)*2)):
        x_coords.append(n*tile_size/2)
    for n in range(int(np.floor(image_resize_height/tile_size)*2)):
        y_coords.append(n*tile_size/2)

    x_coords.append(image_resize_width-tile_size-1)
    y_coords.append(image_resize_height-tile_size-1)
    return x_coords, y_coords
    

def detect_bboxes(im, network, resize_length, mask_th, tile_size):
    ### resize everything
    h, w, c = im.shape
    scale = max(h, w) / float(resize_length)

    image_resize_height = int(round(h / scale / 32) * 32) # desired height
    image_resize_width = int(round(w / scale / 32) * 32)
    # input()
    scale_h = float(h) / image_resize_height # scale factors
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height))
    im = np.asarray(im, dtype=np.float32)
    mean_val = 122.0 # VARIABLE
    im = im - mean_val
    im = np.transpose(im, (2, 0, 1))
    im = im[np.newaxis, :]
    
    bboxes = list()
    bboxes_prob = list()
    tile_x_coords, tile_y_coords = getTileCoords(im, image_resize_width, image_resize_height, tile_size) # get tile coords

    for x in tile_x_coords:
        for y in tile_y_coords:
            tile = im[:,:,y:y+tile_size,x:x+tile_size] # get tile
            tile_bboxes, tile_bboxes_prob = forward_pass(network, tile, mask_th) # run tile
            tile_bboxes[:, :8:2] += x # undo transformation on boxes
            tile_bboxes[:, 1:8:2] += y
            tile_bboxes[:, :8:2] = tile_bboxes[:, :8:2] * scale_w
            tile_bboxes[:, 1:8:2] = tile_bboxes[:, 1:8:2] * scale_h
            bboxes.extend(np.array(tile_bboxes))
            bboxes_prob.extend(np.array(tile_bboxes_prob))

    final_bboxes = list()
    for n in range(len(bboxes)):
        if bboxes[n][8] > 0.001:
            final_bboxes.append(bboxes[n])

    return final_bboxes, bboxes_prob


def displayImageWithBoxes(im, boxes):
    plt.imshow(im)
    currentAxis = plt.gca()
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    for n in range(len(boxes)):
        coords = np.reshape(boxes[n, 0:8], (-1, 2))
        currentAxis.add_patch(plt.Polygon(coords, fill=False, edgecolor=colors[0], linewidth=2))
            
    plt.show()
 
def boundBoxCoords(boxes):
    for index in range(len(boxes)):
        for i in range(4):
            ### bound the x-coord
            boxes[index][2 * i] = int(round(boxes[index][2 * i]))
            boxes[index][2 * i] = max(0, boxes[index][2 * i])
            boxes[index][2 * i] = min(w - 1, boxes[index][2 * i])
            ### bound the y-coord
            boxes[index][2 * i + 1] = int(round(boxes[index][2 * i + 1]))
            boxes[index][2 * i + 1] = max(0, boxes[index][2 * i + 1])
            boxes[index][2 * i + 1] = min(h - 1, boxes[index][2 * i + 1])
    return boxes

def postProcessBoxes(boxes):
    boxes = np.array(boxes)
    boxes = np.reshape(boxes, [-1,9])
    

    ### non-max suppression
    boxes = np.array(boxes).reshape(-1, 9)
    keep_indices, boxes = non_max_suppression(boxes, args.nms)
    keep_indices = np.int32(keep_indices)
    boxes = boxes[keep_indices]
    #plt.plot(boxes[:,8])
    #plt.show()

    boxes = boundBoxCoords(boxes) # keep box within image
    return boxes

def retrieveImages(img):
    img_files = []
    if os.path.isdir(img):
        imgs = os.listdir(img)
        img_files = [img+_ for _ in imgs if is_image(_)]
    elif os.path.isfile(img):
        img = [img]
        img_files = [_ for _ in img if is_image(_)]
    else:
        assert False, 'invalid input image (folder)'

    if len(img_files) == 0:
        assert False, 'invalid input image (folder)'
    
    return img_files

if __name__ == '__main__':
    ### general setup
    args = parse_args()

    if not os.path.exists(args.prototxt) or \
        not os.path.exists(args.weight):
        assert False, 'please put model and prototxts in ./model/'

    imgs_files = retrieveImages(args.img)
    tile_size = args.tile_size

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    caffe.set_mode_gpu()
    caffe.set_device(0)

    ### create neural net
    net_iou = caffe.Net(args.prototxt, args.weight, caffe.TEST)
    thresholds = [float(_) for _ in args.thresholds.strip().split(',')]
    scales = [int(_) for _ in args.scales.strip().split(',')]

    assert len(thresholds) == len(scales), \
        'the length of thresholds and scales should be equal'

    for ind, image_name in enumerate(imgs_files):
        # setup
        new_boxes = np.zeros((0, 9))
        image_id = image_name.split('/')[-1].split('.')[0]
        print '%d / %d: ' % (ind+1, len(imgs_files)), image_name
        im = cv2.imread(image_name)
        h, w, c = im.shape

        # detect boxes
        for k in range(len(scales)):
            image_resize_length = scales[k]
            mask_threshold = thresholds[k]
            bboxes, det_bboxes_prob = detect_bboxes(im, net_iou, image_resize_length, mask_threshold, tile_size)
            if len(bboxes) > 0:
                new_boxes = np.concatenate([new_boxes, np.array(bboxes)], axis=0)
            print "Done with scale {} out of {}".format(k, len(scales))

        # process boxes
        if len(new_boxes) == 0:
            out_name = os.path.join(args.save_dir, 'res_' + image_id + '.txt')
            new_boxes = np.zeros((0, 8))
            write2txt_icdar15_e2e(out_name, new_boxes)
        else:
            boxes = postProcessBoxes(new_boxes)
            out_name = os.path.join(args.save_dir, 'res_' + image_id + '.txt')
            write2txt_icdar15_e2e(out_name, boxes)
            
            # displayImageWithBoxes(im, boxes)
        





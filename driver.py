import sys
sys.path.insert(0, './pylayer')
sys.path.insert(0, './caffe/python')
from tool import is_image, write2txt_icdar15_e2e, contain_num, \
    non_max_suppression
import matplotlib.pyplot as plt
import argparse
import os
import caffe
import numpy as np
import cv2
import cfg


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
        '--nms',
        dest='nms',
        default=0.2,
        help='nms threshold',
        type=float
    )

    parser.add_argument(
        '--save-dir',
        dest='save_dir',
        default='./results',
        type=str
    )

    args = parser.parse_args()
    return args



def forward_iou(im, net_iou, resize_length, mask_th, tile_size):
    ### resize everything
    h, w, c = im.shape
    scale = max(h, w) / float(resize_length)

    image_resize_height = int(round(h / scale / 32) * 32)
    image_resize_width = int(round(w / scale / 32) * 32)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height))
    # change im.shape to be tile size
    # tile image
    im = np.asarray(im, dtype=np.float32)
    im = im - cfg.mean_val
    im = np.transpose(im, (2, 0, 1))
    im = im[np.newaxis, :]

    net_iou.blobs['data'].reshape(*im.shape)
    net_iou.blobs['data'].data[...] = im

    fcn_th_blob = np.zeros((1, 1), dtype=np.float32)
    fcn_th_blob[0, 0] = mask_th
    net_iou.blobs['fcn_th'].reshape(*fcn_th_blob.shape)
    net_iou.blobs['fcn_th'].data[...] = fcn_th_blob

    ### determine bboxes
    net_iou.forward()
    bboxes = net_iou.blobs['rois'].data[:, 1:].copy()
    bboxes[:, :8:2] = bboxes[:, :8:2] * scale_w
    bboxes[:, 1:8:2] = bboxes[:, 1:8:2] * scale_h

    ### determine probability of bboxes
    bboxes_prob = bboxes[:, 8] # bboxes 1-8 contains x or y coords and 9 contains prob
    return bboxes, bboxes_prob

def displayImageWithBoxes(im, boxes):
    plt.imshow(im)
    currentAxis = plt.gca()
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    print boxes
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

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    caffe.set_mode_gpu()
    caffe.set_device(0)

    ### create neural nets
    net_iou = caffe.Net(args.prototxt, args.weight, caffe.TEST)
    thresholds = [float(_) for _ in args.thresholds.strip().split(',')]
    scales = [int(_) for _ in args.scales.strip().split(',')]

    assert len(thresholds) == len(scales), \
        'the length of thresholds and scales should be equal'

    for ind, image_name in enumerate(imgs_files):
        new_boxes = np.zeros((0, 9))
        image_id = image_name.split('/')[-1].split('.')[0]
        print '%d / %d: ' % (ind+1, len(imgs_files)), image_name
        im = cv2.imread(image_name)
        h, w, c = im.shape

        for k in range(len(scales)):
            image_resize_length = scales[k]
            mask_threshold = thresholds[k]
            det_bboxes, det_bboxes_prob = forward_iou(im, net_iou, image_resize_length, mask_threshold, 0)
            boxes_k = det_bboxes[:].copy().tolist()
            if len(boxes_k) > 0:
                new_boxes = np.concatenate([new_boxes, np.array(boxes_k)], axis=0)

        if len(new_boxes) == 0:
            out_name = os.path.join(args.save_dir, 'res_' + image_id + '.txt')
            new_boxes = np.zeros((0, 8))
            write2txt_icdar15_e2e(out_name, new_boxes)
        else:
            boxes = postProcessBoxes(new_boxes)
            out_name = os.path.join(args.save_dir, 'res_' + image_id + '.txt')
            write2txt_icdar15_e2e(out_name, boxes)
            
            displayImageWithBoxes(im, boxes)
        





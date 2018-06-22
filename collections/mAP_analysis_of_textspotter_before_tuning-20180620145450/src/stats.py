'''
Code partially taken from https://github.com/argman/EAST/issues/92
'''

import numpy as np
import os
import importlib
from shapely.geometry.polygon import Polygon

def polygon_iou(poly1, poly2):
  """
  Intersection over union between two shapely polygons.
  """
  if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
    iou = 0
  else:
    try:
      inter_area = poly1.intersection(poly2).area
      union_area = poly1.area + poly2.area - inter_area
      iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
      print('shapely.geos.TopologicalError occured, iou set to 0')
      iou = 0

  # print iou
  return iou

def loadGroundTruths(file):
    A = np.load(file).item()
    polygons = list()
    for item in A.keys():
        verts = A[item]['vertices']
        verts.append(verts[0])
        polygon = Polygon(verts).convex_hull
        polygons.append(polygon)

    return polygons

def loadPredictions(file):
    polygons = list()
    file = open(file, 'r')
    for line in file.readlines():
        verts = line[:-2].split(',')
        p1 = tuple(verts[0:2])
        p2 = tuple(verts[2:4])
        p3 = tuple(verts[4:6])
        p4 = tuple(verts[6:8])
        verts = np.array([p1, p2, p3, p4, p1])
        polygon = Polygon(verts).convex_hull
        polygons.append(polygon)

    return polygons

def main(truth_file, pred_file):
  '''
    truth_files = []
    pred_files = []
    if os.path.isdir(img):
        imgs = os.listdir(img)
        img_files = [img+_ for _ in imgs]
    elif os.path.isfile(img):
        img = [img]
        img_files = [_ for _ in img]
    else:
        assert False, 'invalid input image (folder)'

  for k in range(len(truth_file)):
'''
  ground_truths = loadGroundTruths(truth_file)
  predictions = loadPredictions(pred_file)

  threshold = 0.5

  correct_pred = 0
  for t in range(len(ground_truths)):
    for p in range(len(predictions)):
      if (polygon_iou(ground_truths[t], predictions[p]) > threshold):
        correct_pred += 1
        
  precision = correct_pred/float(len(predictions))
  recall = correct_pred/float(len(ground_truths))
  print '# predictions: ', len(predictions)
  print '# ground truths: ', len(ground_truths)
  print '# correct: ', correct_pred
  print 'iou threshold: ', threshold
  print 'precision: ', precision
  print 'recall: ', recall
  print 'fscore: ', 2 * precision * recall / (precision + recall)

if __name__ == "__main__":
  main("groundtruths/D0042-1070005.npy", "../data/res_D0042-1070005.txt")

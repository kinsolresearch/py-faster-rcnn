# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# File modified by Anqi Xu, inspired by coco.py
# --------------------------------------------------------

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import uuid
import subprocess
from fast_rcnn.config import cfg

class imdb_vocS_cocoC(imdb): # custom image db, with VOC-style structure and COCO-set classes+labels
    def __init__(self, image_set, year):
        imdb.__init__(self, 'cococustom_' + year + '_' + image_set)
        # Parameters
        self.config = {'use_salt'    : True,  # generic parameter: use uuid to salt comp_id
                       'cleanup'     : True,  # generic parameter: remove temp files created for evaluations
                       'min_size'    : 2,     # selective search: filter small boxes
                       'top_k'       : 2000,  # region proposals: top K proposal to consider
                       'use_diff'    : False, # VOC-specific: exclude samples labeled as difficult
                      }
        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'CUSTOM' + self._year)

        # NOTE: manually extracted from coco._classes
        self._classes = ('__background__', u'person', u'bicycle', u'car', u'motorcycle', u'airplane', u'bus', u'train', u'truck', u'boat', u'traffic light',
                u'fire hydrant', u'stop sign', u'parking meter', u'bench', u'bird', u'cat', u'dog', u'horse', u'sheep', u'cow',
                u'elephant', u'bear', u'zebra', u'giraffe', u'backpack', u'umbrella', u'handbag', u'tie', u'suitcase', u'frisbee',
                u'skis', u'snowboard', u'sports ball', u'kite', u'baseball bat', u'baseball glove', u'skateboard', u'surfboard', u'tennis racket', u'bottle',
                u'wine glass', u'cup', u'fork', u'knife', u'spoon', u'bowl', u'banana', u'apple', u'sandwich', u'orange',
                u'broccoli', u'carrot', u'hot dog', u'pizza', u'donut', u'cake', u'chair', u'couch', u'potted plant', u'bed',
                u'dining table', u'toilet', u'tv', u'laptop', u'mouse', u'remote', u'keyboard', u'cell phone', u'microwave', u'oven',
                u'toaster', u'sink', u'refrigerator', u'book', u'clock', u'vase', u'scissors', u'teddy bear', u'hair drier', u'toothbrush')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # Add synonym class texts (e.g. mapping from ImageNet symnet)
        self._to_coco_class = {
          'n02958343': 'car',
          'aeroplane': 'airplane',
          'diningtable': 'dining table',
          'motorbike': 'motorcycle',
          'sofa': 'couch',
          'pottedplant': 'potted plant',
          'tvmonitor': 'tv',
          }
        for key, value in self._to_coco_class.iteritems():
          self._class_to_ind[key] = self._class_to_ind[value]

        self._image_index = self._load_image_set_index()
        self._image_exts = ['.JPEG', '.JPG', '.jpg']
        self._roidb_handler = self.gt_roidb
        self.competition_mode(False)

    # NOTE: copied from pascal_voc.py
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    # NOTE: copied from pascal_voc.py
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = ''
        for _image_ext in self._image_exts:
          image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + _image_ext)
          if os.path.exists(image_path):
            break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # NOTE: copied from pascal_voc.py
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    # NOTE: modified from pascal_voc.py
    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
      box_list = []
      for index in self.image_index:
        filename = os.path.join(self._data_path, 'Proposals', index + '.xml')
        try:
          ann = self._load_pascal_annotation_xml(filename)
          boxes = ann['boxes']
          keep = ds_utils.unique_boxes(boxes)
          boxes = boxes[keep, :]
          keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
          boxes = boxes[keep, :]
          
        except IOError: # If file does not exist, skip
          print 'Failed to load proposals from %s' % filename
          boxes = []
        box_list.append(boxes)

      return self.create_roidb_from_box_list(np.array(box_list), gt_roidb)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    # NOTE: modified from pascal_voc.py, so imitates VOC-style 1-indexing
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        return self._load_pascal_annotation_xml(filename)
        
    def _load_pascal_annotation_xml(self, xmlpath):
        tree = ET.parse(xmlpath)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Assume these are 1-indexed
            x1 = float(bbox.find('xmin').text)-1
            y1 = float(bbox.find('ymin').text)-1
            x2 = float(bbox.find('xmax').text)-1
            y2 = float(bbox.find('ymax').text)-1
            name = obj.find('name').text.lower().strip()
            if name == 'object':
              cls = 0
            else:
              cls = self._class_to_ind[name]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = osp.join(output_dir, ('detections_' +
                                         self._image_set +
                                         self._year +
                                         '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

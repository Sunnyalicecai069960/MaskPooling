"""Refactor file directories, save/rename images and partition the
train/val/test set, in order to support the unified dataset interface.
"""

from __future__ import print_function

import sys
sys.path.insert(0, '.')

from zipfile import ZipFile
import os.path as osp
import numpy as np

from bpm.utils.utils import may_make_dir
from bpm.utils.utils import save_pickle
from bpm.utils.utils import load_pickle

from bpm.utils.dataset_utils import get_im_names
from bpm.utils.dataset_utils import partition_train_val_set
from bpm.utils.dataset_utils import new_im_name_tmpl
from bpm.utils.dataset_utils import parse_im_name as parse_new_im_name
from bpm.utils.dataset_utils import move_ims

dataset=['Market-1501-png','Market-1501-extend_trans_end_0','Market-1501-extend_trans_end_2','Market-1501-extend_trans_end_3'\
    ,'Market_3_extend_trans_end_0','Market_3_extend_trans_end_2','Market_3_extend_trans_end_3']

def parse_original_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = -1 if im_name.startswith('-1') else int(im_name[:4])
  else:
    parsed = int(im_name[4]) if im_name.startswith('-1') \
      else int(im_name[6])
  return parsed


def save_images(data_dir, save_dir=None, train_test_split_file=None):
  """Rename and move all used images to a directory."""

  # print("Extracting zip file")
  # root = osp.dirname(osp.abspath(zip_file))
  # if save_dir is None:
  #   save_dir = root
  # may_make_dir(osp.abspath(save_dir))
  # with ZipFile(zip_file) as z:
  #   z.extractall(path=save_dir)
  # print("Extracting zip file done")
  #get the images and origin name of path
  new_im_dir = osp.join(save_dir, 'images')
  may_make_dir(osp.abspath(new_im_dir))
  # define paths of all images and number of files in four folders
  im_paths = []
  bb_test = []
  bb_test_num = 0
  bb_train_num = 0
  bb_train = []
  query = []
  query_num = 0
  gt_bb_num = 0
  gt_bb = []
  nums = []

  for data in dataset:
      raw_dir = osp.join(data_dir,data)
      im_paths_ = get_im_names(osp.join(raw_dir, 'bounding_box_train'),
                               return_path=True, return_np=False)
      im_paths_.sort()
      bb_train += list(im_paths_)
      bb_train_num += len(im_paths_)

      im_paths_ = get_im_names(osp.join(raw_dir, 'bounding_box_test'),
                               return_path=True, return_np=False)
      im_paths_.sort()
      im_paths_ = [p for p in im_paths_ if not osp.basename(p).startswith('-1')]
      bb_test += list(im_paths_)
      bb_test_num += len(im_paths_)

      im_paths_ = get_im_names(osp.join(raw_dir, 'query'),
                               return_path=True, return_np=False)
      im_paths_.sort()
      query += list(im_paths_)
      query_num += len(im_paths_)
      q_ids_cams = set([(parse_original_im_name(osp.basename(p), 'id'),
                         parse_original_im_name(osp.basename(p), 'cam'))
                        for p in im_paths_])

      im_paths_ = get_im_names(osp.join(raw_dir, 'gt_bbox'),
                               return_path=True, return_np=False)
      im_paths_.sort()
      # Only gather images for those ids and cams used in testing.
      im_paths_ = [p for p in im_paths_
                   if (parse_original_im_name(osp.basename(p), 'id'),
                       parse_original_im_name(osp.basename(p), 'cam'))
                   in q_ids_cams]
      gt_bb += list(im_paths_)
      gt_bb_num += len(im_paths_)

  im_paths = bb_train+bb_test+query+gt_bb
  nums = [bb_train_num]+[bb_test_num]+[query_num]+[gt_bb_num]

  im_names = move_ims(
    im_paths, new_im_dir, parse_original_im_name, new_im_name_tmpl)

  split = dict()
  keys = ['trainval_im_names', 'gallery_im_names', 'q_im_names', 'mq_im_names']
  inds = [0] + nums
  inds = np.cumsum(np.array(inds))
  for i, k in enumerate(keys):
    split[k] = im_names[inds[i]:inds[i + 1]]

  save_pickle(split, train_test_split_file)
  print('Saving images done.')
  return split


def transform(data_dir, save_dir=None):
  """Refactor file directories, rename images and partition the train/val/test
  set.
  """

  train_test_split_file = osp.join(save_dir, 'train_test_split.pkl')
  train_test_split = save_images(data_dir, save_dir, train_test_split_file)
  # train_test_split = load_pickle(train_test_split_file)

  # partition train/val/test set

  trainval_ids = list(set([parse_new_im_name(n, 'id')
                           for n in train_test_split['trainval_im_names']]))
  # Sort ids, so that id-to-label mapping remains the same when running
  # the code on different machines.
  trainval_ids.sort()
  trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
  partitions = partition_train_val_set(
    train_test_split['trainval_im_names'], parse_new_im_name, num_val_ids=100)
  train_im_names = partitions['train_im_names']
  train_ids = list(set([parse_new_im_name(n, 'id')
                        for n in partitions['train_im_names']]))
  # Sort ids, so that id-to-label mapping remains the same when running
  # the code on different machines.
  train_ids.sort()
  train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

  # A mark is used to denote whether the image is from
  #   query (mark == 0), or
  #   gallery (mark == 1), or
  #   multi query (mark == 2) set

  val_marks = [0, ] * len(partitions['val_query_im_names']) \
              + [1, ] * len(partitions['val_gallery_im_names'])
  val_im_names = list(partitions['val_query_im_names']) \
                 + list(partitions['val_gallery_im_names'])

  test_im_names = list(train_test_split['q_im_names']) \
                  + list(train_test_split['mq_im_names']) \
                  + list(train_test_split['gallery_im_names'])
  test_marks = [0, ] * len(train_test_split['q_im_names']) \
               + [2, ] * len(train_test_split['mq_im_names']) \
               + [1, ] * len(train_test_split['gallery_im_names'])

  partitions = {'trainval_im_names': train_test_split['trainval_im_names'],
                'trainval_ids2labels': trainval_ids2labels,
                'train_im_names': train_im_names,
                'train_ids2labels': train_ids2labels,
                'val_im_names': val_im_names,
                'val_marks': val_marks,
                'test_im_names': test_im_names,
                'test_marks': test_marks}
  partition_file = osp.join(save_dir, 'partitions.pkl')
  save_pickle(partitions, partition_file)
  print('Partition file saved to {}'.format(partition_file))


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform Market1501 Dataset")
  parser.add_argument('--save_dir', type=str,
                      default='/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/combined/v1/')
  parser.add_argument('--data_dir', type=str,
                      default='/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/combined')
  args = parser.parse_args()
  data_dir = osp.abspath(osp.expanduser(args.data_dir))
  save_dir = osp.abspath(osp.expanduser(args.save_dir))
  transform(data_dir, save_dir)
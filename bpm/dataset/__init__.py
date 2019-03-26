import numpy as np
import os.path as osp
ospj = osp.join
ospeu = osp.expanduser

from ..utils.utils import load_pickle
from ..utils.dataset_utils import parse_im_name
from .TrainSet import TrainSet
from .TestSet import TestSet


def create_dataset(
    name='market1501',
    part='trainval',
    **kwargs):
  assert name in ['cuhk03_original_2','cuhk03_original_np','market30_retain','market30_retain_pixel3', \
    'market30_retain_rand_1', 'market30_retain_4_1', 'market30_retain_7_1', \
    'market30_retain_72_1', 'market30_retain_mask_batch_18_4', 'market30_retain_mask_batch_32_4', 'market30_retain_mask_batch_64_4', \
    'market30_retain_batch_18_4', 'market30_retain_batch_32_4', 'market30_retain_batch_64_4', \
    'market30_retain_rpp_4_1', \
    'market30_retain_pixel3_rand_1', 'market30_retain_pixel3_4_1', 'market30_retain_pixel3_7_1', \
    'market30_retain_pixel3_72_1', 'market30_retain_pixel3_mask_batch_18_4', 'market30_retain_pixel3_mask_batch_32_4', 'market30_retain_pixel3_mask_batch_64_4', \
    'market30_retain_pixel3_41_batch_18_4', 'market30_retain_pixel3_41_batch_32_4', 'market30_retain_pixel3_41_batch_64_4', \
    'market30_retain_pixel3_71_batch_18_4', 'market30_retain_pixel3_71_batch_32_4', 'market30_retain_pixel3_71_batch_64_4', \
    'market30_retain_pixel3_721_batch_18_4', 'market30_retain_pixel3_721_batch_32_4', 'market30_retain_pixel3_721_batch_64_4', \
    'market30_retain_pixel3_batch_18_4', 'market30_retain_pixel3_batch_32_4', 'market30_retain_pixel3_batch_64_4', \
    'cuhk03', 'cuhk03_33_np', 'cuhk03_33_np_retain','cuhk03_33_1','cuhk03_33_2','cuhk03_33_3','cuhk03_33_4','cuhk03_33_5','cuhk03_33_6','cuhk03_33_7','cuhk03_33_8','cuhk03_33_9','cuhk03_33_10', \
    'cuhk03_33_11','cuhk03_33_12','cuhk03_33_13','cuhk03_33_14','cuhk03_33_15','cuhk03_33_16','cuhk03_33_17','cuhk03_33_18','cuhk03_33_19','cuhk03_33_20', \
    'cuhk33_retain_rand_1', 'cuhk33_retain_4_1', 'cuhk33_retain_7_1', 'cuhk03_33_np_retain_d', 'cuhk03_33_np_retain_l', \
    'cuhk33_retain_batch_18_4', 'cuhk33_retain_batch_32_4', 'cuhk33_retain_batch_64_4', \
    'cuhk03_33_np_retain_4_1', 'cuhk03_33_np_retain_7_1', \
    'cuhk03_33_2_retain_4_1', 'cuhk03_33_2_retain_7_1', \
    'cuhk03_33_4_retain_4_1', 'cuhk03_33_9_retain_4_1', \
    'cuhk03_33_11_retain_4_1', 'cuhk03_33_11_retain_7_1', \
    'duke','duke30','duke33', 'duke_33_pixel5', \
    'duke_33_pixel5_4_1', 'duke_33_pixel5_7_1', 'duke_33_pixel5_batch_18_4', 'duke_33_pixel5_batch_32_4', 'duke_33_pixel5_batch_64_4', \
    'duke_33_pixel5_mask_batch_18_4', 'duke_33_pixel5_mask_batch_32_4', 'duke_33_pixel5_mask_batch_64_4', \
    'duke_33_pixel5_41_batch_18_4', 'duke_33_pixel5_41_batch_32_4', 'duke_33_pixel5_41_batch_64_4', \
    'duke_33_pixel5_71_batch_18_4', 'duke_33_pixel5_71_batch_32_4', 'duke_33_pixel5_71_batch_64_4', \
    'viper30','viper33', 'combined'], \
    "Unsupported Dataset {}".format(name)

  # assert im_type in ['detected', 'labeled'], \
    # "Unsupported Dataset Images Type {}".format(im_type)

  assert part in ['trainval', 'train', 'val', 'test'], \
    "Unsupported Dataset Part {}".format(part)

  ########################################
  # Specify Directory and Partition File #
  ########################################

 
  # market30_retain
  if ('market30_retain' in name) & ('pixel' not in name):
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
    if 'batch' in name:
      if 'mask' in name:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/batch_hard_1/new_shuffle_apn_partitions_mask_batch_'+ name[-4:] +'.pkl')
      else:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/batch_hard_1/new_shuffle_apn_partitions_batch_'+ name[-4:] +'.pkl')
    elif 'rand_1' in name:
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_rand_1.pkl')
    elif name[-3:]=='4_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_4_1.pkl')
    elif name[-3:]=='7_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_7_1.pkl')
    elif name[-4:]=='72_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_72_1.pkl')
    elif name =='market30_retain':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/partitions.pkl')  
  ###
# cuhk03_original_1, cuhk03_original_2, ... , cuhk03_original_19, cuhk03_original_20
# cuhk03_original_np
  elif 'cuhk03_original' in name:
    im_type = ['detected', 'labeled'][0]
    im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/cuhk03', im_type, 'images'))
    partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/cuhk03_33_retain', im_type, 'partitions_'+ name.split('_')[-1] +'.pkl'))
  # market30_retain_pixel3
  elif 'market30_retain_pixel3' in name:
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
    if 'batch' in name:
      if 'mask' in name:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_mask_batch_'+ name[-4:] +'.pkl')
      elif name.split('_')[3]=='41':
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_41_batch_'+ name[-4:] +'.pkl')
      elif name.split('_')[3]=='71':
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_71_batch_'+ name[-4:] +'.pkl')
      elif name.split('_')[3]=='721':
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_721_batch_'+ name[-4:] +'.pkl')
      else:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_batch_'+ name[-4:] +'.pkl')
    elif 'rand_1' in name:
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')
    elif name[-3:]=='4_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
    elif name[-3:]=='7_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_7_1.pkl')
    elif name[-4:]=='72_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_72_1.pkl')
    elif name == 'market30_retain_pixel3':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/partitions.pkl')

  # 767/700 CUHK03_NP
  # before cvpr2019
  # elif name == 'cuhk03_33_np_retain':
  #   im_type = ['detected', 'labeled'][0]
  #   im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
  #   partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'partitions.pkl'))
  ###
  elif name == 'cuhk03_33_np_retain_d':
    im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/cuhk03_3_retain/cuhk03_3_extend_trans_end_3/detected/images'))
    partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/cuhk03_33_retain/detected/partitions_np.pkl'))
  elif name == 'cuhk03_33_np_retain_l':
    im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/cuhk03_3_retain/cuhk03_3_extend_trans_end_3/labeled/images'))
    partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/cuhk03_33_retain/labeled/partitions_np.pkl'))
  ###
  # 1367/100:    cuhk03_33_1, cuhk_33_2, ... , cuhk03_33_19, cuhk03_33_20
  # 767/700:     cuhk03_33_np
  elif ('cuhk03_33' in name) & ('retain' not in name):
    im_type = ['detected', 'labeled'][0]
    im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
    partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/cuhk03_33_retain', im_type, 'partitions_'+ name.split('_')[-1] +'.pkl'))
  
  # cuhk03_33_retain_tri
  elif ('cuhk03_33' in name) & ('retain' in name):
    # 1367/100:    cuhk03_33_1_retain_4_1, cuhk03_33_2_retain_4_1, ... , cuhk03_33_19_retain_4_1,cuhk03_33_20_retain_4_1
    # 767/700:     cuhk03_33_np_retain_4_1
    im_type = ['detected', 'labeled'][0]
    im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
    if name[-3:]=='4_1':
      partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri', name[:-4], im_type, 'new_shuffle_apn_partitions_4_1.pkl'))
    if name[-3:]=='7_1':
      partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri', name[:-4], im_type, 'new_shuffle_apn_partitions_7_1.pkl'))
    if name[-4:]=='72_1':
      partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri', name[:-5], im_type, 'new_shuffle_apn_partitions_72_1.pkl'))


  elif name == 'duke':
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke/images')
    partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke/partitions.pkl')
  elif name == 'duke30':
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke30/images')
    partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke30/partitions.pkl')
  elif name == 'duke33':
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke33/images')
    partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke33/partitions.pkl')
  elif 'duke_33_pixel5' in name:
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke_33_pixel5/images')
    if 'batch' in name:
      if 'mask' in name:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/batch_hard_1/new_shuffle_apn_partitions_mask_batch_'+ name[-4:] +'.pkl')
      else:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/batch_hard_1/new_shuffle_apn_partitions_batch_'+ name[-4:] +'.pkl')
    elif 'rand_1' in name:
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/new_shuffle_apn_partitions_rand_1.pkl')
    elif name[-3:]=='4_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/new_shuffle_apn_partitions_4_1.pkl')
    elif name[-3:]=='7_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/new_shuffle_apn_partitions_7_1.pkl')
    elif name[-4:]=='72_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/new_shuffle_apn_partitions_72_1.pkl')
    else:
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke_33_pixel5/partitions.pkl')

  elif name == 'viper30':
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/viper30/images')
    partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/viper30/partitions.pkl')
  elif name == 'viper33':
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/viper33/images')
    partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/viper33/partitions.pkl')


  elif name == 'combined':
    assert part in ['trainval'], \
      "Only trainval part of the combined dataset is available now."
    im_dir = ospeu('/mnt/data/dataset/pcb/market1501_cuhk03_duke/trainval_images')
    partition_file = ospeu('/mnt/data/dataset/pcb/market1501_cuhk03_duke/partitions.pkl')

  ##################
  # Create Dataset #
  ##################

  # Use standard Market1501 CMC settings for all datasets here.
  cmc_kwargs = dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)

  partitions = load_pickle(partition_file)
  im_names = partitions['{}_im_names'.format(part)]

  if part == 'trainval':
    ids2labels = partitions['trainval_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)

  elif part == 'train':
    ids2labels = partitions['train_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)

  elif part == 'val':
    marks = partitions['val_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)

  elif part == 'test':
    marks = partitions['test_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)

  if part in ['trainval', 'train']:
    num_ids = len(ids2labels)
  elif part in ['val', 'test']:
    ids = [parse_im_name(n, 'id') for n in im_names]
    num_ids = len(list(set(ids)))
    num_query = np.sum(np.array(marks) == 0)
    num_gallery = np.sum(np.array(marks) == 1)
    num_multi_query = np.sum(np.array(marks) == 2)

  # Print dataset information
  print('-' * 40)
  print('{} {} set'.format(name, part))
  print('-' * 40)
  print('NO. Images: {}'.format(len(im_names)))
  print('NO. IDs: {}'.format(num_ids))

  try:
    print('NO. Query Images: {}'.format(num_query))
    print('NO. Gallery Images: {}'.format(num_gallery))
    print('NO. Multi-query Images: {}'.format(num_multi_query))
  except:
    pass

  print('-' * 40)

  return ret_set

def create_dataset_tri(
    name='market1501',
    part='trainval',
    flag='anchor',
    **kwargs):
  assert name in ['market30_retain_rand_1', 'market30_retain_4_1', 'market30_retain_7_1', 'market30_retain_8_1', 'market30_retain_9_1', \
                  'market30_retain_72_1', 'market30_retain_mask_batch_18_4', 'market30_retain_mask_batch_32_4', 'market30_retain_mask_batch_64_4', \
                  'market30_retain_batch_18_4', 'market30_retain_batch_32_4', 'market30_retain_batch_64_4', \
                  'market30_retain_rpp_4_1', \
                  'market30_retain_pixel3_rand_1', 'market30_retain_pixel3_4_1', 'market30_retain_pixel3_7_1', 'market30_retain_pixel3_8_1', 'market30_retain_pixel3_9_1', \
                  'market30_retain_pixel3_72_1', 'market30_retain_pixel3_mask_batch_18_4', 'market30_retain_pixel3_mask_batch_32_4', 'market30_retain_pixel3_mask_batch_64_4', \
                  'market30_retain_pixel3_41_batch_18_4', 'market30_retain_pixel3_41_batch_32_4', 'market30_retain_pixel3_41_batch_64_4', \
                  'market30_retain_pixel3_71_batch_18_4', 'market30_retain_pixel3_71_batch_32_4', 'market30_retain_pixel3_71_batch_64_4', \
                  'market30_retain_pixel3_721_batch_18_4', 'market30_retain_pixel3_721_batch_32_4', 'market30_retain_pixel3_721_batch_64_4', \
                  'market30_retain_pixel3_batch_18_4', 'market30_retain_pixel3_batch_32_4', 'market30_retain_pixel3_batch_64_4', \
                  'cuhk33_retain_rand_1', 'cuhk33_retain_4_1', 'cuhk33_retain_7_1', \
                  'cuhk03_33_np_retain_4_1', 'cuhk03_33_np_retain_7_1', \
                  'cuhk03_33_2_retain_4_1', 'cuhk03_33_2_retain_7_1', \
                  'cuhk03_33_4_retain_4_1', 'cuhk03_33_9_retain_4_1', \
                  'cuhk03_33_11_retain_4_1', 'cuhk03_33_11_retain_7_1', \
                  'duke', 'duke_33_pixel5', 'duke_33_pixel5_4_1', 'duke_33_pixel5_7_1', 'duke_33_pixel5_batch_18_4', 'duke_33_pixel5_batch_32_4', 'duke_33_pixel5_batch_64_4', \
                  'duke_33_pixel5_mask_batch_18_4', 'duke_33_pixel5_mask_batch_32_4', 'duke_33_pixel5_mask_batch_64_4', \
                  'duke_33_pixel5_41_batch_18_4', 'duke_33_pixel5_41_batch_32_4', 'duke_33_pixel5_41_batch_64_4', \
                  'duke_33_pixel5_71_batch_18_4', 'duke_33_pixel5_71_batch_32_4', 'duke_33_pixel5_71_batch_64_4', \
                  'combined'], \
    "Unsupported Dataset {}".format(name)

  assert part in ['trainval', 'train', 'val', 'test'], \
    "Unsupported Dataset Part {}".format(part)

  ########################################
  # Specify Directory and Partition File #
  ########################################
  # market30_retain
  if ('market30_retain' in name) & ('pixel' not in name):
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
    if 'batch' in name:
      if 'mask' in name:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/batch_hard_1/new_shuffle_apn_partitions_mask_batch_'+ name[-4:] +'.pkl')
      else:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/batch_hard_1/new_shuffle_apn_partitions_batch_'+ name[-4:] +'.pkl')
    elif 'rand_1' in name:
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_rand_1.pkl')
    elif name[-3:]=='4_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_4_1.pkl')
    elif name[-3:]=='7_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_7_1.pkl')
    elif name[-4:]=='72_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_72_1.pkl')
    elif name == 'market30_retain_rpp_4_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri_rpp/market30_retain/new_shuffle_apn_partitions_4_1.pkl')


  # market30_retain_pixel3
  elif 'market30_retain_pixel3' in name:
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
    if 'batch' in name:
      if 'mask' in name:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_mask_batch_'+ name[-4:] +'.pkl')
      elif name.split('_')[3]=='41':
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_41_batch_'+ name[-4:] +'.pkl')
      elif name.split('_')[3]=='71':
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_71_batch_'+ name[-4:] +'.pkl')
      elif name.split('_')[3]=='721':
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_721_batch_'+ name[-4:] +'.pkl')
      else:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/batch_hard_1/new_shuffle_apn_partitions_batch_'+ name[-4:] +'.pkl')
    elif 'rand_1' in name:
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')
    elif name[-3:]=='4_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
    elif name[-3:]=='7_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_7_1.pkl')
    elif name[-4:]=='72_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_72_1.pkl')



  # before cvpr2019
  # elif name == 'cuhk33_retain_4_1':
  #   im_type = ['detected', 'labeled'][1]
  #   im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
  #   partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_4_1.pkl'))
  
  # cuhk03_33_retain
  elif ('cuhk03_33' in name) & ('retain' in name):
    # 1367/100:    cuhk03_33_1_retain_4_1, cuhk03_33_2_retain_4_1, ... , cuhk03_33_19_retain_4_1,cuhk03_33_20_retain_4_1
    # 767/700:     cuhk03_33_np_retain_4_1
    im_type = ['detected', 'labeled'][0]
    im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
    if name[-3:]=='4_1':
      partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri', name[:-4], im_type, 'new_shuffle_apn_partitions_4_1.pkl'))
    if name[-3:]=='7_1':
      partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri', name[:-4], im_type, 'new_shuffle_apn_partitions_7_1.pkl'))
    if name[-4:]=='72_1':
      partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri', name[:-5], im_type, 'new_shuffle_apn_partitions_72_1.pkl'))


  # duke
  # duke_33_pixel5
  elif 'duke_33_pixel5' in name:
    im_dir = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/duke_33_pixel5/images')
    if 'batch' in name:
      if 'mask' in name:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/batch_hard_1/new_shuffle_apn_partitions_mask_batch_'+ name[-4:] +'.pkl')
      else:
        partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/batch_hard_1/new_shuffle_apn_partitions_batch_'+ name[-4:] +'.pkl')
    elif 'rand_1' in name:
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/new_shuffle_apn_partitions_rand_1.pkl')
    elif name[-3:]=='4_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/new_shuffle_apn_partitions_4_1.pkl')
    elif name[-3:]=='7_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/new_shuffle_apn_partitions_7_1.pkl')
    elif name[-4:]=='72_1':
      partition_file = ospeu('/GPUFS/nsccgz_ywang_1/wangying/DataSet/pcb/trans/tri/duke_33_pixel5/new_shuffle_apn_partitions_72_1.pkl')




  elif name == 'combined':
    assert part in ['trainval'], \
      "Only trainval part of the combined dataset is available now."
    im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
    partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')

  ##################
  # Create Dataset #
  ##################

  # Use standard Market1501 CMC settings for all datasets here.
  cmc_kwargs = dict(separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)

  partitions = load_pickle(partition_file)
  im_names = partitions['{}_{}_im_names'.format(part,flag)]

  if part == 'trainval':
    ids2labels = partitions['trainval_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)

  elif part == 'train':
    ids2labels = partitions['train_ids2labels']

    ret_set = TrainSet(
      im_dir=im_dir,
      im_names=im_names,
      ids2labels=ids2labels,
      **kwargs)

  elif part == 'val':
    marks = partitions['val_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)

  elif part == 'test':
    marks = partitions['test_marks']
    kwargs.update(cmc_kwargs)

    ret_set = TestSet(
      im_dir=im_dir,
      im_names=im_names,
      marks=marks,
      **kwargs)

  if part in ['trainval', 'train']:
    num_ids = len(ids2labels)
  elif part in ['val', 'test']:
    ids = [parse_im_name(n, 'id') for n in im_names]
    num_ids = len(list(set(ids)))
    num_query = np.sum(np.array(marks) == 0)
    num_gallery = np.sum(np.array(marks) == 1)
    num_multi_query = np.sum(np.array(marks) == 2)

  # Print dataset information
  print('-' * 40)
  print('{} {} set'.format(name, part))
  print('-' * 40)
  print('NO. Images: {}'.format(len(im_names)))
  print('NO. IDs: {}'.format(num_ids))

  try:
    print('NO. Query Images: {}'.format(num_query))
    print('NO. Gallery Images: {}'.format(num_gallery))
    print('NO. Multi-query Images: {}'.format(num_multi_query))
  except:
    pass

  print('-' * 40)

  return ret_set







# import numpy as np
# import os.path as osp
# ospj = osp.join
# ospeu = osp.expanduser

# from ..utils.utils import load_pickle
# from ..utils.dataset_utils import parse_im_name
# from .TrainSet import TrainSet
# from .TestSet import TestSet


# def create_dataset(
#     name='market1501',
#     part='trainval',
#     **kwargs):
#   assert name in ['market_png_4_1','market_png','market30_retain_pixel3_rand_1','market30_retain_pixel1_4_1','market30_retain_pixel2_4_1','market30_retain_pixel4_4_1','market30_retain_pixel5_4_1',\
#                   'market30_retain_pixel6_4_1','market30_retain_pixel7_4_1','market30_retain_pixel8_4_1','market30_retain_pixel9_4_1',\
#                   'market30_retain_pixel10_4_1','market30_retain_pixel1','market30_retain_pixel2','market30_retain_pixel4','market30_retain_pixel5','market30_retain_pixel6',\
#                   'market30_retain_pixel7','market30_retain_pixel8','market30_retain_pixel9','market30_retain_pixel10',\
#                   'market30_retain_rand_1','market30_retain_pixel3_3_1','market30_retain_pixel3_4_1',\
#                   'market30_retain_pixel3_5_3','market30_retain_pixel3_rand_1','market30_retain_pixel3',\
#                   'cuhk33_retain_3_1','cuhk33_retain_4','cuhk33_retain_4_1','cuhk33_retain_5','cuhk33_retain_5_3','cuhk33_retain_5_6',\
#                   'market30_retain_3_1','market30_retain_4','market30_retain_4_1','market30_retain_5',\
#                   'market30_retain_5_3','market30_retain_5_6','market33_retain_5','market33_retain_5_3',\
#                   'market33_retain_5_6','market33_retain_3','market33_retain_3_1','market33_retain_4','market33_retain_4_1',\
#                   'market30_retain_pixel0_4_1','market30_retain_pixel0_5_6','market30_retain_pixel0_5_3',\
#                   'market30_retain_pixel0_5','market30_retain_pixel0_4_5','market30_retain_pixel0_3_1',\
#                   'cuhk33_retain_3','mars30_retain_pixel7','mars32_retain_pixel7','mars33_retain_pixel7',\
#                   'market30_retain_pixel0','market30_retain_2','market30_retain_3','market30_retain_pixel0_2',\
#                   'market30_retain_pixel0_3','mars_oldmask_retain','mars','mars20','mars22','mars23','mars30',\
#                   'mars32','mars33','market','cuhk20','cuhk22','cuhk23','cuhk20_retain','cuhk22_retain',\
#                   'cuhk23_retain','cuhk30','cuhk32','cuhk33','cuhk30_retain','cuhk32_retain','cuhk33_retain',\
#                   'cuhk40','cuhk42','cuhk43','cuhk40_retain','cuhk42_retain','cuhk43_retain','market1501',\
#                   'market_combined','market23','market22', 'market20','market20_retain','market22_retain',\
#                   'market23_retain', 'market30','market32','market33','market30_retain','market32_retain',\
#                   'market33_retain','market40','market42','market43','market40_retain','market42_retain',\
#                   'market43_retain','market_oldmask','market_oldmask_retain','market_trans','market_png',\
#                   'market1501', 'cuhk03', 'duke', 'combined'], \
#     "Unsupported Dataset {}".format(name)

#   assert part in ['trainval', 'train', 'val', 'test'], \
#     "Unsupported Dataset Part {}".format(part)

#   ########################################
#   # Specify Directory and Partition File #
#   ########################################

#   if name == 'market1501':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/market1501/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/market1501/partitions.pkl')
#   elif name == 'market_png':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_origin/market-1501-png/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_origin/market-1501-png/partitions.pkl')
#   elif name == 'market_png_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_origin/market-1501-png/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')

#   elif name == 'market30_retain_pixel3_3_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_3_1.pkl')
#   elif name == 'market30_retain_pixel3_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel3_5_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_5_3.pkl')
#   elif name == 'market30_retain_pixel3_rand_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')
  
#   elif name == 'market33_retain_5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_5_5.pkl')
#   elif name == 'market33_retain_5_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_5_3.pkl')
#   elif name == 'market33_retain_5_6':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_5_6.pkl')
#   elif name == 'market33_retain':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/partitions.pkl')
#   elif name == 'market33_retain_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_3.pkl')
#   elif name == 'market33_retain_3_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_3_1.pkl')
#   elif name == 'market33_retain_4':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_4_5.pkl')
#   elif name == 'market33_retain_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_4_1.pkl')
  
#   elif name == 'market30_retain_pixel0_5_6':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_5_6.pkl')
#   elif name == 'market30_retain_pixel0_5_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_5_3.pkl')
#   elif name == 'market30_retain_pixel0_5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_5_5.pkl')
#   elif name == 'market30_retain_pixel0_4_5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_4_5.pkl')
#   elif name == 'market30_retain_pixel0_3_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_3_1.pkl')
#   elif name == 'market30_retain_pixel0_2':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_2_2.pkl')
#   elif name == 'market30_retain_pixel0_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_3.pkl')

#   elif name == 'market30_retain_pixel0':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/partitions.pkl')
#   elif name == 'market30_retain_pixel0_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_1/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_1/partitions.pkl')
#   elif name == 'market30_retain_pixel1_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_1/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel2':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_2/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_2/partitions.pkl')
#   elif name == 'market30_retain_pixel2_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_2/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/partitions.pkl')
#   elif name == 'market30_retain_pixel3_rand_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')
#   elif name == 'market30_retain_pixel4':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_4/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_4/partitions.pkl')
#   elif name == 'market30_retain_pixel4_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_4/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_5/partitions.pkl')
#   elif name == 'market30_retain_pixel5_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel6':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_6/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_6/partitions.pkl')
#   elif name == 'market30_retain_pixel6_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_6/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel7':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_7/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_7/partitions.pkl')
#   elif name == 'market30_retain_pixel7_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_7/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel8':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_8/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_8/partitions.pkl')
#   elif name == 'market30_retain_pixel8_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_8/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel9':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_9/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_9/partitions.pkl')
#   elif name == 'market30_retain_pixel9_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_9/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel10':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_10/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_10/partitions.pkl')
#   elif name == 'market30_retain_pixel10_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_10/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
  
  
#   elif name == 'market30_retain_rand_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')
#   elif name == 'market30_retain_3_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_3_1.pkl')
#   elif name == 'market30_retain_4':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_4_5.pkl')
#   elif name == 'market30_retain_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_5_5.pkl')
#   elif name == 'market30_retain_5_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_5_3.pkl')
#   elif name == 'market30_retain_5_6':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_5_6.pkl')
#   elif name == 'market30_retain_2':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_2_2.pkl')
#   elif name == 'market30_retain_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_3.pkl')
#   elif name == 'mars_oldmask_retain':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_oldmask_retain/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_oldmask_retain/partitions.pkl')
#   elif name == 'market30_retain':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/partitions.pkl')
#   elif name == 'market32_retain':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_2/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_2/partitions.pkl')
#   elif name == 'market33_retain':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/partitions.pkl')

#   elif name == 'mars20':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_2/mars_2_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_2/mars_2_extend_trans_end_0/partitions.pkl')
#   elif name == 'mars22':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_2/mars_2_extend_trans_end_2/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_2/mars_2_extend_trans_end_2/partitions.pkl')
#   elif name == 'mars23':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_2/mars_2_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_2/mars_2_extend_trans_end_3/partitions.pkl')
#   elif name == 'mars30':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3/mars_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3/mars_3_extend_trans_end_0/partitions.pkl')
#   elif name == 'mars32':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3/mars_3_extend_trans_end_2/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3/mars_3_extend_trans_end_2/partitions.pkl')
#   elif name == 'mars33':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3/mars_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3/mars_3_extend_trans_end_3/partitions.pkl')
#   elif name == 'mars30_retain_pixel7':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3_retain_7/mars_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3_retain_7/mars_3_extend_trans_end_0/partitions.pkl')
#   elif name == 'mars32_retain_pixel7':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3_retain_7/mars_3_extend_trans_end_2/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3_retain_7/mars_3_extend_trans_end_2/partitions.pkl')
#   elif name == 'mars33_retain_pixel7':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3_retain_7/mars_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars_3_retain_7/mars_3_extend_trans_end_3/partitions.pkl')
#   elif name == 'mars':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars/images_RGBA')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/mars/partitions.pkl')
#   elif name == 'cuhk33_retain':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'partitions.pkl'))
#   elif name == 'cuhk33_retain_3':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_3.pkl'))
#   elif name == 'cuhk33_retain_3_1':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_3_1.pkl'))
#   elif name == 'cuhk33_retain_4':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_4_5.pkl'))
#   elif name == 'cuhk33_retain_4_1':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_4_1.pkl'))
#   elif name == 'cuhk33_retain_5':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_5_5.pkl'))
#   elif name == 'cuhk33_retain_5_3':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_5_3.pkl'))
#   elif name == 'cuhk33_retain_5_6':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_5_6.pkl'))
  
#   elif name == 'duke':
#     im_dir = ospeu('~/Dataset/duke/images')
#     partition_file = ospeu('~/Dataset/duke/partitions.pkl')

#   elif name == 'combined':
#     assert part in ['trainval'], \
#       "Only trainval part of the combined dataset is available now."
#     im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
#     partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')

#   ##################
#   # Create Dataset #
#   ##################

#   # Use standard Market1501 CMC settings for all datasets here.
#   cmc_kwargs = dict(separate_camera_set=False,
#                     single_gallery_shot=False,
#                     first_match_break=True)

#   partitions = load_pickle(partition_file)
#   im_names = partitions['{}_im_names'.format(part)]

#   if part == 'trainval':
#     ids2labels = partitions['trainval_ids2labels']

#     ret_set = TrainSet(
#       im_dir=im_dir,
#       im_names=im_names,
#       ids2labels=ids2labels,
#       **kwargs)

#   elif part == 'train':
#     ids2labels = partitions['train_ids2labels']

#     ret_set = TrainSet(
#       im_dir=im_dir,
#       im_names=im_names,
#       ids2labels=ids2labels,
#       **kwargs)

#   elif part == 'val':
#     marks = partitions['val_marks']
#     kwargs.update(cmc_kwargs)

#     ret_set = TestSet(
#       im_dir=im_dir,
#       im_names=im_names,
#       marks=marks,
#       **kwargs)

#   elif part == 'test':
#     marks = partitions['test_marks']
#     kwargs.update(cmc_kwargs)

#     ret_set = TestSet(
#       im_dir=im_dir,
#       im_names=im_names,
#       marks=marks,
#       **kwargs)

#   if part in ['trainval', 'train']:
#     num_ids = len(ids2labels)
#   elif part in ['val', 'test']:
#     ids = [parse_im_name(n, 'id') for n in im_names]
#     num_ids = len(list(set(ids)))
#     num_query = np.sum(np.array(marks) == 0)
#     num_gallery = np.sum(np.array(marks) == 1)
#     num_multi_query = np.sum(np.array(marks) == 2)

#   # Print dataset information
#   print('-' * 40)
#   print('{} {} set'.format(name, part))
#   print('-' * 40)
#   print('NO. Images: {}'.format(len(im_names)))
#   print('NO. IDs: {}'.format(num_ids))

#   try:
#     print('NO. Query Images: {}'.format(num_query))
#     print('NO. Gallery Images: {}'.format(num_gallery))
#     print('NO. Multi-query Images: {}'.format(num_multi_query))
#   except:
#     pass

#   print('-' * 40)

#   return ret_set


# def create_dataset_tri(
#     name='market1501',
#     part='trainval',
#     flag='anchor',
#     **kwargs):
#   assert name in ['market_png_4_1','market_png','market30_retain_pixel3_rand_1','market30_retain_pixel1_4_1','market30_retain_pixel2_4_1','market30_retain_pixel4_4_1','market30_retain_pixel5_4_1',\
#                   'market30_retain_pixel6_4_1','market30_retain_pixel7_4_1','market30_retain_pixel8_4_1','market30_retain_pixel9_4_1',\
#                   'market30_retain_pixel10_4_1','market30_retain_rand_1','market30_retain_pixel3_3_1','market30_retain_pixel3_4_1','market30_retain_pixel3_5_3','market30_retain_pixel3_rand_1',\
#                   'cuhk33_retain_3_1','cuhk33_retain_4','cuhk33_retain_4_1','cuhk33_retain_5','cuhk33_retain_5_3','cuhk33_retain_5_6',\
#                   'market30_retain_3_1','market30_retain_4','market30_retain_4_1','market30_retain_5',\
#                   'market30_retain_5_3','market30_retain_5_6','market33_retain_5','market33_retain_5_3',\
#                   'market33_retain_5_6','market33_retain_3','market33_retain_3_1','market33_retain_4',\
#                   'market33_retain_4_1','market30_retain_pixel0_4_1','market30_retain_pixel0_5_6',\
#                   'market30_retain_pixel0_5_3','market30_retain_pixel0_5','market30_retain_pixel0_4_5',\
#                   'cuhk33_retain_3','market30_retain_pixel0_3_1','market30_retain_2','market30_retain_3',\
#                   'market30_retain_pixel0_2','market30_retain_pixel0_3','mars_oldmask_retain','mars',\
#                   'mars20','mars22','mars23','mars30','mars32','mars33','market','cuhk20','cuhk22',\
#                   'cuhk23','cuhk20_retain','cuhk22_retain','cuhk23_retain','cuhk30','cuhk32','cuhk33',\
#                   'cuhk30_retain','cuhk32_retain','cuhk33_retain','cuhk40','cuhk42','cuhk43',\
#                   'cuhk40_retain','cuhk42_retain','cuhk43_retain','market1501','market_combined',\
#                   'market23','market22', 'market20','market20_retain','market22_retain','market23_retain', \
#                   'market30','market32','market33','market30_retain','market32_retain','market33_retain',
#                   'market40','market42','market43','market40_retain','market42_retain','market43_retain',
#                   'market_oldmask','market_oldmask_retain','market_trans','market_png','market1501', 
#                   'cuhk03', 'duke', 'combined'], \
#     "Unsupported Dataset {}".format(name)

#   assert part in ['trainval', 'train', 'val', 'test'], \
#     "Unsupported Dataset Part {}".format(part)

#   ########################################
#   # Specify Directory and Partition File #
#   ########################################

#   if name == 'market1501':
#     im_dir = ospeu('~/Dataset/market1501/images')
#     partition_file = ospeu('~/Dataset/market1501/partitions.pkl')
#   elif name == 'market_png':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_origin/market-1501-png/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_origin/market-1501-png/partitions.pkl')
#   elif name == 'market_png_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_origin/market-1501-png/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')

#   elif name == 'market30_retain_pixel3_3_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_3_1.pkl')
#   elif name == 'market30_retain_pixel3_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel3_5_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_5_3.pkl')
#   elif name == 'market30_retain_pixel3_rand_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')
#   elif name == 'market30_retain_pixel3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/partitions.pkl')

#   elif name == 'market33_retain_5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_5_5.pkl')
#   elif name == 'market33_retain_5_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_5_3.pkl')
#   elif name == 'market33_retain_5_6':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_5_6.pkl')
#   elif name == 'market33_retain_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_3.pkl')
#   elif name == 'market33_retain_3_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_3_1.pkl')
#   elif name == 'market33_retain_4':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_4_5.pkl')
#   elif name == 'market33_retain_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_3/new_shuffle_apn_partitions_4_1.pkl')

#   elif name == 'market30_retain_pixel0_5_6':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_5_6.pkl')
#   elif name == 'market30_retain_pixel0_5_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_5_3.pkl') 
#   elif name == 'market30_retain_pixel0_5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_5_5.pkl')
#   elif name == 'market30_retain_pixel0_4_5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_4_5.pkl')
#   elif name == 'market30_retain_pixel0_3_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_3_1.pkl')
#   elif name == 'market30_retain_pixel0_2':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_2_2.pkl')
#   elif name == 'market30_retain_pixel0_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/new_shuffle_apn_partitions_3.pkl')
  
#   elif name == 'market30_retain_rand_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')
#   elif name == 'market30_retain_3_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_3_1.pkl')
#   elif name == 'market30_retain_4':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_4_5.pkl')
#   elif name == 'market30_retain_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_5_5.pkl')
#   elif name == 'market30_retain_5_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_5_3.pkl')
#   elif name == 'market30_retain_5_6':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_5_6.pkl')
#   elif name == 'market30_retain_2':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_2_2.pkl')
#   elif name == 'market30_retain_3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/tri/Market_3_retain/Market_3_extend_trans_end_0/new_shuffle_apn_partitions_3.pkl')
  
#   elif name == 'market30_retain_pixel0':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/partitions.pkl')
#   elif name == 'market30_retain_pixel0_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_1/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_1/partitions.pkl')
#   elif name == 'market30_retain_pixel1_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_1/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel2':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_2/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_2/partitions.pkl')
#   elif name == 'market30_retain_pixel2_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_2/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel3':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/partitions.pkl')
#   elif name == 'market30_retain_pixel3_rand_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_rand_1.pkl')
#   elif name == 'market30_retain_pixel4':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_4/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_4/partitions.pkl')
#   elif name == 'market30_retain_pixel4_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_4/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel5':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_5/partitions.pkl')
#   elif name == 'market30_retain_pixel5_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_retain/Market_3_extend_trans_end_0/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel6':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_6/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_6/partitions.pkl')
#   elif name == 'market30_retain_pixel6_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_6/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel7':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_7/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_7/partitions.pkl')
#   elif name == 'market30_retain_pixel7_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_7/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel8':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_8/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_8/partitions.pkl')
#   elif name == 'market30_retain_pixel8_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_8/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel9':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_9/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_9/partitions.pkl')
#   elif name == 'market30_retain_pixel9_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_9/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
#   elif name == 'market30_retain_pixel10':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_10/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_10/partitions.pkl')
#   elif name == 'market30_retain_pixel10_4_1':
#     im_dir = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_10/images')
#     partition_file = ospeu('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/Market_3_pixel/Market_pixel_end_3/new_shuffle_apn_partitions_4_1.pkl')
  
#   elif name == 'cuhk03':
#     im_type = ['detected', 'labeled'][0]
#     im_dir = ospeu(ospj('~/Dataset/cuhk03', im_type, 'images'))
#     partition_file = ospeu(ospj('~/Dataset/cuhk03', im_type, 'partitions.pkl'))
#   elif name == 'cuhk33_retain_3':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'new_shuffle_apn_partitions_3.pkl'))
#   elif name == 'cuhk33_retain_3_1':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_3_1.pkl'))
#   elif name == 'cuhk33_retain_4':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_4_5.pkl'))
#   elif name == 'cuhk33_retain_4_1':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_4_1.pkl'))
#   elif name == 'cuhk33_retain_5':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_5_5.pkl'))
#   elif name == 'cuhk33_retain_5_3':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_5_3.pkl'))
#   elif name == 'cuhk33_retain_5_6':
#     im_type = ['detected', 'labeled'][1]
#     im_dir = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, 'images'))
#     partition_file = ospeu(ospj('/GPUFS/nsccgz_ywang_1/alice/dataset/pcb/trans/cuhk03_3_retain/cuhk03_3_extend_trans_end_3', im_type, im_type+'_new_shuffle_apn_partitions_5_6.pkl'))
  
#   elif name == 'duke':
#     im_dir = ospeu('~/Dataset/duke/images')
#     partition_file = ospeu('~/Dataset/duke/partitions.pkl')

#   elif name == 'combined':
#     assert part in ['trainval'], \
#       "Only trainval part of the combined dataset is available now."
#     im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
#     partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')

#   ##################
#   # Create Dataset #
#   ##################

#   # Use standard Market1501 CMC settings for all datasets here.
#   cmc_kwargs = dict(separate_camera_set=False,
#                     single_gallery_shot=False,
#                     first_match_break=True)

#   partitions = load_pickle(partition_file)
#   im_names = partitions['{}_{}_im_names'.format(part,flag)]

#   if part == 'trainval':
#     ids2labels = partitions['trainval_ids2labels']

#     ret_set = TrainSet(
#       im_dir=im_dir,
#       im_names=im_names,
#       ids2labels=ids2labels,
#       **kwargs)

#   elif part == 'train':
#     ids2labels = partitions['train_ids2labels']

#     ret_set = TrainSet(
#       im_dir=im_dir,
#       im_names=im_names,
#       ids2labels=ids2labels,
#       **kwargs)

#   elif part == 'val':
#     marks = partitions['val_marks']
#     kwargs.update(cmc_kwargs)

#     ret_set = TestSet(
#       im_dir=im_dir,
#       im_names=im_names,
#       marks=marks,
#       **kwargs)

#   elif part == 'test':
#     marks = partitions['test_marks']
#     kwargs.update(cmc_kwargs)

#     ret_set = TestSet(
#       im_dir=im_dir,
#       im_names=im_names,
#       marks=marks,
#       **kwargs)

#   if part in ['trainval', 'train']:
#     num_ids = len(ids2labels)
#   elif part in ['val', 'test']:
#     ids = [parse_im_name(n, 'id') for n in im_names]
#     num_ids = len(list(set(ids)))
#     num_query = np.sum(np.array(marks) == 0)
#     num_gallery = np.sum(np.array(marks) == 1)
#     num_multi_query = np.sum(np.array(marks) == 2)

#   # Print dataset information
#   print('-' * 40)
#   print('{} {} set'.format(name, part))
#   print('-' * 40)
#   print('NO. Images: {}'.format(len(im_names)))
#   print('NO. IDs: {}'.format(num_ids))

#   try:
#     print('NO. Query Images: {}'.format(num_query))
#     print('NO. Gallery Images: {}'.format(num_gallery))
#     print('NO. Multi-query Images: {}'.format(num_multi_query))
#   except:
#     pass

#   print('-' * 40)

#   return ret_set






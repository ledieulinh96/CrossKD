# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),

    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['dataset_type'],

        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='annotations/instances_train2014.json',
                data_prefix=dict(img='train2014/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                #ann_file='annotations/instances_train2017.json',
                ann_file='annotations/instances_valminusminival2014.json',
                #data_prefix=dict(img='train2017/'),
                data_prefix=dict(img='val2014/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline),
        ],
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file='annotations/instances_val2017.json',
        ann_file='annotations/instances_minival2014.json',
        #data_prefix=dict(img='val2017/'),
        data_prefix=dict(img='val2014/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    #ann_file=data_root + 'annotations/instances_val2017.json',
    ann_file=data_root + 'annotations/instances_minival2014.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')

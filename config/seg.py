config = dict(

    model_config = dict(
        name = 'seg_netz',
        in_channel = 3,
        out_channel = 6,
    ),

    train_cfig = dict(
        batch_size = 16,
        num_epoch = 100,
        checkpoint_path =r'/home/Wuwei/WuweiStuH/PycharmProjects/LevirSeg/checkpoint/seg',
        lr = 0.001,
        device = 'cuda:0',
        log_path = r'/home/Wuwei/WuweiStuH/PycharmProjects/LevirSeg/checkpoint/seg',
        criterion = 'CrossEntropyLoss',
        optimer = 'SGD',
        num_workers = 0,
        train_data = dict(
            root_path = r'/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen',
            train_data_files = r'/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen/train.txt',
            val_data_files =r'/data02/zht_vqa/change_detection/LearnGroup/dataset/Vaihingen/valid.txt'
        ),

    interface_cifg = dict()
    )
)
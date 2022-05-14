config = dict(

    model_config = dict(
        name = 'seg_net',
        in_channel = 3,
        out_channel = 6,
    ),

    train_cfig = dict(
        batch_size = 16,
        checkpoint_path =r'checkpoint/seg',
        train_data = dict(
            root_path = r'',
            data_files = r'',
        ),

    interface_cifg = dict()
    )
)
class DefaultConfigs(object):
    #1.string parameters
    # train_data = 'D:\\dataset\\Graduation_project\\GID\\train_512\\zonl_class\\'
    # test_data = 'D:\\dataset\\Graduation_project\\GID\\test_512\\zonl_class\\'
    # val_data = 'D:\\dataset\\Graduation_project\\GID\\val_512\\zonl_class\\'
    train_data = 'D:\\dataset\\action\\train\\'
    test_data = "D:\\dataset\\action\\test\\"
    val_data = 'D:\\dataset\\action\\validation\\'
    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = weights + "best_model_graduation/resnet50/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "2"
    augmen_level = "medium"  # "light","hard","hard2"

    #2.numeric parameters
    epochs = 50
    batch_size = 32
    img_height = 512
    img_weight = 512
    num_classes = 10
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()

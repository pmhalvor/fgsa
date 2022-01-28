from dataset import NorecOneHot 

# TODO Build config file for universal paths
DATA_DIR = "/fp/homes01/u01/ec-pmhalvor/data/norec_fine/"  # TODO hide personal info

def test_dataset():
    train_dataset = NorecOneHot(data_path=DATA_DIR + "train/", proportion=0.01)
    test_dataset = NorecOneHot(data_path=DATA_DIR + "test/", proportion=0.05)
    dev_dataset = NorecOneHot(data_path=DATA_DIR + "dev/", proportion=0.05)

    for i in range(3):
        print('------- Index {} -------'.format(i))
        print(train_dataset[i])
        print(train_dataset[i].shape)
        print('------------------------\n')
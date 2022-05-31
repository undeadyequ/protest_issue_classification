from data_prepare import prepar_data, separte_train_test
from data_preprocess import clean_fine_grain

def main():
    # clean data
    dir_f = '/home/Data/image_data/Presidential'
    new_dir = "/home/Data/image_data/Presidential_clean"
    #clean_fine_grain(dir_f, new_dir)

    # Prepare transcription and label of each image
    fine_grain_dir = "/home/Data/image_data/Presidential_clean"
    label2num = {'Biden': 0,
                 'Clinton': 1,
                 'Obama': 2,
                 'Romney': 3,
                 'Trump': 4}
    prepar_data(fine_grain_dir, label2num)

    # separate
    id_lab_trans = "/home/Data/image_data/id_lab_trans.csv"
    id_path = "/home/Data/image_data/id_path.csv"
    img_dir_traintest = "/home/Data/image_data/Presidential_clean_traintest"
    separte_train_test(id_lab_trans, id_path, img_dir_traintest)
    # start train/evaluation -> in protest-detection-violence-estimation project


if __name__ == '__main__':
    main()
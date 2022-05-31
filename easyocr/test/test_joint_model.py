import pytest
from text_emb.text_embedding import Fasttext, BOW, Tfidf
from util import ProtestDataset_txtfts_2, Lighting
import os
import torchvision.transforms as transforms
from easyocr.joint_model import Sentence_model
from torch.utils.data import Dataset, DataLoader
import torch
from train.collate_fn import CommonCollateFn


def test_sentence_model():
    data_dir = "/home/Data/image_data/Presidential_clean_traintest"
    id_lab_trans_train = os.path.join(data_dir, "id_lab_trans_train.csv")
    id_lab_trans_eval = os.path.join(data_dir, "id_lab_trans_test.csv")
    id_lab_trans = os.path.join(data_dir, "id_lab_trans.csv")
    id_path_train = os.path.join(data_dir, "id_path_train.csv")
    id_path_eval = os.path.join(data_dir, "id_path_test.csv")

    # dataloader

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948,  0.4203]])

    train_dataset = ProtestDataset_txtfts_2(
        id_label_trans_train_f=id_lab_trans_train,
        id_label_trans_f=id_lab_trans,
        id_path_f=id_path_train,
        embedding="fasttext",
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness = 0.4,
                contrast = 0.4,
                saturation = 0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, eigval, eigvec),
            normalize,
        ]),
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=8,
        shuffle=True,
        collate_fn=CommonCollateFn()
    )

    # model
    model = Sentence_model()
    model = model.cuda()
    model.train()
    for data in train_loader:
        _, data_dict = data
        text_enc = data_dict["text_fts"].cuda()
        assert text_enc.size()[2] == 300
        out = model(text_enc)
        assert out.size()[1] == 5
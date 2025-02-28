import os
import h5py
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--predict-result-path', type=str, help='predict labels dir path')
args = parser.parse_args()

class PredResult:
    def __init__(self, result):
        self.result = result
        info = result.split(",")
        self.name = info[0]
        self.id = self.name.split(".")[0]
        self.prediction = info[1]
        self.confidence = info[2]
    
    def __repr__(self):
        return self.result
    
    def __str__(self):
        return self.result

def get_result(s):
    result = PredResult(s.strip())
    return result.name, result


def get_img_name(f, name_col, idx=0):
    img_name = ''.join(map(chr, f[name_col[idx][0]][()].flatten()))
    return (img_name)


def get_img_boxes(f, bbox_col, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    bbox_prop = ['height', 'left', 'top', 'width', 'label']
    meta = {key: [] for key in bbox_prop}

    box = f[bbox_col[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            meta[key].append(int(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                meta[key].append(int(f[box[key][i][0]][()].item()))

    return meta

def get_true_label(bbox):
    ret = ""
    label_cnt = len(bbox['label'])
    for i in range(label_cnt):
        label = bbox['label'][i]
        if label == 10:
            label = 0
        ret = str(label) + ret

    return ret

def get_predict_label(img_name):
    full_path = os.path.join(predict_result_path, img_name.replace("png", "txt"))
    with open(full_path, "r") as f:
        lines = [line for line in f.readlines() if len(line)>0]
        
        label = ''.join(list(map(lambda x: x.split(" ")[0], lines)))
    return label

def compare_result(true_label, predict_label):
    if len(true_label) != len(predict_label):
        return False
    if len(true_label) == 1:
        return true_label == predict_label

    true_label = sorted(true_label)
    predict_label = sorted(predict_label)
    for true_item, false_item in zip(true_label, predict_label):
        if true_item != false_item:
            return False
    return True


if __name__ == "__main__":
    # predict_result_path = "/data/zcd/hj/houseNumberRecoginzation/final/yolov5/runs/detect/test-result/labels"
    # predict_result_path = "/dasta/zcd/hj/houseNumberRecoginzation/yolov5/runs/detect/test-result-yolov5s/labels"
    predict_result_path = args.predict_result_path

    test_result_path = "/data/zcd/hj/houseNumberRecoginzation/final/data/test/digitStruct.mat"
    mat = h5py.File(test_result_path)

    data_size = mat['/digitStruct/name'].shape[0]
    print(f'Data size: {data_size}')

    name_col = mat['/digitStruct/name']
    bbox_col = mat['/digitStruct/bbox']

    acc_num = 0
    error_list = []
    for idx in tqdm(range(data_size), desc="Test Result"):
        img_name = get_img_name(mat, name_col, idx)
        bbox = get_img_boxes(mat, bbox_col, idx)
        try:
            true_label = get_true_label(bbox)
            predict_label = get_predict_label(img_name)
            if compare_result(true_label, predict_label):
                acc_num += 1
        except:
            error_list.append(img_name)
    accuracy = acc_num / data_size
    print(f"Test Accuracy: {accuracy}")
    print(f"Test Data Num: {data_size}")
    print(f"Test Accuracy Num: {acc_num}")

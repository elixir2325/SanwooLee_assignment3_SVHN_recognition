首先需要在 data/train和data/test下存放数据（需要图片和digitStruct.mat）；这里因为空间限制把这两个文件里的数据置空了。
然后跑utils/read_data.py，以获得预处理之后的数据，分别用如下两个代码跑两次就行
1）
if __name__ == '__main__':
    converter = DataConverter('data/test/digit_struct.pickle')
    converter.resize_image_and_bboxes(write_path='data/test_resized/', size=64)
    with open('data/test_dataconverter.pickle', 'wb') as handle:
        pickle.dump(converter, handle, protocol=pickle.HIGHEST_PROTOCOL)
2）
if __name__ == '__main__':
    converter = DataConverter('data/train/digit_struct.pickle')
    converter.resize_image_and_bboxes(write_path='data/train_resized/', size=64)
    with open('data/train_dataconverter.pickle', 'wb') as handle:
        pickle.dump(converter, handle, protocol=pickle.HIGHEST_PROTOCOL)


最后跑main.py，进行模型训练和测试
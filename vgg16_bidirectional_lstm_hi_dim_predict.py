import numpy as np
import sys
import os
from config import TESTING_DATA_PATH, MODELS_PATH


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from recurrent_networks import vgg16BidirectionalLSTMVideoClassifier
    from UCF101_loader import load_ucf, scan_ucf_with_labels

    print(f"Bidirectional LSTM Prediction Configuration:")
    print(f"  Test data path: {TESTING_DATA_PATH}")
    print(f"  Models path: {MODELS_PATH}")

    vgg16_include_top = False
    config_file_path = vgg16BidirectionalLSTMVideoClassifier.get_config_file_path(MODELS_PATH,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = vgg16BidirectionalLSTMVideoClassifier.get_weight_file_path(MODELS_PATH,
                                                                                  vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    load_ucf(os.path.dirname(TESTING_DATA_PATH))

    predictor = vgg16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)

    videos = scan_ucf_with_labels(TESTING_DATA_PATH, [label for (label, label_index) in predictor.labels.items()])

    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    np.random.shuffle(video_file_path_list)

    correct_count = 0
    count = 0

    for video_file_path in video_file_path_list:
        label = videos[video_file_path]
        predicted_label = predictor.predict(video_file_path)
        print('predicted: ' + predicted_label + ' actual: ' + label)
        correct_count = correct_count + 1 if label == predicted_label else correct_count
        count += 1
        accuracy = correct_count / count
        print('accuracy: ', accuracy)


if __name__ == '__main__':
    main()

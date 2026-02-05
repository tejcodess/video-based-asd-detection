import numpy as np
from keras import backend as K
import sys
import os
import gc
from config import TESTING_DATA_PATH, MODELS_PATH

def main():
    K.set_image_data_format('channels_last')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from recurrent_networks import vgg16LSTMVideoClassifier
    from UCF101_loader import load_ucf, scan_ucf_with_labels

    print(f"Prediction Configuration:")
    print(f"  Test data path: {TESTING_DATA_PATH}")
    print(f"  Models path: {MODELS_PATH}")

    vgg16_include_top = False
    config_file_path = vgg16LSTMVideoClassifier.get_config_file_path(MODELS_PATH,
                                                                     vgg16_include_top=vgg16_include_top)
    weight_file_path = vgg16LSTMVideoClassifier.get_weight_file_path(MODELS_PATH,
                                                                     vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    load_ucf(os.path.dirname(TESTING_DATA_PATH))

    predictor = vgg16LSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)

    videos = scan_ucf_with_labels(TESTING_DATA_PATH, [label for (label, label_index) in predictor.labels.items()])

    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    np.random.shuffle(video_file_path_list)

    correct_count = 0
    count = 0

    print(f"\nProcessing {len(video_file_path_list)} videos...")
    for video_file_path in video_file_path_list:
        label = videos[video_file_path]
        predicted_label = predictor.predict(video_file_path)
        print('predicted: ' + predicted_label + ' actual: ' + label)
        correct_count = correct_count + 1 if label == predicted_label else correct_count
        count += 1
        accuracy = correct_count / count
        #print('accuracy: ', accuracy)
        gc.collect()
    
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct_count}/{count})")


if __name__ == '__main__':
    main()

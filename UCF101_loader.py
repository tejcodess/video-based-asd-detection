#import urllib.request
import os
import sys
#import patoolib

#URL_LINK = 'http://crcv.ucf.edu/data/UCF101/UCF101.rar'


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_ucf(data_dir_path):
    ucf_rar = data_dir_path + '/UCF101.rar'

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)

    if not os.path.exists(ucf_rar):
        print('ucf file does not exist, downloading from internet')
#       urllib.request.urlretrieve(url=URL_LINK, filename=ucf_rar,
 #                                  reporthook=reporthook)

    print('unzipping ucf file')
#    patoolib.extract_archive(ucf_rar, outdir=data_dir_path)


def scan_ucf(data_dir_path, limit):
    # Check if we're dealing with the new structure (training_set/testing_set)
    if os.path.basename(data_dir_path) in ['training_set', 'testing_set']:
        input_data_dir_path = data_dir_path
    else:
        input_data_dir_path = data_dir_path + '/autism_data'

    result = dict()

    if not os.path.exists(input_data_dir_path):
        print(f"Warning: Directory does not exist: {input_data_dir_path}")
        return result

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = os.path.join(input_data_dir_path, f)
        if not os.path.isfile(file_path) and os.path.isdir(file_path):
            dir_count += 1
            print(f"Scanning class folder: {f}")
            if os.path.exists(file_path):
                for ff in os.listdir(file_path):
                    video_file_path = os.path.join(file_path, ff)
                    if os.path.isfile(video_file_path):
                        result[video_file_path] = f
        if limit > 0 and dir_count == limit:
            break
    
    print(f"Found {len(result)} video files in {dir_count} classes")
    return result


def scan_ucf_with_labels(data_dir_path, labels):
    # Check if we're dealing with the new structure (training_set/testing_set)
    if os.path.basename(data_dir_path) in ['training_set', 'testing_set']:
        input_data_dir_path = data_dir_path
    else:
        input_data_dir_path = data_dir_path + '/autism_data'

    result = dict()

    if not os.path.exists(input_data_dir_path):
        print(f"Warning: Directory does not exist: {input_data_dir_path}")
        return result

    print(f"Scanning for labels: {labels}")
    dir_count = 0
    for label in labels:
        file_path = os.path.join(input_data_dir_path, label)
        if os.path.exists(file_path) and os.path.isdir(file_path):
            dir_count += 1
            print(f"  Scanning class '{label}' in: {file_path}")
            video_count = 0
            for ff in os.listdir(file_path):
                video_file_path = os.path.join(file_path, ff)
                if os.path.isfile(video_file_path):
                    result[video_file_path] = label
                    video_count += 1
            print(f"    Found {video_count} videos")
    
    print(f"Total: {len(result)} video files from {dir_count} classes")
    return result



def load_ucf(data_dir_path):
    # For the new structure, check if we have training_set or testing_set
    if os.path.exists(os.path.join(data_dir_path, "training_set")):
        print(f"Using new dataset structure at: {data_dir_path}")
        return
    
    # Old structure compatibility
    UFC101_data_dir_path = data_dir_path + "/autism_data"
    if not os.path.exists(UFC101_data_dir_path):
        print(f"Dataset directory not found at: {UFC101_data_dir_path}")
        print(f"Please ensure your dataset is properly set up.")
        # download_ucf(data_dir_path)  # Commented out as we're using custom dataset


def main():
    data_dir_path = 'very_large_data'
    load_ucf(data_dir_path)


if __name__ == '__main__':
    main()

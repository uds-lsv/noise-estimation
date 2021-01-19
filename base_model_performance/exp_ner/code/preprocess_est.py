# Pre-processing script for the Estonian dataset
# text files will be converted to a pickle file to speed up loading
# The original raw text file doesn't have its validation and test set,
# we first split the original (clean) data into train/test/val split then we convert all text files into pickle files

import utils
import os
import math
import argparse
import logging


def preprocess():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument('--input_dir', type=str, required=True,
                        help='dir that contains the dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='root directory that contains images')
    parser.add_argument('--word_embedding_path', type=str, default="../data/fasttext/cc.et.300.bin",
                        help='FastText embedding path')

    # data reading
    parser.add_argument('--data_separator', type=str, default="\t",
                        help='input format: token[data_separator]label')
    parser.add_argument('--label_format', default='io', choices=['io', 'bio'],
                        help='io or bio format')
    parser.add_argument('--context_length', type=int, default=3)

    args = parser.parse_args()

    create_train_test_split(args.input_dir, args.output_dir)
    input_dir = os.path.join(args.output_dir, 'refined_version')

    noisy_file_paths = []
    train_clean_path = ""
    dev_path = ""
    test_path = ""
    for data_folder in os.listdir(input_dir):
        data_folder_path = os.path.join(input_dir, data_folder)
        if "true" in data_folder_path:
            train_clean_path = os.path.join(data_folder_path, "est.train_clean")
            dev_path = os.path.join(data_folder_path, "est.testa")
            test_path = os.path.join(data_folder_path, "est.testb")
        else:
            noisy_file_paths.append(os.path.join(data_folder_path, "est.train_noisy"))

    word_embedding, label_representation, test_label_representation = utils.load_word_embedding(args)
    remove_label_prefix = args.label_format == "io"

    for noisy_file_path in noisy_file_paths:
        save_root = os.path.join(args.output_dir, noisy_file_path.split('/')[-2])
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        utils.create_pickle_data(noisy_file_path, label_representation, remove_label_prefix, input_separator='\t',
                                 word_embedding=word_embedding, context_length=args.context_length,
                                 save_root=save_root, save_name='train_noisy.pickle')

        print(f"file {noisy_file_path} pickled")

    save_root = os.path.join(args.output_dir, train_clean_path.split('/')[-2])
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    utils.create_pickle_data(train_clean_path, label_representation, remove_label_prefix, input_separator='\t',
                             word_embedding=word_embedding, context_length=args.context_length,
                             save_root=save_root, save_name='train_clean.pickle')

    utils.create_pickle_data(dev_path, test_label_representation, False, input_separator='\t',
                             word_embedding=word_embedding, context_length=args.context_length,
                             save_root=save_root, save_name='dev.pickle')

    utils.create_pickle_data(test_path, test_label_representation, False, input_separator='\t',
                             word_embedding=word_embedding, context_length=args.context_length,
                             save_root=save_root, save_name='test.pickle')
    print(f"file {train_clean_path} pickled")


def create_train_test_split(data_dir, outpur_dir):
    doc_length = []
    train_factor, dev_factor = (0.8, 0.1)

    input_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    for input_file in input_files:
        input_file_name = input_file.split(".")[0]
        split_root = os.path.join(outpur_dir, 'refined_version', input_file_name)
        full_path = os.path.join(data_dir, input_file)

        with open(full_path, "r") as f:
            content = f.readlines()
            length = len(content)
            doc_length.append(length)
            train_length = math.floor(length * train_factor)
            dev_length = math.floor(length * dev_factor)

            train_content = content[:train_length]
            dev_content = content[train_length:train_length + dev_length]
            assert (len(dev_content) == dev_length)
            test_conent = content[train_length + dev_length:]

            if "true" in input_file_name:  # GT file
                save_content_split(train_content, 'est.train_clean', split_root)
                save_content_split(dev_content, 'est.testa', split_root)
                save_content_split(test_conent, 'est.testb', split_root)
            else:
                save_content_split(train_content, 'est.train_noisy', split_root)
            print(f"refined version of {input_file_name} created")

    for doc_l in doc_length:
        assert doc_l == doc_length[0]


def save_content_split(content, file_name, split_root):
    if not os.path.exists(split_root):
        os.makedirs(split_root)
    file_path = os.path.join(split_root, file_name)
    with open(file_path, 'w') as file:
        for item in content:
            file.write("%s\n" % item)


if __name__ == "__main__":
    preprocess()

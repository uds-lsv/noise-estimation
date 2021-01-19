import torch
from ner_datacode import DataCreation, WordEmbedding, LabelRepresentation, Evaluation
import pickle
import os
import logging
import numpy as np

def load_and_preprocess_data(path_to_data, label_representation, remove_label_prefix, input_separator, word_embedding,
                             context_length):
    # load dataset
    data_creation = DataCreation(input_separator=input_separator)
    instances = data_creation.load_connl_dataset(path_to_data, context_length, remove_label_prefix)

    # embed words in vector representation
    word_embedding.embed_instances(instances)
    x = word_embedding.instances_to_vectors(instances)

    # convert BIO/IO labels to one hot vectors
    label_representation.embed_instances(instances)
    y = label_representation.instances_to_vectors(instances)

    return instances, x, y


def load_all_processed_data(args):
    # Loading of representations
    label_representation, test_label_representation = get_representations(args)

    remove_label_prefix = args.label_format == "io"
    input_dir = args.input_dir
    train_clean_path = os.path.join(input_dir, 'estner_true/train_clean.pickle')
    train_noisy_path = os.path.join(input_dir, f'estner_noisy_round0{args.label_set}/train_noisy.pickle')
    dev_path = os.path.join(input_dir, 'estner_true/dev.pickle')
    test_path = os.path.join(input_dir, 'estner_true/test.pickle')

    train_clean, train_clean_rm_label, embedding_vector_size = load_processed_data(train_clean_path)
    train_noisy, train_noisy_rm_label, _ = load_processed_data(train_noisy_path)
    dev, dev_rm_label, _ = load_processed_data(dev_path)  # we always test on BIO sheme
    test, test_rm_label, _ = load_processed_data(test_path)

    all_datasets = [train_clean, train_noisy, dev, test]
    return all_datasets, label_representation, test_label_representation, embedding_vector_size


def load_processed_data(path_to_data):
    with open(path_to_data, "rb") as input_file:
        data_dict = pickle.load(input_file)

    return data_dict['instances'], data_dict['remove_label_prefix'], \
           data_dict['embedding_dim']

def get_representations(args):
    # LabelRepresentation, either BIO or IO (for testing always BIO)
    label_representation = LabelRepresentation()
    if args.label_format == "bio":
        label_representation.use_connl_bio_labels()
        test_label_representation = label_representation
    elif args.label_format == "io":
        label_representation.use_connl_io_labels()
        test_label_representation = LabelRepresentation()
        test_label_representation.use_connl_bio_labels()
    else:
        raise ValueError('unknown label_format')

    return label_representation, test_label_representation


def load_word_embedding(args):
    # Loading of fastText word embeddings
    word_embedding = WordEmbedding()
    word_embedding.load_fasttext(args.word_embedding_path)
    label_representation, test_label_representation = get_representations(args)

    return word_embedding, label_representation, test_label_representation



def create_pickle_data(path_to_data, label_representation, remove_label_prefix, input_separator, word_embedding,
                             context_length, save_root, save_name):
    # load dataset
    data_creation = DataCreation(input_separator=input_separator)
    instances = data_creation.load_connl_dataset(path_to_data, context_length, remove_label_prefix)

    # embed words in vector representation
    word_embedding.get_emb_for_instances(instances)


    # convert BIO/IO labels to one hot vectors
    label_representation.get_lbemb_for_instances(instances)


    data_dict = {'instances':instances, 'remove_label_prefix': remove_label_prefix,
                 'embedding_dim': word_embedding.embedding_vector_size}
    file_path = os.path.join(save_root, save_name)

    with open(file_path, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('pickle data: {} created'.format(save_name))
    return


def sample_data_same_n_for_dataset(sample_size, train_clean_full, train_noisy_full, num_labels, random_state):
    sampled_indices = []
    idx_dict = dict()
    for idx, instance in enumerate(train_clean_full):
        if np.argmax(instance.label_emb) not in idx_dict:
            idx_dict[np.argmax(instance.label_emb)] = []
        else:
            idx_dict[np.argmax(instance.label_emb)].append(idx)

    for label_idx in range(num_labels):
        sampled_indices.extend(sample_indices_one_row(sample_size, idx_dict, label_idx, random_state))
    assert len(set(sampled_indices)) == len(sampled_indices)

    sample_clean = [train_clean_full[i] for i in sampled_indices]
    sample_noisy = [train_noisy_full[i] for i in sampled_indices]

    return sample_clean, sample_noisy


def sample_indices_one_row(sample_size, idx_dict, row, random_state):
    row_indices = idx_dict[row]

    assert len(row_indices) >= sample_size

    return random_state.choice(row_indices, sample_size, replace=False)

def create_subset(instances, cs, size, random_state, sequential):
    """ Creates a subset of the data.

    Args:
        instances: list of instances
        size: Size of the subset. Samples at least 1 item if size < 1
        random_state: The subset is randomly sampled, instance of np.random_state
        sequential: If True, a random start point is picked and then a sequence of instances/words
                    is picked. If False, instances are picked randomly.

    Returns:
        subsets of corresponding xs, ys and cs

    """
    # assert len(xs) == len(ys)
    assert len(instances) >= size
    ind = _get_sample_indicies(len(instances), max(size, 1), random_state, sequential)
    instances_sub = np.array([x for i, x in enumerate(instances) if i in ind])
    cs_sub = None
    if cs is not None:
        cs_sub = np.array([y for i, y in enumerate(cs) if i in ind])
    return instances_sub, cs_sub


def _get_sample_indicies(num_items, num_samples, random_state, sequential):
    '''Returns a list of indicies that should be sampled.

    Args:
        num_items: integer value representing the pool size
        num_samples: integer value representing the number of items to be sampled
        random_state: numpy random state that should be used for random processes
        sequential: boolean value indicating whether the items should sampled sequentially or completely random

    Returns:
        A set of indices
    '''
    assert num_items >= num_samples
    numbers = list(range(num_items))
    if num_items == num_samples:
        return list(sorted(numbers))
    if sequential:
        start_number = random_state.randint(0, num_items)
        if start_number <= (num_items - num_samples):  # can generate one sequential sample
            indicies = numbers[start_number:start_number + num_samples]
        else:  # sampled would reach source list bondaries; need to generate two sequential samples
            indicies = numbers[start_number:]
            indicies.extend(numbers[:num_samples - (num_items - start_number)])
    else:
        indicies = random_state.randint(0, num_items - 1, num_samples)
    assert len(indicies) == num_samples
    return set(indicies)


def compute_noise_matrix(clean_instances, noisy_instances, label_representation):
    """
    Computes a noise or confusion matrix between clean and noisy labels.

    Args:
        clean_instances: list of clean instances (GT contained in the instances)
        noisy_instances: list of noisy instances (automated labels)

    Returns:
        A noise matrix of size num_labels x num_labels. Each row represents
        p(y_noisy| y_clean=i) for a specific clean label i
        (Formula 4 in the paper, without the log)
    """
    num_labels = label_representation.get_num_labels()
    clean_ys = np.asarray([instance.label_emb for instance in clean_instances])
    noisy_ys = np.asarray([instance.label_emb for instance in noisy_instances])


    assert num_labels == len(clean_ys[0]), f'Expected {num_labels} labels, but got: {len(clean_ys[0])}'
    assert len(clean_ys) == len(noisy_ys)

    noise_matrix = np.zeros((num_labels, num_labels))

    for clean_y, noisy_y in zip(clean_ys, noisy_ys):
        clean_y_idx = np.argmax(clean_y)
        noisy_y_idx = np.argmax(noisy_y)

        noise_matrix[clean_y_idx, noisy_y_idx] += 1

    for row in noise_matrix:
        row_sum = np.sum(row)
        if row_sum != 0:
            row /= row_sum

    return noise_matrix


# check whether a list of dirs exists, if not, create them
def check_existence_and_create(file_dirs):
    for file_dir in file_dirs:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)



def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """ Train a single epoch for given X and Y data.

    Args:
        model: Pytorch model used for training
    """

    model.train()

    for data in dataloader:
        xs, labels = data[0], data[1]
        xs = xs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        y_pred = model(xs)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()



def test_epoch(model, dataloader, device):
    """ test a single epoch for given X and Y data.

    Args:
        model: Pytorch model used for training
    """
    model.eval()
    predictions = []
    words = []
    for xs, labels, word_batch in dataloader:
        words.extend(word_batch)
        xs = xs.to(device)
        y_pred = model(xs)
        predictions += torch.argmax(y_pred, dim=1).cpu().tolist()

    return predictions, words



def simple_evaluation(model, data, dataloader, device, label_representation, SETTINGS):
    evaluation_output = long_evaluation(model, data, dataloader, device, label_representation, SETTINGS)
    return Evaluation.extract_f_score(evaluation_output)


def long_evaluation(model, data, dataloader, device, label_representation, SETTINGS):
    predictions, words = test_epoch(model, dataloader, device)
    predictions = label_representation.predictions_to_labels(predictions)

    # if predictions are in IO format, convert to BIO used for evaluation when working on test set
    if SETTINGS["LABEL_FORMAT"] == "io":
        predictions = LabelRepresentation.convert_io_to_bio_labels(predictions)

    evaluation = Evaluation(separator=SETTINGS["DATA_SEPARATOR"])
    connl_evaluation_string = evaluation.create_connl_evaluation_format(data, words, predictions)
    return evaluation.evaluate_evaluation_string(connl_evaluation_string)




def unify_scale(l1, l2):
    """
    unify the scales of values in the two lists.
    """
    assert(len(l1) == len(l2))
    l1 = (l1-np.min(l1))/(np.max(l1)-np.min(l1))
    l2 = (l2-np.min(l2))/(np.max(l2)-np.min(l2))
    return l1, l2


def create_logger(log_root, args, starting_time):
    log_path, log_dir = create_log_path(log_root, args, starting_time)

    # check if the file exist

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_path)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger, log_dir


def create_log_path(log_root, args, starting_time):
    staring_time_str = starting_time.strftime("%m_%d_%H_%M_%S")
    suffix = staring_time_str

    suffix += f'_ls{args.label_set}_{args.exp_settings}'
    # suffix += f'_nlb{args.nl_batch_size}'
    #
    # if args.use_vat:
    #     suffix += f'_ulb{args.ul_batch_size}_vs{args.vat_start_epoch}_vw{args.vat_weight}'
    #
    # if args.dual_ul:
    #     suffix += f'_dul'

    log_dir = os.path.join(log_root, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, 'log.txt')
    return log_path, log_dir
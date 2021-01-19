import itertools
import os

def load_mnist_data():
    from keras.datasets import mnist

    (X_train, y_train), _ = mnist.load_data()
    num_labels = 10

    class Instance:

        def __init__(self, label):
            self.label = label

    clean_data = [Instance(label) for label in y_train]

    return clean_data, num_labels


def load_ner_data_specific_round(labeling_round):
    return load_ner_data(os.path.join("..", "data", "noisy-ner", "estner_clean_train.tsv"),
                         os.path.join("..", "data", "noisy-ner", f"NoisyNER_labelset{labeling_round}_train.tsv"))


def load_ner_data(path_clean_data, path_noisy_data):
    from .ner_datacode import DataCreation

    remove_label_prefix = True
    
    data_creation = DataCreation(input_separator="\t")
    clean_data = data_creation.load_conll_dataset(path_clean_data, 0, remove_label_prefix=remove_label_prefix)
    noisy_data = data_creation.load_conll_dataset(path_noisy_data, 0, remove_label_prefix=remove_label_prefix)
        
    label_idx_to_label = {0: "O", 1: "LOC", 2: "ORG", 3: "PER"}
    label_to_label_idx = {v:k for k,v in label_idx_to_label.items()}
    num_labels = len(label_to_label_idx)
    
    for instance in itertools.chain(clean_data, noisy_data):
        instance.label = label_to_label_idx[instance.label] # using numeric labels
    
    return clean_data, noisy_data, label_idx_to_label, label_to_label_idx, num_labels


def load_clothing1m_data(path_data_clean = "../data/clothing1m/annotations/noisy_label_kv.txt",
                         path_data_noisy =  "../data/clothing1m/annotations/clean_label_kv.txt"):
    
    class Instance:
        def __init__(self, image, label):
            self.image = image
            self.label = label

    # go through file with clean labels
    # create a map from image to Instance object
    # when going through the noisy file later on, 
    # we need to combine clean and noisy Instance 
    # for the same image
    clean_file_map = {}
    with open(path_data_clean, "r") as clean_file:
        for line in clean_file:
            image, label = line.split(" ")
            if image in clean_file_map:
                raise Exception("Image {} occurs multiple times".format(image))

            clean_file_map[image] = Instance(image, int(label))
    
    # combine clean and noisy labels for the same image
    clean_instances = []
    noisy_instances = []
    with open(path_data_noisy, "r") as noisy_file:
        for line in noisy_file:
            image, label = line.split(" ")
            if image in clean_file_map:
                clean_instances.append(clean_file_map[image])
                noisy_instances.append(Instance(image, int(label)))
    
    # evaluate
    correct = 0            
    for clean_instance, noisy_instance in zip(clean_instances, noisy_instances):
        assert noisy_instance.image == clean_instance.image
        if noisy_instance.label == clean_instance.label:
            correct += 1
    print("Accuracy of Clothing1M Pairs clean-noisy: {}".format(correct/len(clean_instances)))
    
    # label data
    label_names = ["T-Shirt", "Shirt", "Knitwear", "Chiffon", "Sweater", "Hoodie", "Windbreaker", "Jacket", 
                   "Downcoat", "Suit", "Shawl", "Dress", "Vest", "Underwear"]
    label_idx_to_label_name_map = {idx:name for idx, name in zip(range(len(label_names)), label_names)}
    num_labels = len(label_idx_to_label_name_map)
    
    return clean_instances, noisy_instances, label_idx_to_label_name_map, num_labels

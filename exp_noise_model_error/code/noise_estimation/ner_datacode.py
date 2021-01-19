class Instance:
    
    def __init__(self, word, label, left_context, right_context):
        self.word = word
        self.label = label
        self.left_context = left_context
        self.right_context = right_context
        self.label_emb = None
        
    def __str__(self):
        return f"NERInstance({self.word}|{self.label}|{self.left_context}-{self.right_context}"
    
class DataCreation:
    
    def __init__(self, input_separator=" ", padding_word_string="<none>"):
        self.input_separator = input_separator
        self.padding_word_string = padding_word_string
    
    def remove_label_prefix(self, label):
        """ CoNLL2003 distinguishes between I- and B- labels,
            e.g. I-LOC and B-LOC. Drop this distinction to
            reduce the number of labels/increase the number
            of instances per label.
        """
        if label.startswith("I-") or label.startswith("B-"):
            return label[2:]
        else:
            return label
        
    def pad_before(self, a_list, target_length):
        return (target_length - len(a_list)) * [self.padding_word_string] + a_list

    def pad_after(self, a_list, target_length):
        return a_list + (target_length - len(a_list)) * [self.padding_word_string]
    
    def load_conll_dataset(self, path, context_length, remove_label_prefix=False):
        with open(path, mode="r", encoding="utf-8") as input_file:
            lines = input_file.readlines()
        
        tokens = []
        labels = []
        for line in lines:
            line = line.strip()
        
            # skip begin of document indicator (used in some files)
            if line.startswith("-DOCSTART-"):
                continue
            
            # skip empty line / end of sentence marker (used in some files) 
            if len(line) == 0:
                continue
            
            # Skip marker (used in some files)
            if line == "--": 
                continue
                
            elements = line.split(self.input_separator)
            
            tokens.append(elements[0])
            # Take last element of this line as label, in between there might be e.g. the POS tag which we ignore here
            if len(elements) > 1: 
                labels.append(elements[-1])
            else:
                raise Exception(f"Line {line} did not provide a label. Elements are {elements}")
        
        assert len(tokens) == len(labels)
        
        instances = []
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if remove_label_prefix:
                label = self.remove_label_prefix(label)

            left_context = tokens[max(0,i-context_length):i]
            right_context = tokens[i+1:i+1+context_length]

            left_context = self.pad_before(left_context, context_length)
            right_context = self.pad_after(right_context, context_length)
            instances.append(Instance(token, label, left_context, right_context))
        
        return instances

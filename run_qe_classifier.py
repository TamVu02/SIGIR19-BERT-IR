# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import random
import json
import numpy as np

flags = tf.compat.v1.flags

FLAGS = flags.Flag

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 2,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 5,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "fold", 5,
    "run fold")

flags.DEFINE_string(
    "query_field", None,
    "None if no field, else title, desc, narr, question")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a_list, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a_list: list of strings.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid

        #  a list of queries
        # E.g. ["obama family tree"]
        # or ["obama family tree", "find information about obama's history"]
        self.text_a_list = text_a_list

        # a string of the document
        # #.g. "barack obama -- wikipeid"
        self.text_b = text_b

        # relevance label: 0 or 1
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.guid=guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.io.gfile.GFile(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


# class RobustProcessor(DataProcessor):

#     def __init__(self):
#         self.max_test_depth = 100  # for testing, we re-rank the top 100 results
#         self.max_train_depth = 1000  # for training, we use negative samples from the top 1000 documents

#         self.n_folds = 5
#         self.fold = FLAGS.fold
#         self.query_field = FLAGS.query_field

#         self.train_folds = [(self.fold + i) % self.n_folds + 1 for i in range(self.n_folds - 1)]
#         self.test_folds = (self.fold + self.n_folds - 1) % self.n_folds + 1
#         tf.compat.v1.logging.info("Train Folds: {}".format(str(self.train_folds)))
#         tf.compat.v1.logging.info("Test Fold: {}".format(str(self.test_folds)))

#     def get_train_examples(self, data_dir):
#         examples = []
#         train_files = ["{}.trec.with_json".format(i) for i in self.train_folds]
#         qrel_file = open(os.path.join(data_dir, "qrels"))
#         qrels = self._read_qrel(qrel_file)
#         q_fields = FLAGS.query_field.split(' ')
#         tf.compat.v1.logging.info("Using query fields {}".format(' '.join(q_fields)))

#         for file_name in train_files:
#             train_file = open(os.path.join(data_dir, file_name))
#             for i, line in enumerate(train_file):
#                 items = line.strip().split('#')
#                 trec_line = items[0]
#                 json_dict = json.loads('#'.join(items[1:]))
#                 q = json_dict["query"]
#                 q_text_list = [tokenization.convert_to_unicode(q[field]) for field in q_fields]
#                 d = tokenization.convert_to_unicode(json_dict["doc"]["body"])

#                 qid, _, docid, r, _, _ = trec_line.strip().split(' ')
#                 r = int(r)
#                 if r > self.max_train_depth:
#                     continue
#                 label = tokenization.convert_to_unicode("0")
#                 if (qid, docid) in qrels or (qid, docid.split('_')[0]) in qrels:
#                     label = tokenization.convert_to_unicode("1")
#                 guid = "train-%s-%s" % (qid, docid)
#                 examples.append(
#                     InputExample(guid=guid, text_a_list=q_text_list, text_b=d, label=label)
#                 )
#             train_file.close()
#         random.shuffle(examples)
#         return examples

#     def get_test_examples(self, data_dir):
#         examples = []
#         dev_file = open(os.path.join(data_dir, "{}.trec.with_json".format(self.test_folds)))
#         qrel_file = open(os.path.join(data_dir, "qrels"))
#         qrels = self._read_qrel(qrel_file)
#         q_fields = FLAGS.query_field.split(' ')
#         tf.compat.v1.logging.info("Using query fields {}".format(' '.join(q_fields)))

#         for i, line in enumerate(dev_file):
#             items = line.strip().split('#')
#             trec_line = items[0]
#             json_dict = json.loads('#'.join(items[1:]))
#             q = json_dict["query"]
#             q_text_list = [tokenization.convert_to_unicode(q[field]) for field in q_fields]

#             d = tokenization.convert_to_unicode(json_dict["doc"]["body"])
#             qid, _, docid, r, _, _ = trec_line.strip().split(' ')
#             r = int(r)
#             if r > self.max_test_depth:
#                 continue
#             label = tokenization.convert_to_unicode("0")
#             if (qid, docid) in qrels or (qid, docid.split('_')[0]) in qrels:
#                 label = tokenization.convert_to_unicode("1")
#             guid = "test-%s-%s" % (qid, docid)
#             examples.append(
#                 InputExample(guid=guid, text_a_list=q_text_list, text_b=d, label=label)
#             )
#         dev_file.close()
#         return examples

#     def _read_qrel(self, qrel_file):
#         qrels = set()
#         for line in qrel_file:
#             qid, _, docid, rel = line.strip().split(' ')
#             rel = int(rel)
#             if rel > 0:
#                 qrels.add((qid, docid))
#         return qrels

#     def get_labels(self):
#         return ["0", "1"]
class MyRobust04Processor(DataProcessor):

    def __init__(self):
        self.max_test_depth = 100
        self.max_train_depth = 200
        self.n_folds = 5
        self.fold = FOLD
        self.q_fields = QUERY_FIELD.split(' ')
        tf.compat.v1.logging.info("Using query fields {}".format(' '.join(self.q_fields)))

        self.train_folds = [(self.fold + i) % self.n_folds + 1 for i in range(self.n_folds - 1)]
        self.dev_folds = (self.fold + self.n_folds - 2) % self.n_folds + 1
        self.test_folds = (self.fold + self.n_folds - 1) % self.n_folds + 1
        tf.compat.v1.logging.info("Train Folds: {}".format(str(self.train_folds)))
        tf.compat.v1.logging.info("Dev Fold: {}".format(str(self.dev_folds)))
        tf.compat.v1.logging.info("Test Fold: {}".format(str(self.test_folds)))

    def get_train_examples(self, data_dir):
        examples = []
        train_files = ["{}.trec.with_json".format(i) for i in self.train_folds]
        ###############

        qrel_file = tf.io.gfile.GFile(os.path.join(data_dir, "qrels"))
        qrels = self._read_qrel(qrel_file)
        tf.compat.v1.logging.info("Positive relevance Qrel size: {}".format(len(qrels)))

        query_file = tf.io.gfile.GFile(os.path.join(data_dir, "queries.json"))
        qid2queries = self._read_queries(query_file)
        tf.compat.v1.logging.info("Loaded {} queries.".format(len(qid2queries)))

        for file_name in train_files:
            train_file = tf.io.gfile.GFile(os.path.join(data_dir, file_name))
            for i, line in enumerate(train_file):
               
                items = line.strip().split('#')
                trec_line = items[0]
                if len(trec_line.strip().split(' '))!=2:
                    continue
                qid, docid= trec_line.strip().split(' ')
                
                # if int(docid.split('_')[-1].split('-')[-1])!=0: #and random.random() > 0.1:
                #     continue
                    
                assert qid in qid2queries, "QID {} not found".format(qid)
                q_json_dict = qid2queries[qid]
                # for fiel in self.q_fields:
                #     print(fiel)
                q_text_list = [tokenization.convert_to_unicode(q_json_dict[field]) for field in self.q_fields]
                json_dict = json.loads('#'.join(items[1:]).replace("\\",'-'))
                d = tokenization.convert_to_unicode(json_dict["doc"]["body"])
                # r = int(r)
                # if r > self.max_train_depth:
                #     continue
                label = tokenization.convert_to_unicode("0")
                if (qid, docid) in qrels:
                    #print('///////////////////////////////////')
                    label = tokenization.convert_to_unicode("1")
                guid = "train-%s-%s" % (qid, docid)
                examples.append(
                    InputExample(guid=guid, text_a_list=q_text_list, text_b=d, label=label)
                )
            train_file.close()
        random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir):
        examples = []
        dev_file = tf.io.gfile.GFile(os.path.join(data_dir, "{}.trec.with_json".format(self.dev_folds)))
        qrel_file = tf.io.gfile.GFile(os.path.join(data_dir, "qrels"))
        qrels = self._read_qrel(qrel_file)
        tf.compat.v1.logging.info("Positive relevance Qrel size: {}".format(len(qrels)))

        query_file = tf.io.gfile.GFile(os.path.join(data_dir, "queries.json"))
        qid2queries = self._read_queries(query_file)
        tf.compat.v1.logging.info("Loaded {} queries.".format(len(qid2queries)))
        
        flag = False
        for i, line in enumerate(dev_file):
            items = line.strip().split('#')
            trec_line = items[0]

            qid, docid= trec_line.strip().split(' ')
            assert qid in qid2queries, "QID {} not found".format(qid)
            q_json_dict = qid2queries[qid]
            q_text_list = [tokenization.convert_to_unicode(q_json_dict[field]) for field in self.q_fields]
            #print(items[1:])
            json_dict = json.loads('#'.join(items[1:]).replace("\\",'-'))
            d = tokenization.convert_to_unicode(json_dict["doc"]["body"])

            # r = int(r)
            # if r > self.max_test_depth:
            #     continue
            label = tokenization.convert_to_unicode("0")
            if (qid, docid) in qrels:
                label = tokenization.convert_to_unicode("1")
                flag = True
            guid = "dev-%s-%s" % (qid, docid)
            examples.append(
                InputExample(guid=guid, text_a_list=q_text_list, text_b=d, label=label)
            )
        dev_file.close()
        if not flag:
            tf.compat.v1.logging.warning("No relevant document is labeled!")
        return examples

    def get_test_examples(self, data_dir):
        examples = []
        dev_file = tf.io.gfile.GFile(os.path.join(data_dir, "{}.trec.with_json".format(self.test_folds)))
        qrel_file = tf.io.gfile.GFile(os.path.join(data_dir, "qrels"))
        qrels = self._read_qrel(qrel_file)
        tf.compat.v1.logging.info("Positive relevance Qrel size: {}".format(len(qrels)))

        query_file = tf.io.gfile.GFile(os.path.join(data_dir, "queries.json"))
        qid2queries = self._read_queries(query_file)
        tf.compat.v1.logging.info("Loaded {} queries.".format(len(qid2queries)))

        for i, line in enumerate(dev_file):
            items = line.strip().split('#')
            trec_line = items[0]

            qid, docid= trec_line.strip().split(' ')
            assert qid in qid2queries, "QID {} not found".format(qid)
            q_json_dict = qid2queries[qid]
            q_text_list = [tokenization.convert_to_unicode(q_json_dict[field]) for field in self.q_fields]

            json_dict = json.loads('#'.join(items[1:]).replace("\\",'-'))
            d = tokenization.convert_to_unicode(json_dict["doc"]["body"])
            # if int(r) > self.max_test_depth:
            #     continue
            label = tokenization.convert_to_unicode("0")
            if (qid, docid) in qrels or (qid, docid.split('_')[0]) in qrels:
                label = tokenization.convert_to_unicode("1")
            guid = "test-%s-%s" % (qid, docid)
            examples.append(
                InputExample(guid=guid, text_a_list=q_text_list, text_b=d, label=label)
            )
        dev_file.close()
        return examples

    def _read_qrel(self, qrel_file):
        qrels = set()
        for line in qrel_file:
            if len(line.strip().split(','))!=3:
                continue
            json_dict = json.loads(line)
            qid = json_dict['query_id']
            docid = json_dict['doc_id']
            rel = json_dict['relevance']
            #qid,docid,rel = line.strip().split(',')
            #rel = int(rel.split(':')[1][:-1])
            if rel > 0:
                qrels.add((qid, docid))
        return qrels

    def _read_queries(self, query_file):
        qid2queries = {}
        for i, line in enumerate(query_file):
            json_dict = json.loads(line)
            #print(json_dict)
            qid = json_dict['query_id']
            qid2queries[qid] = json_dict
            if i < 3:
              tf.compat.v1.logging.info("Example Q: {}".format(json_dict))
        return qid2queries
   
    def get_labels(self):
        return ["0", "1"]


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            guid="",
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    guid=example.guid
    
    tokens_a = []
    for text in example.text_a_list:
        tokens = tokenizer.tokenize(text)
        tokens_a += tokens
        tokens_a.append('[SEP]')
    tokens_a = tokens_a[:-1]

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.compat.v1.logging.info("*** Example ***")
        tf.compat.v1.logging.info("guid: %s" % (example.guid))
        tf.compat.v1.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        guid=guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        
        features["guid"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.guid.encode('utf-8')]))
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        
#         feat_without_id = collections.OrderedDict({key{k:v for k,v in value.items() if k != 'guid'}
#                                                  key:value for key,value in features.item()})

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "guid" : tf.io.FixedLenFeature([], tf.string),
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
        "is_real_example": tf.io.FixedLenFeature([], tf.int64),
    }
    #copy_name_to_feat={i:name_to_features[i] for i in name_to_features if i!='guid'}

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.compat.v1.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        #batch_size = params["batch_size"]
        batch_size = 32

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1]

    output_weights = tf.Variable(
        initial_value=np.zeros((num_labels, hidden_size),dtype=np.float32),
        name="output_weights")

    output_bias = tf.Variable(
        initial_value=np.zeros((num_labels),dtype=np.float32),
        name="output_bias")

    with tf.compat.v1.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, rate =0.1)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        #guid = features["guid"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                #print(tf.train.list_variables(init_checkpoint))

                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            #global_step = tf.compat.v1.train.get_or_create_global_step()
#             init_lr=learning_rate
#             if num_warmup_steps:
#                 global_steps_int = tf.cast(global_step, tf.int32)
#                 warmup_steps_int = tf.convert_to_tensor(num_warmup_steps, dtype=tf.int32)

#                 global_steps_float = tf.cast(global_steps_int, tf.float32)
#                 warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

#                 warmup_percent_done = global_steps_float / warmup_steps_float
#                 warmup_learning_rate = init_lr * warmup_percent_done

#                 is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
#                 learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
#             optimizer=tf.keras.optimizers.experimental.AdamW(
#                 learning_rate=learning_rate,
#                 weight_decay=0.01,
#                 beta_1=0.9,
#                 beta_2=0.999,
#                 epsilon=1e-06)
#             tvars = tf.compat.v1.trainable_variables()
#             grads = tf.gradients(total_loss, tvars)
#             (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
#             train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
#             new_global_step = global_step + 1
#             train_op = tf.group(train_op, [global_step.assign(new_global_step)])
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.estimator.EstimatorSpec(
                 mode=mode,
                 loss=total_loss,
                 train_op=train_op)
                 #scaffold=scaffold_fn)

#             output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
#                 mode=mode,
#                 loss=total_loss,
#                 train_op=train_op,
#                 scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.compat.v1.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.compat.v1.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.estimator.EstimatorSpec(
                 mode=mode,
                 loss=total_loss,
                 eval_metrics=eval_metrics)
                 #scaffold=scaffold_fn)
#             output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
#                 mode=mode,
#                 loss=total_loss,
#                 eval_metrics=eval_metrics,
#                 scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities,
                            "guid": features["guid"]})
                #scaffold=scaffold_fn)
#             output_spec = tf.compat.v1.estimator.tpu.TPUEssample_id': features['sample_id']timatorSpec(
#                 mode=mode,
#                 predictions={"probabilities": probabilities},
#                 scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    processors = {
        "robust": RobustProcessor,
        "clueweb": ClueWebProcessor,
        "robustpassage": RobustPassageProcessor,
        "cluewebpassage": ClueWebPassageProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.io.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    
    if FLAGS.do_train:
        
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        train_file = os.path.join(FLAGS.output_dir, "train.tfrecord")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.compat.v1.logging.info("***** Running prediction*****")
        tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.io.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.compat.v1.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

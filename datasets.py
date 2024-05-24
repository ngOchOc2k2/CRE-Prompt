import random
import os
import math
import json
import numpy as np
import pickle

from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class RelationDataset(Dataset):
    def __init__(self, data, target, config=None):
        self.data = data
        self.target = target
        self.config = config
        self.bert = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.target[idx])
    

def collate_fn(data):
    tokens, label = zip(*data)
    # label = torch.tensor(label)
    label = torch.LongTensor(label)
    tokens = torch.tensor(tokens)
    # tokens = [torch.tensor(item[0]["tokens"]) for item in data]
    # ind = [item[1] for item in data]

    # try:
    #     key = [torch.tensor(item[0]["key"]) for item in data]
    #     return (label, tokens, key, ind)
    # except:
    #     return (label, tokens, ind)

    return tokens, label



def get_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_path, 
        additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"]
    )
    return tokenizer


def target_transform(file):
    id2rel = json.load(open(file, "r", encoding="utf-8"))
    rel2id = {}
    for i, x in enumerate(id2rel):
        rel2id[x] = i
    return id2rel, rel2id


def extract_tokens_between_markers(tokens, start_marker, end_marker):
    start_idx = tokens.index(start_marker)
    end_idx = tokens.index(end_marker)
    return " ".join(tokens[start_idx + 1:end_idx])


def _read_data(tokenizer, file, save_data_path, rel2id, args):
    if os.path.isfile(save_data_path):
        with open(save_data_path, "rb") as f:
            datas = pickle.load(f)
        train_dataset, val_dataset, test_dataset = datas
        return train_dataset, val_dataset, test_dataset
    else:
        data = json.load(open(file, "r", encoding="utf-8"))
        train_dataset = [[] for i in range(args.num_of_relation)]
        val_dataset = [[] for i in range(args.num_of_relation)]
        test_dataset = [[] for i in range(args.num_of_relation)]
        for relation in data.keys():
            rel_samples = data[relation]
            random.shuffle(rel_samples)
            count = 0
            count1 = 0
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {}
                tokenized_sample["relation"] = rel2id[sample["relation"]]
                
                text = extract_tokens_between_markers(sample["tokens"], "[E11]", "[E12]") \
                    + "[MASK]" \
                    + extract_tokens_between_markers(sample["tokens"], "[E21]", "[E22]") \
                    + "[SEP]" \
                    + " ".join(sample["tokens"]) \
                    + "[SEP]" \
                
                tokenized_sample["tokens"] = tokenizer.encode(text, padding="max_length", truncation=True, max_length=args.max_length)
                if args.task_name == "FewRel":
                    if i < args.num_of_train:
                        train_dataset[rel2id[relation]].append(tokenized_sample)
                    elif i < args.num_of_train + args.num_of_val:
                        val_dataset[rel2id[relation]].append(tokenized_sample)
                    else:
                        test_dataset[rel2id[relation]].append(tokenized_sample)
                else:
                    if i < len(rel_samples) // 5 and count <= 40:
                        count += 1
                        test_dataset[rel2id[relation]].append(tokenized_sample)
                    else:
                        count1 += 1
                        train_dataset[rel2id[relation]].append(tokenized_sample)
                        if count1 >= 320:
                            break
        with open(save_data_path, "wb") as f:
            pickle.dump((train_dataset, val_dataset, test_dataset), f)
        return train_dataset, val_dataset, test_dataset



def build_continual_dataloader(args):
    dataloader = list()
    dataloader_per_cls = dict()
    class_mask = list()
    target_task_map = dict()

    # Set up data path
    use_marker = ""
    if args.dataname in ["FewRel"]:
        args.data_file = os.path.join(args.data_path, "data_with{}_marker.json".format(use_marker))
        args.relation_file = os.path.join(args.data_path, "id2rel.json")
        args.num_of_relation = 80
        args.nb_classes = 80
        args.num_of_train = 420
        args.num_of_val = 140
        args.num_of_test = 140
    elif args.dataname in ["TACRED"]:
        args.data_file = os.path.join(args.data_path, "data_with{}_marker_tacred.json".format(use_marker))
        args.relation_file = os.path.join(args.data_path, "id2rel_tacred.json")
        args.num_of_relation = 40
        args.nb_classes = 40
        args.num_of_train = 420
        args.num_of_val = 140
        args.num_of_test = 140

    file_name = "{}.pkl".format("-".join([str(x) for x in [args.dataname, args.seed]]))
    mid_dir = ""
    for temp_p in ["local_datasets", "_process_path"]:
        mid_dir = os.path.join(mid_dir, temp_p)
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
    
    save_data_path = os.path.join(mid_dir, file_name)

    # Transform on dataset
    tokenizer = get_tokenizer(args)
    id2rel, rel2id = target_transform(args.relation_file)

    args.num_tasks = args.num_of_relation // args.rel_per_task

    training_dataset, valid_dataset, test_dataset = _read_data(tokenizer, args.data_file, save_data_path, rel2id, args)
    training_dataset_mean, valid_dataset_mean, test_dataset_mean = _read_data(tokenizer, args.data_file, save_data_path, rel2id, args)

    splited_dataset, class_mask, target_task_map = split_single_dataset(training_dataset, valid_dataset, test_dataset, args)
    splited_dataset_per_cls = split_single_class_dataset(training_dataset_mean, valid_dataset_mean, test_dataset_mean, class_mask, args)

    for i in range(args.num_tasks):
        dataset_train, dataset_val, dataset_test = splited_dataset[i]

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=collate_fn
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=collate_fn
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=collate_fn
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'test': data_loader_test})


    for i in range(len(class_mask)):
        for cls_id in class_mask[i]:
            dataset_train_cls, dataset_val_cls, dataset_test_cls = splited_dataset_per_cls[cls_id]

            sampler_train = torch.utils.data.RandomSampler(dataset_train_cls)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_cls)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test_cls)

            data_loader_train_cls = torch.utils.data.DataLoader(
                dataset_train_cls, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                collate_fn=collate_fn
            )

            data_loader_val_cls = torch.utils.data.DataLoader(
                dataset_val_cls, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                collate_fn=collate_fn
            )

            data_loader_test_cls = torch.utils.data.DataLoader(
                dataset_test_cls, sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                collate_fn=collate_fn
            )

            dataloader_per_cls[cls_id] = {'train': data_loader_train_cls, 'val': data_loader_val_cls, 'test': data_loader_test_cls}

    return dataloader, dataloader_per_cls, class_mask, target_task_map


def split_single_dataset(dataset_train, dataset_val, dataset_test, args):
    nb_classes = args.num_of_relation
    classes_per_task = args.rel_per_task

    labels = [i for i in range(nb_classes)]

    split_datasets = list()
    mask = list()

    random.shuffle(labels)

    target_task_map = {}

    for i in range(args.num_tasks):

        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)
        for k in scope:
            target_task_map[k] = i


        subset_train_data = []
        subset_train_targets = []
        subset_val_data = []
        subset_val_targets = []
        subset_test_data = []
        subset_test_targets = []

        for k in scope:
            for sample in dataset_train[k]:
                subset_train_data.append(sample['tokens'])
                subset_train_targets.append(sample['relation'])
            
            for sample in dataset_val[k]:
                subset_val_data.append(sample['tokens'])
                subset_val_targets.append(sample['relation'])

            for sample in dataset_test[k]:
                subset_test_data.append(sample['tokens'])
                subset_test_targets.append(sample['relation'])

        subset_train = RelationDataset(subset_train_data, subset_train_targets)
        subset_val = RelationDataset(subset_val_data, subset_val_targets)
        subset_test = RelationDataset(subset_test_data, subset_test_targets)

        split_datasets.append([subset_train, subset_val, subset_test])

    return split_datasets, mask, target_task_map



def split_single_class_dataset(dataset_train, dataset_val, dataset_test, mask, args):
    nb_classes = args.num_of_relation
    print(nb_classes)
    split_datasets = dict()
    print(mask)
    for i in range(len(mask)):
        single_task_labels = mask[i]

        for cls_id in single_task_labels:
            subset_train_data = []
            subset_train_targets = []
            subset_val_data = []
            subset_val_targets = []
            subset_test_data = []
            subset_test_targets = []

            for sample in dataset_train[cls_id]:
                subset_train_data.append(sample['tokens'])
                subset_train_targets.append(sample['relation'])

            for sample in dataset_val[cls_id]:
                subset_val_data.append(sample['tokens'])
                subset_val_targets.append(sample['relation'])

            for sample in dataset_test[cls_id]:
                subset_test_data.append(sample['tokens'])
                subset_test_targets.append(sample['relation'])

            subset_train = RelationDataset(subset_train_data, subset_train_targets)
            subset_val = RelationDataset(subset_val_data, subset_val_targets)
            subset_test = RelationDataset(subset_test_data, subset_test_targets)

            split_datasets[cls_id] = [subset_train, subset_val, subset_test]


    return split_datasets



#     for i in range(len(class_mask)):
#         for cls_id in class_mask[i]:
#             dataset_train_cls, dataset_val_cls = splited_dataset_per_cls[cls_id]

#             if args.distributed and utils.get_world_size() > 1:
#                 num_tasks = utils.get_world_size()
#                 global_rank = utils.get_rank()

#                 sampler_train = torch.utils.data.DistributedSampler(
#                     dataset_train_cls, num_replicas=num_tasks, rank=global_rank, shuffle=True)

#                 sampler_val = torch.utils.data.SequentialSampler(dataset_val_cls)
#             else:
#                 sampler_train = torch.utils.data.RandomSampler(dataset_train_cls)
#                 sampler_val = torch.utils.data.SequentialSampler(dataset_val_cls)

#             data_loader_train_cls = torch.utils.data.DataLoader(
#                 dataset_train_cls, sampler=sampler_train,
#                 batch_size=args.batch_size,
#                 num_workers=args.num_workers,
#                 pin_memory=args.pin_mem,
#             )

#             data_loader_val_cls = torch.utils.data.DataLoader(
#                 dataset_val_cls, sampler=sampler_val,
#                 batch_size=args.batch_size,
#                 num_workers=args.num_workers,
#                 pin_memory=args.pin_mem,
#             )

#             dataloader_per_cls[cls_id] = {'train': data_loader_train_cls, 'val': data_loader_val_cls}

#     return dataloader, dataloader_per_cls, class_mask, target_task_map

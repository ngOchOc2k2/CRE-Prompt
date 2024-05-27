from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader

from .swag import SWAG
from .model import *
from .backbone import *
from .prompt import *
from .utils import *

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from sklearn.mixture import GaussianMixture

from tqdm import tqdm, trange
import pickle

class Manager(object):
    def __init__(self, args):
        super().__init__()

    def train_classifier(self, args, classifier, swag_classifier, replayed_epochs, name):
        classifier.train()
        swag_classifier.train()

        modules = [classifier]
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.classifier_lr}])

        def train_data(data_loader_, name=""):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0
            for step, (labels, tokens, _) in enumerate(td):
                try:
                    optimizer.zero_grad()

                    # batching
                    sampled += len(labels)
                    targets = labels.type(torch.LongTensor).to(args.device)
                    tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                    # classifier forward
                    reps = classifier(tokens)

                    # prediction
                    probs = F.softmax(reps, dim=1)
                    _, pred = probs.max(1)
                    hits = (pred == targets).float()

                    # accuracy
                    total_hits += hits.sum().data.cpu().numpy().item()

                    # loss components
                    loss = F.cross_entropy(input=reps, target=targets, reduction="mean")
                    losses.append(loss.item())
                    loss.backward()

                    # params update
                    torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                    optimizer.step()

                    # display
                    td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled)
                except:
                    continue

        for e_id in range(args.classifier_epochs):
            data_loader = get_data_loader(args, replayed_epochs[e_id % args.replay_epochs], shuffle=True)
            train_data(data_loader, f"{name}{e_id + 1}")
            swag_classifier.collect_model(classifier)
            if e_id % args.sample_freq == 0 or e_id == args.classifier_epochs - 1:
                swag_classifier.sample(0.0)
                bn_update(data_loader, swag_classifier)

    def train_encoder(self, args, encoder, training_data, task_id):
        encoder.train()
        classifier = Classifier(args=args).to(args.device)
        classifier.train()
        data_loader = get_data_loader(args, training_data, shuffle=True)


        # Train classifier and model embeddings

        base_params = list(encoder.encoder.embeddings.parameters())
        classifier_params = list(classifier.parameters())
        base_params = {
            'params': base_params,
            'lr': args.encoder_lr
        }
        classifier_params = {
            'params': classifier_params,
            'lr': args.encoder_lr * 10
        }

        # modules = [classifier, encoder.encoder.embeddings]
        # modules = nn.ModuleList(modules)
        # modules_parameters = modules.parameters()

        # optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.encoder_lr}])

        optimizer = torch.optim.Adam([base_params, classifier_params])

        def train_data(data_loader_, name="", e_id=0):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0

            for step, (labels, tokens, _) in enumerate(td):
                try:
                    optimizer.zero_grad()

                    # batching
                    sampled += len(labels)
                    targets = labels.type(torch.LongTensor).to(args.device)
                    tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                    # encoder forward
                    encoder_out = encoder(tokens)

                    # classifier forward
                    reps = classifier(encoder_out["x_encoded"])

                    # prediction
                    probs = F.softmax(reps, dim=1)
                    _, pred = probs.max(1)
                    total_hits += (pred == targets).float().sum().data.cpu().numpy().item()

                    # loss components
                    CE_loss = F.cross_entropy(input=reps, target=targets, reduction="mean")
                    loss = CE_loss
                    losses.append(loss.item())
                    loss.backward()

                    # params update
                    torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                    optimizer.step()

                    # display
                    td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled)
                except:
                    continue

        for e_id in range(args.encoder_epochs):
            train_data(data_loader, f"train_encoder_epoch_{e_id + 1}", e_id)

    def train_prompt_pool(self, args, encoder, prompt_pool, training_data, task_id):
        encoder.eval()
        classifier = Classifier(args=args).to(args.device)
        classifier.train()
        modules = [classifier, prompt_pool]
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.prompt_pool_lr}])

        data_loader = get_data_loader(args, training_data, shuffle=True)
        new_training_data = []
        td = tqdm(data_loader, desc=f"get_prompt_key_task_{task_id+1}")
        for step, (labels, tokens, _) in enumerate(td):
            targets = labels.type(torch.LongTensor).to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            # encoder forward
            encoder_out = encoder(tokens)

            tokens = tokens.cpu().detach().numpy()
            x_key = encoder_out["x_encoded"].cpu().detach().numpy()
            # add to new training data
            for i in range(len(labels)):
                new_training_data.append({"relation": labels[i], "tokens": tokens[i], "key": x_key[i]})
            td.set_postfix()


        # new data loader
        data_loader = get_data_loader(args, new_training_data, shuffle=True)

        def train_data(data_loader_, name="", e_id=0):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0

            for step, (labels, tokens, keys, _) in enumerate(td):
                optimizer.zero_grad()

                # batching
                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                x_key = torch.stack([x.to(args.device) for x in keys], dim=0)

                # encoder forward
                encoder_out = encoder(tokens, prompt_pool, x_key)

                # classifier forward
                reps = classifier(encoder_out["x_encoded"])

                # prediction
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)
                total_hits += (pred == targets).float().sum().data.cpu().numpy().item()

                # loss components
                prompt_reduce_sim_loss = -args.pull_constraint_coeff * encoder_out["reduce_sim"]
                CE_loss = F.cross_entropy(input=reps, target=targets, reduction="mean")
                loss = CE_loss + prompt_reduce_sim_loss
                losses.append(loss.item())
                loss.backward()

                # params update
                torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                optimizer.step()

                # display
                td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled)


        for e_id in range(args.prompt_pool_epochs):
            train_data(data_loader, f"train_prompt_pool_epoch_{e_id + 1}", e_id)

    @torch.no_grad()
    def sample_memorized_data(self, args, encoder, prompt_pool, relation_data, name, task_id):
        encoder.eval()
        data_loader = get_data_loader(args, relation_data, shuffle=False)
        td = tqdm(data_loader, desc=name)

        # output dict
        out = {}

        # x_data
        x_key = []
        x_encoded = []

        for step, (labels, tokens, _) in enumerate(td):
            try:
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                x_key.append(encoder(tokens)["x_encoded"])
                x_encoded.append(encoder(tokens, prompt_pool, x_key[-1])["x_encoded"])
            except:
                continue

        x_key = torch.cat(x_key, dim=0)
        x_encoded = torch.cat(x_encoded, dim=0)

        key_mixture = GaussianMixture(n_components=args.gmm_num_components, random_state=args.seed).fit(x_key.cpu().detach().numpy())
        encoded_mixture = GaussianMixture(n_components=args.gmm_num_components, random_state=args.seed).fit(x_encoded.cpu().detach().numpy())

        if args.gmm_num_components == 1:
            key_mixture.weights_[0] = 1.0
            encoded_mixture.weights_[0] = 1.0

        out["replay_key"] = key_mixture
        out["replay"] = encoded_mixture
        return out

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, classifier, prompted_classifier, test_data, name, task_id):
        # models evaluation mode
        encoder.eval()
        classifier.eval()

        # data loader for test set
        data_loader = get_data_loader(args, test_data, batch_size=1, shuffle=False)

        # tqdm
        td = tqdm(data_loader, desc=name)

        # initialization
        sampled = 0
        total_hits = np.zeros(4)

        # testing
        for step, (labels, tokens, _) in enumerate(td):
            try:
                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                # encoder forward
                encoder_out = encoder(tokens)

                # prediction
                reps = classifier(encoder_out["x_encoded"])
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)

                # accuracy_0
                total_hits[0] += (pred == targets).float().sum().data.cpu().numpy().item()

                # pool_ids
                pool_ids = [self.id2taskid[int(x)] for x in pred]
                for i, pool_id in enumerate(pool_ids):
                    total_hits[1] += pool_id == self.id2taskid[int(labels[i])]

                # get pools
                prompt_pools = [self.prompt_pools[x] for x in pool_ids]

                # prompted encoder forward
                prompted_encoder_out = encoder(tokens, None, encoder_out["x_encoded"], prompt_pools)

                # prediction
                reps = prompted_classifier(prompted_encoder_out["x_encoded"])
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)

                # accuracy_2
                total_hits[2] += (pred == targets).float().sum().data.cpu().numpy().item()

                # pool_ids
                pool_ids = [self.id2taskid[int(x)] for x in labels]

                # get pools
                prompt_pools = [self.prompt_pools[x] for x in pool_ids]

                # prompted encoder forward
                prompted_encoder_out = encoder(tokens, None, encoder_out["x_encoded"], prompt_pools)

                # prediction
                reps = prompted_classifier(prompted_encoder_out["x_encoded"])
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)

                # accuracy_3
                total_hits[3] += (pred == targets).float().sum().data.cpu().numpy().item()

                # display
                td.set_postfix(acc=np.round(total_hits / sampled, 3))
            except:
                sampled -= len(labels)
                continue
        
        return total_hits / sampled

    def train(self, args):
        device = args.device

        # initialize test results list
        test_cur = []
        test_total = []

        # replayed data
        self.replayed_key = [[] for e_id in range(args.replay_epochs)]
        self.replayed_data = [[] for e_id in range(args.replay_epochs)]

        # sampler
        sampler = data_sampler(args=args, seed=args.seed)
        self.rel2id = sampler.rel2id
        self.id2rel = sampler.id2rel

        # convert
        self.id2taskid = {}

        # model
        encoder = BertRelationEncoder(config=args).to(args.device)
        classifier = Classifier(args=args).to(args.device)

        # pools
        self.prompt_pools = []

        # initialize memory
        self.memorized_samples = {}

        # load data and start computation
        all_train_tasks = []
        all_tasks = []
        seen_data = {}

        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            print("=" * 100)
            print(f"task={steps+1}")
            print(f"current relations={current_relations}")

            self.steps = steps
            self.not_seen_rel_ids = [rel_id for rel_id in range(args.num_tasks * args.rel_per_task) if rel_id not in [self.rel2id[relation] for relation in seen_relations]]

            # initialize
            cur_training_data = []
            cur_test_data = []
            for i, relation in enumerate(current_relations):
                cur_training_data += training_data[relation]
                seen_data[relation] = training_data[relation]
                cur_test_data += test_data[relation]

                rel_id = self.rel2id[relation]
                self.id2taskid[rel_id] = steps

            # train encoder
            if steps == 0:
                self.train_encoder(args, encoder, cur_training_data, task_id=steps)

            # new prompt pool
            new_prompt_pool = Prompt(args).to(device)
            # if steps > 0:  # Initialize new prompt pool with the previous prompt pool
            #     with torch.no_grad():
            #         self.prompt_pools[-1].prompt.grad.zero_()
            #         self.prompt_pools[-1].prompt_key.grad.zero_()
            #         new_prompt_pool.prompt = self.prompt_pools[-1].prompt.detach().clone()
            #         new_prompt_pool.prompt_key = self.prompt_pools[-1].prompt_key.detach().clone()


            self.prompt_pools.append(new_prompt_pool)

            self.train_prompt_pool(args, encoder, self.prompt_pools[-1], cur_training_data, task_id=steps)

            # memory
            for i, relation in enumerate(current_relations):
                self.memorized_samples[sampler.rel2id[relation]] = self.sample_memorized_data(args, encoder, self.prompt_pools[steps], training_data[relation], f"sampling_relation_{i+1}={relation}", steps)

            # replay data for classifier
            for relation in current_relations:
                print(f"replaying data {relation}")
                rel_id = self.rel2id[relation]
                replay_data = self.memorized_samples[rel_id]["replay"].sample(args.replay_epochs * args.replay_s_e_e)[0].astype("float32")
                for e_id in range(args.replay_epochs):
                    for x_encoded in replay_data[e_id * args.replay_s_e_e : (e_id + 1) * args.replay_s_e_e]:
                        self.replayed_data[e_id].append({"relation": rel_id, "tokens": x_encoded})

            for relation in current_relations:
                print(f"replaying key {relation}")
                rel_id = self.rel2id[relation]
                replay_key = self.memorized_samples[rel_id]["replay_key"].sample(args.replay_epochs * args.replay_s_e_e)[0].astype("float32")
                for e_id in range(args.replay_epochs):
                    for x_encoded in replay_key[e_id * args.replay_s_e_e : (e_id + 1) * args.replay_s_e_e]:
                        self.replayed_key[e_id].append({"relation": rel_id, "tokens": x_encoded})

            # all
            all_train_tasks.append(cur_training_data)
            all_tasks.append(cur_test_data)

            # evaluates
            need_evaluates = list(range(1, 11))
            if steps + 1 in need_evaluates:
                # classifier
                classifier = Classifier(args=args).to(args.device)
                swag_classifier = SWAG(Classifier, no_cov_mat=not (args.cov_mat), max_num_models=args.max_num_models, args=args)

                # classifier
                prompted_classifier = Classifier(args=args).to(args.device)
                swag_prompted_classifier = SWAG(Classifier, no_cov_mat=not (args.cov_mat), max_num_models=args.max_num_models, args=args)

                # train
                self.train_classifier(args, classifier, swag_classifier, self.replayed_key, "train_classifier_epoch_")
                self.train_classifier(args, prompted_classifier, swag_prompted_classifier, self.replayed_data, "train_prompted_classifier_epoch_")

                # prediction
                print("===NON-SWAG===")
                results = []
                for i, i_th_test_data in enumerate(all_tasks):
                    results.append([len(i_th_test_data), self.evaluate_strict_model(args, encoder, classifier, prompted_classifier, i_th_test_data, f"test_task_{i+1}", steps)])
                cur_acc = results[-1][1]
                total_acc = sum([result[0] * result[1] for result in results]) / sum([result[0] for result in results])
                print(f"current test accuracy: {cur_acc}")
                print(f"history test accuracy: {total_acc}")
                test_cur.append(cur_acc)
                test_total.append(total_acc)

                print("===SWAG===")
                results = []
                for i, i_th_test_data in enumerate(all_tasks):
                    results.append([len(i_th_test_data), self.evaluate_strict_model(args, encoder, swag_classifier, swag_prompted_classifier, i_th_test_data, f"test_task_{i+1}", steps)])
                cur_acc = results[-1][1]
                total_acc = sum([result[0] * result[1] for result in results]) / sum([result[0] for result in results])
                print(f"current test accuracy: {cur_acc}")
                print(f"history test accuracy: {total_acc}")
                test_cur.append(cur_acc)
                test_total.append(total_acc)

                acc_sum =[]
                print("===UNTIL-NOW==")
                print("accuracies:")
                for x in test_cur:
                    print(x)
                print("arverages:")
                for x in test_total:
                    print(x)
                    acc_sum.append(x)
                    
                results.append({
                    "task": steps,
                    "results": list(acc_sum),
                })
                
                if not os.path.exists(f"./results/{args.seed}_coda_prompt"):
                    os.makedirs(f"./results/{args.seed}_coda_prompt")  
                                
                with open(f"./results/{args.seed}_coda_prompt/task_{steps}.pickle", "wb") as file:
                    pickle.dump(results, file)


        del self.memorized_samples, self.prompt_pools, all_train_tasks, all_tasks, seen_data, results, encoder, self.id2taskid, sampler, self.replayed_data, self.replayed_key, test_cur, test_total
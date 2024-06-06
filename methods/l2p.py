import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig



def mean_pooling(hidden_states, attention_mask):
    pooled_output = torch.sum(
        hidden_states * attention_mask.unsqueeze(-1), dim=1
    ) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
    return pooled_output


class L2P(nn.Module):
    def __init__(self, args, prompt_pool=None):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.bert_path)
        self.model = AutoModel.from_pretrained(
            args.bert_path, config=self.config
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if prompt_pool != None:
            self.pool = PromptPool(args, self.device)
        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.num_old_labels = 0
        self.num_labels = 0

        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = args.encoder_output_size // self.config.num_attention_heads

        self.model.resize_token_embeddings(args.vocab_size + args.marker_size)
        self.frozen = False
        
        
    def get_prompt_pool(self):
        return self.pool

    def set_prompt_pool(self, prompt_pool):
        self.pool = prompt_pool

    def set_frozen_encoder(self, frozen=True):
        if frozen == True:
            self.frozen = True
            for param in self.model.parameters():
                param.requires_grad = False
                

    def get_concat_entities(self, input_ids):
        outputs = self.model(
            input_ids,
        )
        # Trích xuất token e11 và e21
        e11 = []
        e21 = []

        for i in range(input_ids.size()[0]):
            try:
                tokens = input_ids[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])
            except:
                print(input_ids[i])

        tokens_output = outputs.last_hidden_state  # Token embeddings từ mạng encoder

        logits = []

        for i in range(len(e11)):
            instance_output = torch.index_select(
                tokens_output, 0, torch.tensor(i).cuda()
            )
            instance_output = torch.index_select(
                instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda()
            )
            logits.append(instance_output)

        logits = torch.cat(logits, dim=0)
        logits = logits.view(logits.shape[0], -1)
        return logits
        
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        prompt_pool=None,
        use_custom_prompt=False,
        use_prompt=True,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if use_prompt == True:
            querys = self.get_concat_entities(input_ids)
            

            if use_custom_prompt==False:
                prompts, similarity, loss_similar = self.pool(querys)
            else:
                prompts, similarity, loss_similar = prompt_pool(querys)
            
            bs, topk, psl, hs = prompts.size()
            if self.args.prompt_mode == "prompt":
                prompts = prompts.view(bs, topk * psl, hs)
                raw_embedding = self.model.embeddings(
                    input_ids, position_ids, token_type_ids
                )
                inputs_embeds = torch.cat([prompts, raw_embedding], dim=1)
            elif self.args.prompt_mode == "prefix":
                past_key_values = prompts.view(
                    bs, topk * psl, self.n_layer * 2, self.n_head, self.n_embd
                )
                past_key_values = self.dropout(past_key_values)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
                inputs_embeds = self.model.embeddings(
                    input_ids, position_ids, token_type_ids
                )
            prompt_attention_mask = torch.ones(
                bs, topk * psl, dtype=torch.long, device=attention_mask.device
            )
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

            if prompt_pool == None and self.pool == None:
                outputs = self.model(
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                outputs = self.model(
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    past_key_values=past_key_values,
                )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
            )
            

        # Trích xuất token e11 và e21
        e11 = []
        e21 = []

        for i in range(input_ids.size()[0]):
            try:
                tokens = input_ids[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])
            except:
                print(input_ids[i])

        tokens_output = outputs.last_hidden_state  # Token embeddings từ mạng encoder

        logits = []

        for i in range(len(e11)):
            instance_output = torch.index_select(
                tokens_output, 0, torch.tensor(i).cuda()
            )
            instance_output = torch.index_select(
                instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda()
            )
            logits.append(instance_output)

        logits = torch.cat(logits, dim=0)
        logits = logits.view(logits.shape[0], -1)

        try:
            return outputs, logits, loss_similar
        except:
            return outputs, logits




def get_prompts_data(args, device, prompt_init=None):
    if prompt_init is None:
        prompt_init = torch.randn

    if args.prompt_mode == "prompt":
        new_prompt_data = prompt_init(args.prompt_length, args.hidden_size, device=device)
    elif args.prompt_mode == "prefix":
        # simple prefix module
        new_prompt_data = prompt_init(
            args.prompt_length,
            args.num_hidden_layers * args.hidden_size * 2,
            device=device,
        )
    else:
        raise NotImplementedError
    return new_prompt_data



class PromptPool(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.pool_size = args.prompt_pool_size
        self.topk = args.prompt_top_k
        self.prompt_length = args.prompt_length
        self.hidden_size = args.hidden_size

        self.query_mode = args.query_mode  # ["cosine", "euclidean"]
        self.prompt_mode = args.prompt_mode  # ["prompt", "prefix"]
        self.device = device

        prompts_key_init = torch.zeros  # randn, rand, zeros
        prompts_init = torch.rand  # randn, rand, zeros

        prompts_key = prompts_key_init(self.pool_size, self.hidden_size * 2, device=self.device)
        self.prompts_key = nn.Parameter(prompts_key, requires_grad=True)

        new_prompt_data = torch.stack([get_prompts_data(args, self.device, prompts_init) for _ in range(self.pool_size)])
        self.prompts = nn.Parameter(new_prompt_data, requires_grad=True)

    def forward(self, querys, x_key=None):
        # Calculate the loss
        query_norm = F.normalize(querys, dim=-1)
        prompts_key_norm = F.normalize(self.prompts_key, dim=-1)
        # querys: [bs, hidden_size]
        if self.query_mode == "cosine":
            # [bs, pool_size]
            similarity = F.cosine_similarity(querys.unsqueeze(1), self.prompts_key.unsqueeze(0), dim=-1)
        elif self.query_mode == "euclidean":
            # negative euclidean distance for sorting
            similarity = -torch.cdist(querys, self.prompts_key, p=2)
        else:
            raise NotImplementedError
        
        _, indices = torch.topk(similarity, self.topk, dim=-1)  # [bs, topk]
        similarity = torch.gather(similarity, dim=-1, index=indices)  # [bs, topk]

        loss = -torch.sum(query_norm.unsqueeze(1) * prompts_key_norm.unsqueeze(0), dim=-1).mean()

        return self.prompts[indices], similarity, loss

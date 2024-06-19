import transformers
import torch
from dataclasses import dataclass


IGNORE_INDEX = -100


@dataclass
class DataCollatorWithMaskForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, batch):
        input_ids = []
        labels = []
        attention_masks = []

        for item_dict in batch:
            input_ids.append(torch.tensor(item_dict["input_ids"]))
            attention_masks.append(torch.tensor(item_dict["attention_mask"]))
            label = torch.tensor(item_dict["labels"])
            #label[:-1] = IGNORE_INDEX
            labels.append(label)

        input_ids = torch.vstack(input_ids)
        attention_masks = torch.vstack(attention_masks)
        labels = torch.vstack(labels)

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict

@dataclass
class DistillDataCollatorWithMaskForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, batch):
        input_ids = []
        labels = []
        attention_masks = []
        logits = []

        for item_dict in batch:
            input_ids.append(torch.tensor(item_dict["input_ids"]))
            attention_masks.append(torch.tensor(item_dict["attention_mask"]))
            label = torch.tensor(item_dict["labels"])
            #label[:-1] = IGNORE_INDEX
            labels.append(label)
            logits.append(torch.tensor(item_dict['logits']).unsqueeze(0))

        input_ids = torch.vstack(input_ids)
        attention_masks = torch.vstack(attention_masks)
        labels = torch.vstack(labels)
        logits = torch.vstack(logits)
            
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'logits': logits
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

@dataclass
class DistillDataCollatorSeq2Seq(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, batch):
        input_ids = []
        labels = []
        attention_masks = []
        logits = []

        for item_dict in batch:
            input_ids.append(torch.tensor(item_dict["input_ids"]))
            attention_masks.append(torch.tensor(item_dict["attention_mask"]))
            label = torch.tensor(item_dict["labels"])
            labels.append(label)
            logits.append(torch.tensor(item_dict['logits']))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks =torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels =torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        logits =torch.nn.utils.rnn.pad_sequence(
            logits, batch_first=True, padding_value=0
        )
            
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'logits': logits
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

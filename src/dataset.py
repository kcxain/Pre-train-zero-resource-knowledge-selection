import logging
import pprint
import torch
import json
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
wow_test_data_path = "./data/wow/valid_topic_split_WoW_one_file_retrieve.json"
wow_train_data_path = "./data/wow/train_WoW_.json"
wow_doc_path = "./data/wow/Doc_WoW_.json"


class WowDataset(Dataset):
    def __init__(self, args, tokenizer, split_type,
                 test_path=wow_test_data_path,
                 train_path=wow_train_data_path,
                 doc_path=wow_doc_path):
        self.args = args
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.test_path = test_path
        self.train_path = train_path
        self.doc_path = wow_doc_path

        self.max_query_length = self.args.max_query_length
        self.max_knowledge_length = self.args.max_knowledge_length
        self.max_response_length = self.args.max_response_length

        self._create_examples()

    def _create_examples(self):
        """ Creating examples for model training and evaluation """
        self.examples = []
        if self.split_type == "train":
            self.examples = self.process_train_data_from_wow(self.train_path)
        else:
            # TODO
            self.examples = self.process_eval_data_from_wow(self.test_path)

    def __getitem__(self, index):
        example = self.examples[index]
        instance = {
            "id": example["id"],
        }
        if self.split_type == "train":
            query_input_ids, query_attention_mask, query_token_type_ids = \
                self.build_build_input_from_segments(example["query"], self.max_query_length)
            knowledge_input_ids, knowledge_attention_mask, knowledge_token_type_ids = \
                self.build_build_input_from_segments(example["knowledge"], self.max_knowledge_length)
            response_input_ids, response_attention_mask, response_token_type_ids = \
                self.build_build_input_from_segments(example["response"], self.max_response_length)

            instance["query_input_ids"] = query_input_ids
            instance["query_attention_mask"] = query_attention_mask
            instance["query_token_type_ids"] = query_token_type_ids

            instance["knowledge_input_ids"] = knowledge_input_ids
            instance["knowledge_attention_mask"] = knowledge_attention_mask
            instance["knowledge_token_type_ids"] = knowledge_token_type_ids

            instance["response_input_ids"] = response_input_ids
            instance["response_attention_mask"] = response_attention_mask
            instance["response_token_type_ids"] = response_token_type_ids

            return instance

        else:
            query_input_ids, query_attention_mask, query_token_type_ids = \
                self.build_build_input_from_segments(example["query"], self.max_query_length)
            knowledge_input_ids, knowledge_attention_mask, knowledge_token_type_ids = \
                self.build_build_input_from_segments(example["knowledge"], self.max_knowledge_length)
            response_input_ids, response_attention_mask, response_token_type_ids = \
                self.build_build_input_from_segments(example["response"], self.max_response_length)

            ne_knowledge_input_ids_list = []
            ne_knowledge_attention_mask_list = []
            ne_knowledge_token_type_ids_list = []
            for ne in example["negative_knowledge"]:
                ne_knowledge_input_ids, ne_knowledge_attention_mask, ne_knowledge_token_type_ids = \
                self.build_build_input_from_segments(ne, self.max_knowledge_length)
                ne_knowledge_input_ids_list.append(ne_knowledge_input_ids)
                ne_knowledge_attention_mask_list.append(ne_knowledge_attention_mask)
                ne_knowledge_token_type_ids_list.append(ne_knowledge_token_type_ids)
            
            ne_response_input_ids_list = []
            ne_response_attention_mask_list = []
            ne_response_token_type_ids_list = []
            for ne in example["negative_response"]:
                ne_response_input_ids, ne_response_attention_mask, ne_response_token_type_ids = \
                self.build_build_input_from_segments(ne, self.max_response_length)
                ne_response_input_ids_list.append(ne_response_input_ids)
                ne_response_attention_mask_list.append(ne_response_attention_mask)
                ne_response_token_type_ids_list.append(ne_response_token_type_ids)
            
            instance["query_input_ids"] = query_input_ids
            instance["query_attention_mask"] = query_attention_mask
            instance["query_token_type_ids"] = query_token_type_ids

            instance["knowledge_input_ids"] = knowledge_input_ids
            instance["knowledge_attention_mask"] = knowledge_attention_mask
            instance["knowledge_token_type_ids"] = knowledge_token_type_ids

            instance["response_input_ids"] = response_input_ids
            instance["response_attention_mask"] = response_attention_mask
            instance["response_token_type_ids"] = response_token_type_ids

            instance["ne_knowledge_input_ids_list"] = ne_knowledge_input_ids_list
            instance["ne_knowledge_attention_mask_list"] = ne_knowledge_attention_mask_list
            instance["ne_knowledge_token_type_ids_list"] = ne_knowledge_token_type_ids_list

            instance["ne_response_input_ids_list"] = ne_response_input_ids_list
            instance["ne_response_attention_mask_list"] = ne_response_attention_mask_list
            instance["ne_response_token_type_ids_list"] = ne_response_token_type_ids_list

            return instance


    def build_build_input_from_segments(self, sentence, max_length):
        sentence_tokens = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        input_ids = sentence_tokens.pop("input_ids")
        attention_mask = sentence_tokens.pop("attention_mask")
        token_type_ids = sentence_tokens.pop("token_type_ids")

        return input_ids, attention_mask, token_type_ids

    def collate_fn(self, batch):
        if self.split_type == "train":
            query_input_ids = torch.tensor([ins["query_input_ids"] for ins in batch]).to(self.args.device)
            knowledge_input_ids = torch.tensor([ins["knowledge_input_ids"] for ins in batch]).to(self.args.device)
            response_input_ids = torch.tensor([ins["response_input_ids"] for ins in batch]).to(self.args.device)

            query_token_type_ids = torch.tensor([ins["query_token_type_ids"] for ins in batch]).to(self.args.device)
            knowledge_token_type_ids = torch.tensor([ins["knowledge_token_type_ids"] for ins in batch]).to(self.args.device)
            response_token_type_ids = torch.tensor([ins["response_token_type_ids"] for ins in batch]).to(self.args.device)

            query_attention_mask = torch.tensor([ins["query_attention_mask"] for ins in batch]).to(self.args.device)
            knowledge_attention_mask = torch.tensor([ins["knowledge_attention_mask"] for ins in batch]).to(self.args.device)
            response_attention_mask = torch.tensor([ins["response_attention_mask"] for ins in batch]).to(self.args.device)

            input_ids = {
                "query_input_ids": query_input_ids,
                "knowledge_input_ids": knowledge_input_ids,
                "response_input_ids": response_input_ids
            }

            attention_mask = {
                "query_attention_mask": query_attention_mask,
                "knowledge_attention_mask": knowledge_attention_mask,
                "response_attention_mask": response_attention_mask
            }

            token_type_ids = {
                "query_token_type_ids": query_token_type_ids,
                "knowledge_token_type_ids": knowledge_token_type_ids,
                "response_token_type_ids": response_token_type_ids
            }

            return input_ids, attention_mask, token_type_ids
        
        else:
            query_input_ids = torch.tensor([ins["query_input_ids"] for ins in batch]).to(self.args.device)
            knowledge_input_ids = torch.tensor([ins["knowledge_input_ids"] for ins in batch]).to(self.args.device)
            response_input_ids = torch.tensor([ins["response_input_ids"] for ins in batch]).to(self.args.device)

            query_token_type_ids = torch.tensor([ins["query_token_type_ids"] for ins in batch]).to(self.args.device)
            knowledge_token_type_ids = torch.tensor([ins["knowledge_token_type_ids"] for ins in batch]).to(self.args.device)
            response_token_type_ids = torch.tensor([ins["response_token_type_ids"] for ins in batch]).to(self.args.device)

            query_attention_mask = torch.tensor([ins["query_attention_mask"] for ins in batch]).to(self.args.device)
            knowledge_attention_mask = torch.tensor([ins["knowledge_attention_mask"] for ins in batch]).to(self.args.device)
            response_attention_mask = torch.tensor([ins["response_attention_mask"] for ins in batch]).to(self.args.device)

            ne_knowledge_input_ids_list = torch.tensor([ins["ne_knowledge_input_ids_list"] for ins in batch]).to(self.args.device)
            ne_knowledge_attention_mask_list = torch.tensor([ins["ne_knowledge_attention_mask_list"] for ins in batch]).to(self.args.device)
            ne_knowledge_token_type_ids_list = torch.tensor([ins["ne_knowledge_token_type_ids_list"] for ins in batch]).to(self.args.device)

            ne_response_input_ids_list = torch.tensor([ins["ne_response_input_ids_list"] for ins in batch]).to(self.args.device)
            ne_response_attention_mask_list = torch.tensor([ins["ne_response_attention_mask_list"] for ins in batch]).to(self.args.device)
            ne_response_token_type_ids_list = torch.tensor([ins["ne_response_token_type_ids_list"] for ins in batch]).to(self.args.device)
            
            input_ids = {
                "query_input_ids": query_input_ids,
                "knowledge_input_ids": knowledge_input_ids,
                "response_input_ids": response_input_ids,
                "ne_knowledge_input_ids_list": ne_knowledge_input_ids_list,
                "ne_response_input_ids_list": ne_response_input_ids_list
            }

            attention_mask = {
                "query_attention_mask": query_attention_mask,
                "knowledge_attention_mask": knowledge_attention_mask,
                "response_attention_mask": response_attention_mask,
                "ne_knowledge_attention_mask_list": ne_knowledge_attention_mask_list,
                "ne_response_attention_mask_list": ne_response_attention_mask_list
            }

            token_type_ids = {
                "query_token_type_ids": query_token_type_ids,
                "knowledge_token_type_ids": knowledge_token_type_ids,
                "response_token_type_ids": response_token_type_ids,
                "ne_knowledge_token_type_ids_list": ne_knowledge_token_type_ids_list,
                "ne_response_token_type_ids_list": ne_response_token_type_ids_list
            }

            return input_ids, attention_mask, token_type_ids

    def process_train_data_from_wow(
            self,
            file_path=wow_train_data_path,
            doc_path=wow_doc_path,
    ):
        with open(file_path, "r") as f:
            raw_test_data = json.load(f)

        with open(doc_path, 'r') as f:
            doc_data = json.load(f)["doc_data"]

        dial_data = raw_test_data["dial_data"]
        result = []

        cnt = 0
        for domain, d_doc_dials in dial_data.items():
            for doc_id, dials in d_doc_dials.items():
                dials_tq = tqdm(dials)
                for dial in dials_tq:

                    all_prev_utterances = []
                    for idx, turn in enumerate(dial["turns"]):
                        all_prev_utterances.append(turn["utterance"])
                        if turn["role"] == "agent":
                            continue
                        if idx + 1 < len(dial["turns"]):
                            if dial["turns"][idx + 1]["role"] == "agent":
                                turn_to_predict = dial["turns"][idx + 1]
                            else:
                                continue
                        else:
                            continue

                        query = all_prev_utterances[-1]
                        query = " ".join(query.split()[:self.max_query_length])

                        id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"])
                        spans = doc_data[domain][turn_to_predict["doc_id"]]["spans"]

                        correct_knowledge = []
                        for ref in turn_to_predict["references"]:
                            correct_knowledge.append(int(ref["sp_id"]))

                        correct_knowledge = correct_knowledge[0]

                        response = turn_to_predict["utterance"]
                        knowledge = " ".join(
                            spans[str(correct_knowledge)]["text_sp"].split()[-self.max_knowledge_length:])
                        result.append(
                            {
                                "query": query,
                                "knowledge": knowledge,
                                "response": response,
                                "id": id_,
                            }
                        )
                        cnt += 1
        return result

    def process_eval_data_from_wow(
            self,
            file_path=wow_test_data_path,
            doc_path=wow_doc_path,
    ):
        with open(file_path, "r") as f:
            data = json.load(f)
        with open(doc_path, 'r') as f:
            doc_data = json.load(f)["doc_data"]

        dial_data = data["dial_data"]
        result = []
        cnt2 = -1
        idx = 0
        nk = 0
        nr = 0
        for domain, d_doc_dials in dial_data.items():
            for doc_id, dials in d_doc_dials.items():
                for dial in dials:
                    cnt2 += 1
                    if cnt2 >=100:
                        break
                    if cnt2 == 345:
                        continue
                    all_prev_utterances = []
                    for idx, turn in enumerate(dial["turns"]):
                        all_prev_utterances.append(turn["utterance"])
                        if turn["role"] == "agent":
                            continue
                        if idx + 1 < len(dial["turns"]):
                            if dial["turns"][idx + 1]["role"] == "agent":
                                turn_to_predict = dial["turns"][idx + 1]
                            else:
                                continue
                        else:
                            continue
                        reverse_all_prev_utterances = list(reversed(all_prev_utterances))

                        query = " ".join(reverse_all_prev_utterances[0].split()[:self.max_query_length])
                        id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"])

                        spans = doc_data[domain][turn_to_predict["doc_id"]]["spans"]
                        correct_knowledge = []
                        for ref in turn_to_predict["references"]:
                            correct_knowledge.append(int(ref["sp_id"]))

                        negative_knowledge = []
                        for i in range(1, len(spans.keys()) + 1):
                            if i not in correct_knowledge:
                                cur_ne = spans[str(i)]["text_sp"].split()[-self.max_knowledge_length:]
                                negative_knowledge.append(" ".join(cur_ne))
                        correct_knowledge = correct_knowledge[0]

                        knowledge = " ".join(
                            spans[str(correct_knowledge)]["text_sp"].split()[-self.max_knowledge_length:])

                        correct_response = turn_to_predict["utterance"]
                        response_spans = turn_to_predict["spans"]

                        negative_response = []
                        for response in response_spans.values():
                            if response["text_sp"] != correct_response:
                                negative_response.append(" ".join(response["text_sp"].split()[:self.max_response_length]))

                        if len(negative_knowledge)==0 or len(negative_response)==0:
                            continue
                        if(len(negative_knowledge) != 99):
                            print(len(negative_knowledge))
                        result.append(
                            {
                                "knowledge": knowledge,
                                "negative_knowledge": negative_knowledge,
                                "query": query,
                                "response": " ".join(correct_response.split()[:self.max_response_length]),
                                "negative_response": negative_response,
                                "id": id_,
                            }
                        )
        return result
    
    def __len__(self):
        return len(self.examples)


    
    
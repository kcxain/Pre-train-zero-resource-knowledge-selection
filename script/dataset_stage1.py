from tqdm import tqdm
import json
import random
from datasets import Dataset

query_file_path_train = "data/qa/queries.train.tsv"
query_file_path_dev = "data/qa/queries.dev.tsv"
query_file_path_eval = "data/qa/queries.eval.tsv"

collection_file_path = "data/qa/collection.tsv"

qa_file_path = "data/qa/qidpidtriples.train.full.2.tsv"
qrel_file_path = "data/qa/qrels.train.tsv"
qrel_dev_file_path = "data/qa/qrels.dev.tsv"

query_response_macth_file_path = "data/qa/query_response_match.json"
response_query_macth_file_path = "data/qa/response_query_match.json"
query_response_macth_dev_file_path = "data/qa/query_response_match_dev.json"
response_query_macth_dev_file_path = "data/qa/response_query_match_dev.json"

pos_and_neg_train_file_path = "data/qa/pos_neg_train.json"
pos_and_neg_dev_file_path = "data/qa/pos_neg_dev.json"
pos_and_neg_train_file_negative_response = "data/qa/pos_neg_response_train.json"
pos_and_neg_dev_file_negative_response = "data/qa/pos_neg_response_dev.json"

wow_test_data_path = "data/wow/test_topic_split_WoW_.json"
wow_train_data_path = "data/wow/train_WoW_.json"
wow_doc_path = "data/wow/Doc_WoW_.json"

NEGATIVE_SAMPLES = 6
MAX_KNOWLEDGE_LEN = 128
MAX_QUERY_LEN = 60
MAX_RESPONSE_LEN = 40


def load_wow_dataset(
        train_file_path=wow_train_data_path,
        dev_file_path=wow_test_data_path,
):
    dataset = {"train": Dataset.from_dict(process_data_from_wow(file_path=train_file_path)),
               "validation": Dataset.from_dict(process_data_from_wow(file_path=dev_file_path)),
               "test": Dataset.from_dict(process_data_from_wow(file_path=dev_file_path))}
    return dataset


def load_query(
        split="train",
):
    if split == "train":
        file_path = query_file_path_train
    elif split == "dev":
        file_path = query_file_path_dev
    else:
        file_path = query_file_path_eval

    dataset = {}
    with open(file_path, "r") as f:
        for line in f:
            items = line.split("\t")
            if len(items) != 2:
                continue
            dataset[items[0]] = items[1]
    return dataset


def load_response(
        file_path=collection_file_path,
):
    dataset = {}
    with open(file_path, "r") as f:
        for line in tqdm(f):
            items = line.split("\t")
            if len(items) != 2:
                continue
            dataset[items[0]] = items[1]

    return dataset


def load_pq_pairs(
        file_path=qa_file_path,
):
    dataset = {}
    with open(file_path, "r") as f:
        for line in f:
            items = line.split("\t")
            if len(items) != 3:
                continue
            dataset[items[0]] = items[1]
    return dataset


def qa_match_check():
    query_response_dataset = {}
    response_query_dataset = {}
    qr_statistic = {}
    rq_statistic = {}
    with open(qa_file_path, "r") as f:
        for line in tqdm(f):
            items = line[:-1].split("\t")
            if len(items) != 3:
                continue
            if items[0] not in query_response_dataset.keys():
                query_response_dataset[items[0]] = [items[1]]
            else:
                if items[1] not in query_response_dataset[items[0]]:
                    query_response_dataset[items[0]].append(items[1])
            if items[1] not in response_query_dataset.keys():
                response_query_dataset[items[1]] = [items[0]]
            else:
                if items[0] not in response_query_dataset[items[1]]:
                    response_query_dataset[items[1]].append(items[0])
    with open(response_query_macth_file_path, "w") as f:
        json.dump(response_query_dataset, f)
    with open(query_response_macth_file_path, "w") as f:
        json.dump(query_response_dataset, f)

    print("total used query in training dataset:" + str(len(query_response_dataset.keys())))
    for value in query_response_dataset.values():
        length = len(value)
        if length not in qr_statistic.keys():
            qr_statistic[length] = 1
        else:
            qr_statistic[length] += 1
    print("the statistic of query match response:")
    print(qr_statistic)
    print("total used response in training dataset:" + str(len(response_query_dataset.keys())))
    for value in response_query_dataset.values():
        length = len(value)
        if length not in rq_statistic.keys():
            rq_statistic[length] = 1
        else:
            rq_statistic[length] += 1
    print("the statistic of response match query:")
    print(rq_statistic)


def qa_macth_from_qrel():
    query_response_dataset = {}
    response_query_dataset = {}
    qr_statistic = {}
    rq_statistic = {}
    with open(qrel_dev_file_path, "r") as f:
        for line in tqdm(f):
            items = line[:-1].split("\t")
            if len(items) != 4:
                continue
            if items[0] not in query_response_dataset.keys():
                query_response_dataset[items[0]] = [items[2]]
            else:
                if items[2] not in query_response_dataset[items[0]]:
                    query_response_dataset[items[0]].append(items[2])
            if items[2] not in response_query_dataset.keys():
                response_query_dataset[items[2]] = [items[0]]
            else:
                if items[0] not in response_query_dataset[items[2]]:
                    response_query_dataset[items[2]].append(items[0])

        with open(response_query_macth_dev_file_path, "w") as f:
            json.dump(response_query_dataset, f)
        with open(query_response_macth_dev_file_path, "w") as f:
            json.dump(query_response_dataset, f)

        print("total used query in training dataset:" + str(len(query_response_dataset.keys())))
        for value in query_response_dataset.values():
            length = len(value)
            if length not in qr_statistic.keys():
                qr_statistic[length] = 1
            else:
                qr_statistic[length] += 1
        print("the statistic of query match response:")
        print(qr_statistic)
        print("total used response in training dataset:" + str(len(response_query_dataset.keys())))
        for value in response_query_dataset.values():
            length = len(value)
            if length not in rq_statistic.keys():
                rq_statistic[length] = 1
            else:
                rq_statistic[length] += 1
        print("the statistic of response match query:")
        print(rq_statistic)


def get_dataset(
        file_path=response_query_macth_dev_file_path,
        query_split="dev"
):
    query_data = load_query(query_split)
    query_list = query_data.keys()
    dataset = {}
    with open(file_path, "r") as f:
        rq_match = json.load(f)
        for key, value in tqdm(rq_match.items()):
            if len(value) > 1:
                continue
            dataset[key] = {
                "positive": value,
                "negative": random.sample(query_list, NEGATIVE_SAMPLES)
            }
    with open(pos_and_neg_dev_file_path, "w") as f:
        json.dump(dataset, f)

    return dataset


def get_dataset_negative_response(
        file_path=response_query_macth_dev_file_path,
        query_split="dev"
):
    # response_data = load_response()
    response_list = list(range(8841823))
    dataset = {}
    with open(file_path, "r") as f:
        rq_match = json.load(f)
        for key, value in tqdm(rq_match.items()):
            if len(value) > 1:
                continue
            dataset[value[0]] = {
                "positive": [key],
                "negative": random.sample(response_list, NEGATIVE_SAMPLES)
            }
    with open(pos_and_neg_dev_file_negative_response, "w") as f:
        json.dump(dataset, f)

    return dataset


# def load_dataset(
#         train_file_path=pos_and_neg_train_file_negative_response,
#         dev_file_path=pos_and_neg_dev_file_negative_response,
# ):
#     dataset = {}
#     response_data = load_response()
#     dataset["train"] = Dataset.from_dict(process_dataset_negative_response(train_file_path, "train", response_data))
#     dataset["validation"] = Dataset.from_dict(process_dataset_negative_response(dev_file_path, "dev", response_data))
#     dataset["test"] = Dataset.from_dict(process_dataset_negative_response(dev_file_path, "dev", response_data))
#     return dataset


def process_dataset(
        file_path=pos_and_neg_train_file_path,
        split="train",
        response_data=None,
):
    query_data = load_query(split)
    datasets = {
        "knowledge": [],  ## This is quer
        "context": [],
        "label": [],
        "id": [],
    }
    with open(file_path, "r") as f:
        row_data = json.load(f)
    cnt = 0
    for key, value in tqdm(row_data.items()):
        if cnt > 500 and split == "dev":
            return datasets
        knowledge = response_data[str(key)]
        knowledge = " ".join(knowledge.split()[-MAX_KNOWLEDGE_LEN:])
        pos_context = query_data[str(value["positive"][0])]
        datasets["knowledge"].append(knowledge)
        datasets["context"].append(" ".join(pos_context.split()[:MAX_QUERY_LEN]))
        datasets["label"].append(1)
        datasets["id"].append(cnt)
        negative_cnt = 0
        for nag in value["negative"]:
            if negative_cnt >= NEGATIVE_SAMPLES:
                break
            datasets["knowledge"].append(knowledge)
            datasets["context"].append(" ".join(query_data[str(nag)].split()[:MAX_QUERY_LEN]))
            datasets["label"].append(0)
            datasets["id"].append(cnt)
            negative_cnt += 1
        cnt += 1
    return datasets


def process_dataset_negative_response(
        file_path=pos_and_neg_train_file_negative_response,
        split="train",
        response_data=None,
):
    query_data = load_query(split)
    datasets = {
        "knowledge": [],
        "context": [],
        "label": [],
        "id": [],
    }
    with open(file_path, "r") as f:
        row_data = json.load(f)
    cnt = 0
    for key, value in tqdm(row_data.items()):
        if cnt > 1000 and split == "dev":
            return datasets
        knowledge = response_data[str(value["positive"][0])]
        context = query_data[str(key)]
        context = " ".join(context.split()[:MAX_QUERY_LEN])
        datasets["knowledge"].append(" ".join(knowledge.split()[-MAX_KNOWLEDGE_LEN:]))
        datasets["context"].append(context)
        datasets["label"].append(1)
        datasets["id"].append(cnt)
        # negative_cnt = 0
        for nag in value["negative"]:
            # if negative_cnt >= NEGATIVE_SAMPLES:
            #     break
            datasets["knowledge"].append(" ".join(response_data[str(nag)].split()[-MAX_KNOWLEDGE_LEN:]))
            datasets["context"].append(context)
            datasets["label"].append(0)
            datasets["id"].append(cnt)
        cnt += 1
    return datasets


def process_dataset_stage1_joint(
        file_path=pos_and_neg_train_file_negative_response,
        split="train",
):
    query_data = load_query(split)
    response_data = load_response()
    datasets = {
        "history": [],
        "query": [],
        "response": [],
        "label": [],
        "id": [],
    }
    with open(file_path, "r") as f:
        row_data = json.load(f)
    cnt = 0
    for key, value in tqdm(row_data.items()):
        if cnt > 200000:
            return datasets
        cnt_str = "1_" + str(cnt)
        knowledge = response_data[str(value["positive"][0])]
        context = query_data[str(key)]
        context = " ".join(context.split()[:MAX_QUERY_LEN])
        datasets["history"].append(" ".join(knowledge.split()[-MAX_KNOWLEDGE_LEN:]))
        datasets["query"].append(context)
        datasets["response"].append("")
        datasets["label"].append(1)
        datasets["id"].append(cnt_str)

        for nag in value["negative"]:
            datasets["history"].append(" ".join(response_data[str(nag)].split()[-MAX_KNOWLEDGE_LEN:]))
            datasets["query"].append(context)
            datasets["response"].append("")
            datasets["label"].append(0)
            datasets["id"].append(cnt_str)
        cnt += 1
    return datasets


def process_data_from_wow(
        file_path=wow_test_data_path,
        doc_path=wow_doc_path,
):
    with open(file_path, "r") as f:
        row_test_data = json.load(f)
    with open(doc_path, 'r') as f:
        doc_data = json.load(f)["doc_data"]
    dial_data = row_test_data["dial_data"]
    result = {}
    result["knowledge"] = []
    result["context"] = []
    result["response"] = []
    result["id"] = []
    result["label"] = []
    cnt = 0
    cnt2 = -1
    for domain, d_doc_dials in dial_data.items():
        for doc_id, dials in d_doc_dials.items():
            dials_tq = tqdm(dials)
            for dial in dials_tq:
                cnt2 += 1
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
                    question = all_prev_utterances[-1]
                    question = " ".join(question.split()[:MAX_QUERY_LEN])

                    response = turn_to_predict["utterance"]
                    response = " ".join(response.split()[:MAX_RESPONSE_LEN])

                    id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"])
                    spans = doc_data[domain][turn_to_predict["doc_id"]]["spans"]

                    correct_knowledge = []
                    for ref in turn_to_predict["references"]:
                        correct_knowledge.append(int(ref["sp_id"]))
                    negative_knowledge = []
                    for i in range(1, len(spans.keys()) + 1):
                        if i not in correct_knowledge:
                            negative_knowledge.append(i)
                    correct_knowledge = correct_knowledge[0]

                    # 每个正例的负例数量都要相等
                    if len(negative_knowledge) < NEGATIVE_SAMPLES:
                        continue
                    negative_knowledge = random.sample(negative_knowledge, NEGATIVE_SAMPLES)

                    knowledge = " ".join(spans[str(correct_knowledge)]["text_sp"].split()[-MAX_KNOWLEDGE_LEN:])
                    result["query"].append(question)
                    result["knowledge"].append(knowledge)
                    result["response"].append(response)
                    result["id"].append(id_)
                    result["label"].append(1)
                    print(len(negative_knowledge))
                    try:
                        for span_id in negative_knowledge:
                            span = spans[str(span_id)]
                            knowledge = span["text_sp"].split()[-MAX_KNOWLEDGE_LEN:]
                            result["query"].append(question)
                            result["knowledge"].append(" ".join(knowledge))
                            result["response"].append(response)
                            result["id"].append(id_)
                            result["label"].append(0)
                    except:
                        print(turn_to_predict["doc_id"])
                    cnt += 1
    return result


if __name__ == "__main__":
    result = process_data_from_wow(file_path=wow_test_data_path)
    print(len(result["context"]))
    print(result["context"][:7])
    print(result["knowledge"][:7])
    print(result["response"][:7])
    print(result["id"][:7])
    print(result["label"][:7])

    # load_wow_dataset()

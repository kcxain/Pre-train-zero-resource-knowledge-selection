import json
from tqdm import tqdm
from datasets import Dataset

wow_test_data_path = "./data/wow/test_topic_split_WoW_.json"
wow_train_data_path = "./data/wow/train_WoW_.json"
wow_doc_path = "./data/wow/Doc_WoW_.json"

NEGATIVE_SAMPLES = 6
MAX_KNOWLEDGE_LEN = 128
MAX_QUERY_LEN = 60
MAX_RESPONSE = 40
MAX_CONTEXT = 256


def process_data_from_wow(
        file_path=wow_test_data_path,
        doc_path=wow_doc_path,
):
    with open(file_path, "r") as f:
        raw_test_data = json.load(f)

    with open(doc_path, 'r') as f:
        doc_data = json.load(f)["doc_data"]

    dial_data = raw_test_data["dial_data"]
    result = {}
    result["knowledge"] = []
    result["query"] = []
    result["response"] = []
    result["id"] = []
    result["label"] = []
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
                    query = " ".join(query.split()[:MAX_QUERY_LEN])

                    id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"])
                    spans = doc_data[domain][turn_to_predict["doc_id"]]["spans"]

                    correct_knowledge = []
                    for ref in turn_to_predict["references"]:
                        correct_knowledge.append(int(ref["sp_id"]))

                    correct_knowledge = correct_knowledge[0]

                    response = turn_to_predict["utterance"]
                    knowledge = " ".join(spans[str(correct_knowledge)]["text_sp"].split()[-MAX_KNOWLEDGE_LEN:])
                    result["query"].append(query)
                    result["knowledge"].append(knowledge)
                    result["response"].append(response)
                    result["id"].append(id_)
                    result["label"].append(1)

                    cnt += 1
    return result


def load_dataset(
):
    dataset = {}
    dataset["train"] = Dataset.from_dict(process_data_from_wow(file_path=wow_train_data_path))
    dataset["validation"] = Dataset.from_dict(process_data_from_wow(file_path=wow_test_data_path))
    dataset["test"] = Dataset.from_dict(process_data_from_wow(file_path=wow_test_data_path))
    return dataset


if __name__ == '__main__':
    result = process_data_from_wow(file_path=wow_train_data_path)
    for i, (q, k, r) in enumerate(zip(result["query"], result["knowledge"], result["response"])):
        if i > 10:
            break
        print(q)
        print(k)
        print(r)
        print('-------------------------------------')

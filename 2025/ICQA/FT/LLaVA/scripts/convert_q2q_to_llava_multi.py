import os
import json
import argparse
import pandas as pd
from glob import glob
from datasets import load_from_disk, load
from tqdm import tqdm

def build_prompt_chatbot(problems, sceneGraphs, is_bounding_box=False):
    examples = {}
    for idx in tqdm(range(1, len(problems)), total=len(problems)):
        entity = problems.iloc[idx]["ambiguous_entity"]
        entity_id = str(problems.iloc[idx]["entity_id"])
        image_id = str(problems.iloc[idx]["image_id"])
        q_id = problems.iloc[idx]["q_id"]
        additional_question = problems.iloc[idx]["additional_question"]
        label = problems.iloc[idx]["label"]

        if label != "O":
            continue
        
        # boxes
        target_entity = sceneGraphs[image_id]["objects"][entity_id]
        target_entity_name = target_entity["name"]
        width = sceneGraphs[image_id]["width"]
        height = sceneGraphs[image_id]["height"]

        if is_bounding_box is True:
            bounding_boxs = []
            for object_id, object_value in sceneGraphs[image_id]["objects"].items():
                if object_value["name"] == target_entity_name:
                    x = object_value["x"] / width
                    y = object_value["y"] / height
                    w = x + object_value["w"] / width
                    h = y + object_value["h"] / height
                    bounding_boxs.append(
                        f"{target_entity_name}: [{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}]"
                    )
                    if object_id == entity_id:
                        target_entity = f"{target_entity_name}: [{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}]"
            bounding_boxs_context = ",".join(bounding_boxs)

            input = (
                bounding_boxs_context
                + f"\n Target Entity: {target_entity}"
                + "\n"
                + f"Generate a sub-question to classify ambiguous entities.\n"
            )
        else:
            input = f"Entity: {entity}\nSub-Question:\n"

        output = f"Sub-Question: {additional_question}"
        input = input.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        examples[f"{idx}"] = input, output

    return examples


def convert_to_llava(args):

    base_dir = args.base_dir
    split = args.split
    bounding_box = args.bounding_box
    problems = pd.read_csv(os.path.join(base_dir, f"Q2Q_{split}.csv"), dtype={'entity_id': str, 'q_id': str, 'image_id': str})
    print(len(problems))
    if args.add_sample:
        file_paths = glob(os.path.join(base_dir, "new_sub_*.csv"))
        for file_path in file_paths:
            new_problems = pd.read_csv(file_path, dtype={'entity_id': str, 'q_id': str, 'image_id': str})
            new_problems['additional_question'] = new_problems['sub_question']
            new_problems['label'] = 'O'
            remove_columns = [col for col in new_problems.columns if col not in problems.columns]
            new_problems = new_problems.drop(columns=remove_columns)
            problems = pd.concat([problems, new_problems], ignore_index=True)
            
    sceneGraphs_f = open(os.path.join(base_dir, "train_sceneGraphs.json"), "r")
    sceneGraphs = json.load(sceneGraphs_f)

    sceneGraphs_f.close()

    split_problems = build_prompt_chatbot(problems, sceneGraphs, bounding_box)
    
    if args.image_num == 0:
        args.image_num = len(problems)
    
    image_set = set()
    
    if split == "train":
        target_format = []
        for prob_id, (input, output) in tqdm(split_problems.items(), total=len(split_problems.keys())):
            
            raw_prob_data = problems.iloc[int(prob_id)]
            
            if len(image_set) >= args.image_num:
                break
            else:
                image_set.add(raw_prob_data["image_id"])
                
            
            if input.startswith("Question: "):
                input = input.replace("Question: ", "")
            if output.startswith("Answer: "):
                output = output.replace("Answer: ", "")

            
            if raw_prob_data["image_id"] is None:
                target_format.append(
                    {
                        "id": prob_id,
                        "conversations": [
                            {"from": "human", "value": f"{input}"},
                            {"from": "gpt", "value": answer_text},
                        ],
                    }
                )

            else:
                target_format.append(
                    {
                        "id": prob_id,
                        "image": str(raw_prob_data["image_id"]) + ".jpg",
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{input}"},
                            {"from": "gpt", "value": answer_text},
                        ],
                    }
                )
                
            
        
        print(f"Number of samples: {len(target_format)}")
        print(f"Number of images: {len(image_set)}")
        file_name = f"llava_q2q_{split}"
        
        if len(target_format) < len(problems) or args.add_sample is True:
            file_name = file_name + f'{len(target_format)}'
        
        if bounding_box is True:
            file_name += "_bb"
        with open(os.path.join(base_dir, file_name + f".json"), "w") as f:
            json.dump(target_format, f, indent=2)
    else:
        file_name = f"llava_q2q_{split}_multi"
        if bounding_box is True:
            file_name += "_bb"
        q_writer = open(os.path.join(base_dir, file_name + "_question.jsonl"), "w")
        a_writer = open(os.path.join(base_dir, file_name + "_answer.jsonl"), "w")
        
        prev_entity_id = None
        answer_text = []
        prev_prob_id = None  # question_id

        for prob_id, (input, output) in split_problems.items():
            if input.startswith("Question: "):
                input = input.replace("Question: ", "")
            if output.startswith("Answer: "):
                output = output.replace("Answer: ", "")

            raw_prob_data = problems.iloc[int(prob_id)]
            cur_entity_id = raw_prob_data['entity_id']

             # If the entity changes, save the previous one to the file
            if prev_entity_id is not None and cur_entity_id != prev_entity_id:
                # prev_prob_id, answer_text 사용해서 answer_data 생성
                answer_data = {
                    "question_id": prev_prob_id,
                    "text": answer_text,
                    "category": "conv",
                }
                a_writer.write(json.dumps(answer_data) + "\n")
                answer_text = []

            if cur_entity_id != prev_entity_id:
                if pd.isnull(raw_prob_data["image_id"]):  # None 체크 (pandas)
                    question_data = {
                        "question_id": prob_id,
                        "text": f"{input}",
                        "category": "conv",
                    }
                else:
                    question_data = {
                        "question_id": prob_id,
                        "image": str(raw_prob_data["image_id"]) + ".jpg",
                        "text": f"{input}",
                        "category": "conv",
                    }
                q_writer.write(json.dumps(question_data) + "\n")
                prev_prob_id = prob_id  # 답변 쓸 id 저장

            # append output
            answer_text.append(output)
            prev_entity_id = cur_entity_id

        # last entity
        if answer_text:
            answer_data = {
                "question_id": prev_prob_id,
                "text": answer_text,
                "category": "conv",
            }
            a_writer.write(json.dumps(answer_data) + "\n")    

        q_writer.close()
        a_writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--bounding_box", action='store_true')
    parser.add_argument("--image_num", default=0, type=int )
    parser.add_argument("--add_sample", action="store_true")

    args = parser.parse_args()
    convert_to_llava(args)

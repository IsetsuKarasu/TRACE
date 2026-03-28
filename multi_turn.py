import argparse
import json
import ast
import os
import re
import warnings
import pandas as pd
from collections import defaultdict
from sortedcontainers import SortedList

import ollama

from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY", base_url="YOUR_BASE_URL")
closed_models = {'gpt-4', 'gpt-4o-mini', 'gpt-4o'}

import spacy
from difflib import SequenceMatcher
nlp = spacy.load("en_core_web_md")


def isSemanticComplete(table, que, prevQues):
    if que in prevQues:
        return prevQues[que] == 'complete'

    system_prompt = "You are an expert language model trained to evaluate whether a question is semantically complete."
    user_prompt = f"""
Given the following question, determine whether it is semantically complete or semantically incomplete. Explain your reasoning briefly.
A semantically complete question is self-contained and unambiguous. A semantically incomplete question relies on omitted information, references, or prior context to make sense.

Examples:
Complete: How's the weather in Shanghai?
Incomplete: What about Beijing?

The response you generate should be formatted in the following format:
{{"judgment": "complete" or "incomplete", "reason": "brief explanation"}}

Table:
{table}

Question:
{que}

Your Judgment: """

    if args.log:
        with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"----------Judge Semantic Completeness----------\n")
            f.write("----------Prompt----------")
            f.write(user_prompt + '\n')

    model = args.engine
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    retries = args.retries
    while retries > 0:
        try:
            retries -= 1
            if model in closed_models:
                output = client.chat.completions.create(model=model, messages=messages)
                response = output.choices[0].message.content
            else:
                output = ollama.chat(model=model, messages=messages, stream=False)
                response = output['message']['content']
            answers = re.findall(r'\{([^\}]*)\}', response, flags=re.DOTALL)
            if not answers:
                continue
            for answer in answers:
                try:
                    output = json.loads('{' + answer + '}')
                    judgment = output['judgment'].lower()
                except Exception as e:
                    # print("Scan Output: ", str(e) or repr(e))
                    continue
            if judgment not in {'complete', 'incomplete'}:
                continue

            break
        except Exception as e:
            pass
            # print("Judge Question: ", str(e) or repr(e))
    if retries <= 0:
        judgment = 'incomplete'

    if args.log:
        with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
            f.write("----------Response----------\n")
            f.write(response + '\n')
            f.write("----------Judgment----------\n")
            f.write(judgment + '\n')
        with open('./Logs/judge_log.txt', 'a', encoding='utf-8') as f:
            f.write("----------Judgment----------\n")
            f.write(f"{que} is {judgment}" + '\n')

    prevQues[que] = judgment

    return judgment == 'complete'

def needOverallSearch(table, que):
    #TODO: implement overall search detection
    return False

def findSimilarQuestions(test, ques):
    def isSyntaxSimilar(text1, text2, method='llm'):
        if method == 'dep':
            fp1_dep = [token.dep_ for token in nlp(text1)]
            fp2_dep = [token.dep_ for token in nlp(text2)]

            return SequenceMatcher(None, fp1_dep, fp2_dep).ratio()

        system_prompt = "You are an expert language model trained to evaluate whether two questions are syntactically similar."
        user_prompt = f"""
Given two questions, determine whether they are syntactically similar or not. Explain your reasoning briefly.
A syntactically similar question has the same structure and wording patterns.

Examples——Questions about the weather:
Similar: What about Shanghai?
Similar: What about Beijing?

The response you generate should be formatted in the following format:
{{"judgment": "yes" or "no", "reason": "brief explanation"}}

Question:
1.{text1}
2.{text2}

Your Judgment: """

        if args.log:
            with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"----------Judge Syntax Similarity----------\n")
                f.write("----------Prompt----------")
                f.write(user_prompt + '\n')

        model = args.engine
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        retries = args.retries
        while retries > 0:
            try:
                retries -= 1
                if model in closed_models:
                    output = client.chat.completions.create(model=model, messages=messages)
                    response = output.choices[0].message.content
                else:
                    output = ollama.chat(model=model, messages=messages, stream=False)
                    response = output['message']['content']
                answers = re.findall(r'\{([^\}]*)\}', response, flags=re.DOTALL)
                if not answers:
                    continue
                for answer in answers:
                    try:
                        output = json.loads('{' + answer + '}')
                        judgment = output['judgment'].lower()
                    except Exception as e:
                        # print("Scan Output: ", str(e) or repr(e))
                        continue
                if judgment not in {'yes', 'no'}:
                    continue

                break
            except Exception as e:
                pass
                # print("Judge Question: ", str(e) or repr(e))
        if retries <= 0:
            judgment = 'no'

        if args.log:
            with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
                f.write("----------Response----------\n")
                f.write(response + '\n')
                f.write("----------Judgment----------\n")
                f.write(judgment + '\n')
            with open('./Logs/judge_log.txt', 'a', encoding='utf-8') as f:
                f.write("----------Judgment----------\n")
                f.write(f"{text1} and {text2} is {judgment}" + '\n')

        return judgment == 'yes'

    res = SortedList()
    threshold = 0.75
    for i, que in enumerate(ques):
        # if isSyntaxSimilar(test, que, 'llm'):
        #     res.add(i)
        if isSyntaxSimilar(test, que, 'dep') >= threshold:
            res.add(i)
    return res

def findSimilarBrothers(que, pool):
    bros = set()
    for parent in que["parents"]:
        for child in pool[parent]["children"]:
            que_id, question = pool[child]["id"], pool[child]["question"]
            if que_id != que["id"]:
                bros.add((que_id, question))
    bros = list(bros)

    res = set()
    if bros:
        for id in findSimilarQuestions(que["question"], [x[1] for x in bros]):
            res.add(bros[id][0])
    return res

def getHistory(table, que, pool, prevQues):
    id = len(pool)
    que_ids = set()

    if id == 1 or isSemanticComplete(table, que, prevQues):
        prevQues[que] = 'complete'
        que_ids = {0}
    elif needOverallSearch(table, que):
        # TODO: implement overall search
        pass
    else:
        last_que = pool[-1]
        if isSemanticComplete(table, last_que["question"], prevQues) or not findSimilarQuestions(que, [last_que["question"]]):
            que_ids = findSimilarBrothers(last_que, pool) | {id-1}
        else:
            que_ids = pool[-1]["parents"]

    for parent in que_ids:
        pool[parent]["children"].add(id)

    pool.append({"id": id, "question": que, "parents": que_ids, "children": set(), "plan": ""})
    return que_ids

def getPlan(prompt_tables, utterance, pool, prevQues):
    plan = []

    system_prompt = "You are a tabular data analyst."
    module_prompt = f"""1.Input Module: Load the table file. Use the literal file path 'PATH.csv' for substitution.
2.Preprocess Module: Normalize or clean the table data, like transferring string to numeric format, parsing date data and trimming the unnecessary string characters.
3.Filter Module: Produce structured selection conditions and target fields. The condition fields should be a list of strings formatted as '<Field> <Operator> <Value>'; the target fields should be a list of column names. Note: take care of some special instructions, e.g., 'first/last few records', 'all the data', 'unique data'.
4.Operate Module: Apply operations to the filtered data, like calculating(Sum/Average/Count), grouping(Rank/Aggregate/Offset after grouping) or logically comparing(isNull/Compare/Rank/Min/Max).
5.Output Module: Form the final answer string and set variable 'answer'.

You can choose and combine the above modules in any way to form a modular plan to answer the question step by step. For each module in the plan, you need to describe its function in detail.
Note: you can only use each module ONCE in the plan."""

    history_ids = getHistory(prompt_tables, utterance, pool, prevQues)

    history_ids = [hid for hid in history_ids if hid != 0]
    if not history_ids:
        user_prompt = f"""
Given a table analysis question, design a modular plan to solve the question step by step. The module includes:
{module_prompt}

Table(Partial):
{prompt_tables}

Question:
{utterance}

The plan you generate should be formatted in the following format:
[{{"id": 0, "Module": "Module_Name", "Description": "Description of the module"}}, ...]

Your Plan: """
    else:
        prev_que = f""""""
        for i, id in enumerate(history_ids):
            prev_que += f"""
{i + 1}. {pool[id]['question']}"""
        prev_plan = pool[history_ids[-1]]["plan"]

        user_prompt = f"""
Given previous relevant table analysis questions and the modular plan for one of them, MODIFY the plan to solve the current question step by step. The module includes:
{module_prompt} If necessary, you should NOT delete the existing information in the given plan, but make modifications based on it.

Table(Partial):
{prompt_tables}

Previous Questions: {prev_que}

Modular Plan:
{json.dumps(prev_plan, indent=4)}

Current Question:
{utterance}

The format of the plan you generate should be as follows:
[{{"id": 0, "Module": "Module_Name", "Description": "Description of the module"}}, ...]

Your Plan: """

    if args.log:
        with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"----------Gen Plan----------\n")
            f.write("----------Prompt----------")
            f.write(user_prompt + '\n')

    model = args.engine
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    while True:
        try:
            if model in closed_models:
                output = client.chat.completions.create(model=model, messages=messages)
                response = output.choices[0].message.content
            else:
                output = ollama.chat(model=model, messages=messages, stream=False)
                response = output['message']['content']
            answers = re.findall(r'\[([^\]]*)\]', response, flags=re.DOTALL)
            if not answers:
                continue
            for answer in answers:
                try:
                    plan = json.loads('[' + answer + ']')
                    for module in plan:
                        if not isinstance(module, dict):
                            raise ValueError("Not a dict")
                        if 'id' not in module or 'Module' not in module or 'Description' not in module:
                            raise ValueError("Missing keys")
                except Exception as e:
                    # print("Scan Plan: ", str(e) or repr(e))
                    continue
            if not plan:
                continue

            for i, module in enumerate(plan):
                if module["Module"] == "Input Module":
                    plan[i]["Description"] = "Load the table file. Use the literal file path 'PATH.csv' for substitution."
                if module["Module"] == "Output Module":
                    plan[i]["Description"] = "Form the final answer string and set variable 'answer'."

            break
        except Exception as e:
            pass
            print("Get Plan: ", str(e) or repr(e))

    if args.log:
        with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
            f.write("----------Response----------\n")
            f.write(response + '\n')
            f.write("----------Plan----------\n")
            f.write(json.dumps(plan, indent=4) + '\n')
        with open('./Logs/module_log.txt', 'a', encoding='utf-8') as f:
            f.write("----------Plan----------\n")
            f.write(json.dumps(plan, indent=4) + '\n')

    return plan


def getCode(prompt_tables, utterance, table_file_path, plan):
    code = ""

    system_prompt = "You are a tabular data analyst proficient in Python for data analysis. "
    user_prompt = f"""
Given the following table analysis question and the plan, produce working Python code that implements the plan to answer the question.

Table(Partial):
{prompt_tables}

Question:
{utterance}

Plan:
{json.dumps(plan, indent=4)}

Your Code: """
    if args.log:
        with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"----------Gen Code----------\n")
            f.write("----------Prompt----------")
            f.write(user_prompt + '\n')

    model = args.engine
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    retries = args.retries
    while True:
        try:
            if model in closed_models:
                output = client.chat.completions.create(model=model, messages=messages)
                response = output.choices[0].message.content
            else:
                output = ollama.chat(model=model, messages=messages, stream=False)
                response = output['message']['content']
            answers = re.findall(r'```python\n?(.*?)```', response, flags=re.DOTALL)
            if not answers:
                continue
            for answer in answers:
                try:
                    code = answer.strip('```').strip('python').strip('\n').replace("PATH.csv", f"{table_file_path}")
                except Exception as e:
                    continue
            if not code:
                continue

            error = "No Error"
            analysis_answer = "No Answer"
            exec_globals = {'__builtins__': __builtins__}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(code, exec_globals)
                analysis_answer = exec_globals['answer']
                if not analysis_answer or isinstance(analysis_answer, (pd.Series, pd.DataFrame)):
                    continue
                analysis_answer = str(analysis_answer)
            except Exception as e:
                error = str(e) or repr(e)

            if error != "No Error":
                if retries > 0:
                    retries -= 1
                    continue
                else:
                    output = f"Analysis Fail: {error}"
            else:
                output = analysis_answer
            break
        except Exception as e:
            pass
            # print("Get Code: ", str(e) or repr(e))

    if args.log:
        with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
            f.write("----------Response----------\n")
            f.write(response + '\n')
            f.write("----------Code----------\n")
            f.write(code + '\n')

    return output


def execute(index, item, dataset, csv_path, max_tbl_size):
    pool = [{"id": 0, "question": "", "parents": set(), "children": set(), "plan": ""}]
    prevQues = {}

    name = f'Table {index+1}'
    table = item['Table']
    utterances = item['Questions']
    values = item['Answers']
    table_file_path = os.path.join(csv_path, table)

    table_text = pd.read_csv(table_file_path, encoding='utf-8')
    col = [x.replace('\n', '-') for x in list(table_text.columns)]

    pre_text = ""
    post_text = ""
    if dataset != 'SQA':
        pre_text = "Before Table: "
        post_text = """
After Table: """
        if isinstance(item['Pre_Text'], str):
            pre_text += item['Pre_Text']
        else:
            for pre in item['Pre_Text']:
                pre_text += pre + " "
        if isinstance(item['Post_Text'], str):
            post_text += item['Post_Text']
        else:
            for post in item['Post_Text']:
                post_text += post + " "

    prompt_schema = f"""
Table Name : `{name}`
Table Fields : {' | '.join(f"`{x}`" for x in col)}"""
    prompt_records = f""""""
    num_row = min(max_tbl_size, len(table_text))
    for i in range(num_row):
        row = [str(x) for x in list(table_text.iloc[i])]
        prompt_record = f"""
Row {i + 1} : {' | '.join(' '.join(str(x).split(' ')[:10]) + '...' if len(str(x)) >= 100 else str(x) for x in row)}"""
        prompt_records += prompt_record
    if num_row < len(table_text):
         prompt_records += f"""
......
"""
    prompt_tables = pre_text + prompt_schema + prompt_records + post_text

    outputs = []
    test_values = []
    for cnt, (utterance, value) in enumerate(zip(utterances, values)):
        print(f"-----Executing Question {cnt+1}/{len(utterances)}-----")
        if args.log:
            with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"----------Analysis {cnt + 1} Start----------\n")

        plan = getPlan(prompt_tables, utterance, pool, prevQues)
        answer = getCode(prompt_tables, utterance, table_file_path, plan)
        pool[-1]["plan"] = plan

        try:
            answer = ast.literal_eval(answer)
        except Exception as e:
            pass

        if isinstance(value, list):
            value = ", ".join(str(x) for x in value)
        if isinstance(answer, list):
            answer = ", ".join(str(x) for x in answer)

        if str(answer).isdigit() and str(value).isdigit():
            if str(answer) == str(value):
                answer = value

        outputs.append(answer)
        test_values.append(value)

        if args.log:
            with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"----------Analysis {cnt + 1} End----------\n")
            with open('./Logs/trial_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"----------Question {cnt + 1} Info----------\n")
                f.write(f"----------Final Result----------\n")
                f.write(str(answer))
                f.write('\n')
                f.write(f"----------Correct Result----------\n")
                f.write(str(value))
                f.write('\n\n')

    if args.log:
        for i, node in enumerate(pool):
            pool[i]["children"] = list(node["children"])
            pool[i]["parents"] = list(node["parents"])
        with open('./Logs/history_pool.json', 'w', encoding='utf-8') as f:
            json.dump(pool, f, indent=4)

    return [utterance for utterance in utterances], outputs, test_values, len(table_text)


def llm_judge(question, prediction, value):
    if str(prediction).strip().lower() == str(value).strip().lower():
        return True
    
    system_prompt = "You are an expert language model trained to evaluate the correctness of answers."
    user_prompt = f"""Given the following question, predicted answer and the correct answer, determine whether the predicted answer is semantically equivalent to the correct answer. Explain your reasoning briefly.
Your Judgment should be either "true" or "false", along with a brief explanation.

The response you generate should be formatted in the following format:
{{"judgment": "true" or "false", "reason": "brief explanation"}}

Question:
{question}

Predicted Answer:
{prediction}

Correct Answer:
{value}

Your Judgment: """
    if args.log:
        with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f_l:
            f_l.write(f"----------Judge Answer----------\n")
            f_l.write("----------Prompt----------\n")
            f_l.write(user_prompt + '\n')

    model = args.engine
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    retries = args.retries
    while retries > 0:
        try:
            retries -= 1
            if model in closed_models:
                output = client.chat.completions.create(model=model, messages=messages)
                response = output.choices[0].message.content
            else:
                output = ollama.chat(model=model, messages=messages, stream=False)
                response = output['message']['content']
            answers = re.findall(r'\{([^\}]*)\}', response, flags=re.DOTALL)
            if not answers:
                continue
            for answer in answers:
                try:
                    output = json.loads('{' + answer + '}')
                    judgment = output['judgment'].lower()
                except Exception as e:
                    # print("Scan Output: ", str(e) or repr(e))
                    continue
            if judgment not in {'true', 'false'}:
                continue

            break
        except Exception as e:
            pass
            # print("Judge Answer: ", str(e) or repr(e))
    if retries <= 0:
        judgment = 'false'

    if args.log:
        with open('./Logs/run_log.txt', 'a', encoding='utf-8') as f_l:
            f_l.write("----------Response----------\n")
            f_l.write(response + '\n')

    return judgment == 'true'


def result_stat():
    with open('./Logs/exec_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        total_ques = data['questions']
        total_outputs = data['outputs']
        total_values = data['values']
        total_lens = data['lens']
        total_turns = data['turns']

    data = [{"Small": [], "Mid": [], "Large": []}, defaultdict(list), defaultdict(list)]

    cnt = 0
    for ques, outputs, values, tbl_len, turn in zip(total_ques, total_outputs, total_values, total_lens, total_turns):
        tbl = ""
        qm = 0
        sm = 0
        im = 0
        isSm = True

        if tbl_len <= 10:
            tbl = "Small"
        elif tbl_len <= 20:
            tbl = "Mid"
        else:
            tbl = "Large"
        for i, (que, output, value) in enumerate(zip(ques, outputs, values)):
            data[2].setdefault(i+1, [0, 0])
            print(f'-----Judging Question {cnt+1}-----')
            if llm_judge(que, output, value):
                qm += 1
                if isSm:
                    sm += 1
                data[2][i+1][0] += 1
            else:
                isSm = False
            data[2][i+1][1] += 1
            cnt += 1
        if qm == turn:
            im += 1
        output = {"QM": qm, "SM": sm, "IM": im, "Total": turn}
        data[0][tbl].append(output)
        data[1][turn].append(output)

    def compute_average(sizes, size):
        qm = 0
        im = 0
        sm = 0
        que_cnt = 0
        tbl_cnt = 0
        for tbl in sizes[size]:
            qm += tbl["QM"]
            im += tbl["IM"]
            sm += tbl["SM"]
            que_cnt += tbl["Total"]
            tbl_cnt += 1
        average_qm = qm / que_cnt if que_cnt > 0 else 0
        average_im = im / tbl_cnt if tbl_cnt > 0 else 0
        average_sm = sm / que_cnt if que_cnt > 0 else 0
        print(f"{size} Tables: QM: {average_qm}, IM: {average_im}, SM: {average_sm}")

    ttl_tbl = len(data[0]["Small"])+len(data[0]["Mid"])+len(data[0]["Large"])
    ttl_que = 0

    compute_average(data[0], "Small")
    compute_average(data[0], "Mid")
    compute_average(data[0], "Large")

    turns = data[1]
    pos = data[2]
    ttl_qm = 0
    ttl_im = 0
    ttl_sm = 0
    for turn in sorted(turns):
        qm = 0
        im = 0
        sm = 0
        que_cnt = 0
        tbl_cnt = 0
        for tbl in turns[turn]:
            qm += tbl["QM"]
            im += tbl["IM"]
            sm += tbl["SM"]
            que_cnt += tbl["Total"]
            tbl_cnt += 1
        ttl_que += que_cnt
        average_qm = qm / que_cnt if que_cnt > 0 else 0
        average_im = im / tbl_cnt if tbl_cnt > 0 else 0
        average_sm = sm / que_cnt if que_cnt > 0 else 0
        print(f"Turn {turn}: QM: {average_qm}, IM: {average_im}, SM: {average_sm}")

        ttl_qm += qm
        ttl_im += im
        ttl_sm += sm

    for p in sorted(pos):
        qm, total = pos[p]
        print(f"Position {p}: {qm/total}")

    print(f"Overall: QM: {ttl_qm/ttl_que}, IM: {ttl_im/ttl_tbl}, SM: {ttl_sm/ttl_que}")


def main():
    dataset = args.dataset
    max_tbl_size = args.max_tbl_size

    que_path = f'./Dataset/{dataset}/questions/test.json'
    csv_path = f'./Dataset/{dataset}'
    with open('./Logs/history_pool.json', 'w', encoding='utf-8') as f:
        json.dump([], f, indent=4)
    with open('./Logs/run_log.txt', 'w', encoding='utf-8') as f:
        pass
    with open('./Logs/trial_log.txt', 'w', encoding='utf-8') as f:
        pass
    with open('./Logs/module_log.txt', 'w', encoding='utf-8') as f:
        pass
    with open('./Logs/judge_log.txt', 'w', encoding='utf-8') as f:
        pass

    with open(que_path, 'r', encoding='utf-8') as f_i:
        que_data = json.load(f_i)

    # que_data = que_data[:12]+que_data[14:100]
    que_data = [que_data[697]]

    total_ques = []
    total_outputs = []
    total_values = []
    total_lens = []
    total_turns = []
    for i in range(len(que_data)):
        print(f"-----Executing Table {i+1}-----")
        ques, outputs, test_values, tbl_len = execute(i, que_data[i], dataset, csv_path, max_tbl_size)
        total_ques.append(ques)
        total_outputs.append(outputs)
        total_values.append(test_values)
        total_lens.append(tbl_len)
        total_turns.append(len(test_values))

    with open('./Logs/exec_results.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data['questions'] += total_ques
        data['outputs'] += total_outputs
        data['values'] += total_values
        data['lens'] += total_lens
        data['turns'] += total_turns
        f.seek(0)
        json.dump(data, f, indent=4)

    result_stat()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='SQA', choices=['SQA'])
    parser.add_argument('--engine', type=str, default='gpt-4o', choices=['llama3.1', 'qwen3:8b', 'gpt-4', 'gpt-4o-mini', 'gpt-4o'])
    parser.add_argument('--max-tbl-size', type=int, default=6)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--log', action='store_true')

    args = parser.parse_args()

    main()

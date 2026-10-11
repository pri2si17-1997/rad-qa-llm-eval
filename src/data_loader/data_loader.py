import json

import pandas as pd


def get_flatten_radqa_data(file_path: str) -> pd.DataFrame:
    """
    This method flattens nested json structure.

    :param file_path: Json data file path
    :type file_path: str
    :return: Pandas dataframe with flat structure.
    :rtype: DataFrame
    """
    with open(file_path, "r") as fp:
        data = json.load(fp)

    radqa_json_data = data["data"]
    num_radqa_json_data = len(radqa_json_data)

    # Hack for pd.json_normalize
    for i in range(num_radqa_json_data):
        paragraphs = data["data"][i]["paragraphs"]
        for sub_para in paragraphs:
            for qas in sub_para["qas"]:
                if len(qas["answers"]) == 0:
                    qas["answers"] = [{}]

    # Flatten
    rad_qa_flatten_df = pd.json_normalize(
        radqa_json_data,
        record_path=["paragraphs", "qas", "answers"],
        meta=[
            ["paragraphs", "context"],  # Context
            ["paragraphs", "qas", "id"],  # Question ID
            ["paragraphs", "qas", "question"],  # Question text,
            ["paragraphs", "qas", "is_impossible"],
        ],
    )

    # Rename columns
    rad_qa_flatten_df = rad_qa_flatten_df.rename(
        columns={
            "text": "answer_text",
            "paragraphs.context": "context",
            "paragraphs.qas.id": "question_id",
            "paragraphs.qas.question": "question",
            "paragraphs.qas.is_impossible": "is_impossible",
        }
    )

    rad_qa_flatten_df = rad_qa_flatten_df[
        [
            "context",
            "question",
            "is_impossible",
            "answer_start",
            "answer_text",
            "question_id",
            "answer_id",
        ]
    ]

    return rad_qa_flatten_df

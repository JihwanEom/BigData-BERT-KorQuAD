import json 
import logging 
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


with open('KorQuAD_v1.0_train.json', 'r') as f:
    train_data = json.load(f) 

with open('KorQuAD_v1.0_dev.json', 'r') as f:
    dev_data = json.load(f) 

train_data = [item for topic in train_data['data'] for item in topic['paragraphs'] ]
dev_data = [item for topic in dev_data['data'] for item in topic['paragraphs'] ]

model_args = QuestionAnsweringArgs(num_train_epochs=10, manual_seed=1234, overwrite_output_dir=True)
model_args.train_batch_size = 128
model = QuestionAnsweringModel(
    "bert", "bert-base-multilingual-cased", args=model_args
)

metric, eval_info = model.eval_model(dev_data)
print(f"Before fine-tuning - correct: {metric['correct']} | similar: {metric['similar']} | incorrect: {metric['incorrect']}")
model.train_model(train_data=train_data, eval_data=dev_data)
metric, eval_info = model.eval_model(dev_data)
print(f"After fine-tuning - correct: {metric['correct']} | similar: {metric['similar']} | incorrect: {metric['incorrect']}")

preds, _ = model.predict(dev_data, n_best_size=1)

with open('KorQuAD_v1.0_dev.json', 'r') as f:
    gt_data = json.load(f) 
gt_data = [item for topic in gt_data['data'] for item in topic['paragraphs'] ]

count = 0
for i in range(len(gt_data)):
    qas = gt_data[i]['qas']
    context = gt_data[i]['context']
    for qa in qas:
        q = qa['question']
        a = qa['answers'][0]['text']
        p = preds[count]['answer'][0]
        if p != 'empty':
            print()
            print(f"C: {context}")
            print(f"Q: {q}")
            print(f"A: {a}")
            print(f"P: {p}")
        count += 1
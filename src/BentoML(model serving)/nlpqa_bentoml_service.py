from __future__ import annotations
import bentoml
from bentoml.io import Text, JSON
from transformers import pipeline
import json

nlpa_qa_settings = json.loads('''{
  "nlp_qa": {
    "model_name": "deepset/roberta-base-squad2",
    "questions": [
      "What should be improved?",
      "What was the topic of the class?",
      "What was the lesson about?",
      "What are the next steps?",
      "What is the email address?",
      "How was the conversation?"
    ]
  },
  "student_analysis": {
    "nlp_questions_to_store": [
      "What should be improved?",
      "What are the next steps?",
      "What is the email address?",
      "How was the conversation?"
    ]
  }
}''')


class RobertaQARunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.nlp_qa = pipeline('question-answering', model=nlpa_qa_settings['nlp_qa']['model_name'])

    @bentoml.Runnable.method(batchable=False)
    def __call__(self, input_context):
        model_answers = {}
        for question in nlpa_qa_settings['nlp_qa']['questions']:
            answer = self.nlp_qa(context=input_context, question=question).get('answer', '').strip()
            model_answers[question] = answer if not '<s>' in answer else ''
        return model_answers

runner = bentoml.Runner(RobertaQARunnable, name="pretrained_qa_roberta")

svc = bentoml.Service('pretrained_qa_roberta_svc', runners=[runner])
@svc.api(input=Text(), output=JSON())
def get_answers(text: str) -> dict:
    return runner.run(text)
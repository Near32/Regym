from typing import List
import dspy
from dspy.functional import TypedPredictor
from dspy import ChainOfThought
from dspy.predict.aggregation import majority

import pydantic
import weave


class MultiChoiceAnswer(pydantic.BaseModel):
    answer_id: int

class MultiChoiceQAOutline(dspy.Signature):
    """
    Your task is to answer the given question.
    Please use the context efficiently.
    """

    question: str = dspy.InputField()
    answer: MultiChoiceAnswer = dspy.OutputField()

class MajorityVotingMultiSamplingMultiChoiceTypedCoTQA(dspy.Module):
    def __init__(self, n_samples=10):
        super().__init__()
        self.n_samples = n_samples
        self.cot_prog = ChainOfThought('question -> answer', n=self.n_samples)
        self.typed_prod = TypedPredictor(MultiChoiceQAOutline)
        self.aggregation = majority
    
    @weave.op
    def forward(self, question:str)->List[object]:
        dspy.settings.lm.kwargs['temperature'] = 0.7
        #dspy.settings.lm.kwargs['do_sample'] = True
        dspy.settings.lm.kwargs['max_tokens'] = 16384
        predictions = []
        preds = self.cot_prog(question=question)
        '''
        preds = []
        for nidx in range(self.n_samples):
            resp = self.cot_prog(question=question)
            preds.append(resp)
        for p in preds:
        '''
        for pred in preds.completions.answer:
            #pred = p.answer
            typed_pred = self.typed_prod(question=pred)
            predictions.append({'answer': f"{typed_pred.answer.answer_id}"})
        pred = self.aggregation(predictions)
        return pred


def input_formatting_fn(
    prompt:str,
    options:List[str],
):
    '''
    Returns a dictionnary whose key and values follow the signature of the DSPy Module.
    :param prompt: str 
    :param options: List[str] 
    '''
    odict = {
        'question':prompt,
    }

    return odict 

def output_formatting_fn(
    prompt:str,
    response:object,
):
    '''
    The gt label starts at 0, and a label is expected, but the answer options start at value '1'.
    '''
    odict = {
        'answer_id': int(response.answer)-1
    }
    return odict


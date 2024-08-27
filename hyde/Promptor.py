SQUAD_H = """Create text that can answer the following question: Question: {0} Answer:"""
SQUAD_A = """Please answer based on the following text. ### Question:{0} Text:{1} Answer:"""

KOQUAD_H = """다음 질문에 대답할 수 있는 텍스트를 만들어주세요. ###질문: {0} ###답변:"""
KOQUAD_A = """다음 텍스트를 참고해서 답해주세요. ###질문:{0} ###텍스트:{1} ###답변:"""


class Promptor(object):
    """
    Class for Prompt, which is one of the HyDEQ Pipe Line. 
    """
    def __init__(self, task):
        self.task = task

    def build_prompt(self, query: str, text=None):
        # Korean Prompt
        if self.task == 'koquad' and text == None:
            return KOQUAD_H.format(query)
        elif self.task == 'koquad' and text != None:
            return KOQUAD_A.format(query, text)
        
        # English Prompt
        elif self.task == 'squad' and text == None:
            return SQUAD_H.format(query)
        elif self.task == 'squad' and text != None:
            return SQUAD_A.format(query, text)
        else:
            raise ValueError('Task not supported')
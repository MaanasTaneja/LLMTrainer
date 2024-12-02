import pandas
import openai
from dotenv import load_dotenv
import os

from dspy.teleprompt import BootstrapFewShot
import pandas as pd
from dspy import ChainOfThought
import dspy

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from datasets import load_dataset

'''
Testing dspy by training a language model to be snarky and answer questions with sarcasm 
without using ine tuning, but instead using in context learning to train the model to be snarky.

Ofcourse using lovely DSPY, which is incredible!

and then i test and try to do style transfer using a news scippet, and write tweets in that style.
'''

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   

lm = dspy.LM('openai/gpt-3.5-turbo', api_key=OPENAI_API_KEY,  cache=False)
dspy.configure(lm=lm)

df = pd.read_csv("finetune_data_csv.csv")
X = df.iloc[:, 0].to_numpy()
Y = df.iloc[:, 1].to_numpy()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def validate_answer(example , pred, trace = None):
    example_tensor = embedding_model.encode(example.answer)
    pred_tensor = embedding_model.encode(pred.answer)
    similarity = embedding_model.similarity(example_tensor, pred_tensor)

    if(similarity > 0.8):
        return True 
    else:
        return False

class ShortQA(dspy.Signature):
    '''Answer questions in one line and be snarky about the answer.'''
    question = dspy.InputField()
    answer = dspy.OutputField(desc="answer in one line or less with sarcasm.")


program = dspy.ChainOfThought(ShortQA)

Xtrain_raw, Xtest_raw, Ytrain_raw, Ytest_raw = train_test_split(X, Y, test_size = 0.2)

train_set , test_set = [], []

for i in range(0, len(Xtrain_raw)):
    example = dspy.Example(question = Xtrain_raw[i], answer = Ytrain_raw[i])
    example = example.with_inputs("question")
    train_set.append(example)

for i in range(0, len(Xtest_raw)):
    example = dspy.Example(question = Xtest_raw[i], answer = Xtest_raw[i])
    example = example.with_inputs("question")
    test_set.append(example)

print(train_set)

config = {"max_bootstrapped_demos" : 4, "max_labeled_demos" : 4, "max_rounds" : 10}
teleprompter = BootstrapFewShot(validate_answer, **config)

#student is the language model we want to train, and teacher is our metric essentially on the training data.

compiled_program = teleprompter.compile(student=program, trainset=train_set)

compiled_program.save("snarky_model.json")

lm.inspect_history(4)

for ex in test_set:
    resp = compiled_program(question = ex.question)
    print(resp.answer)
    print(validate_answer(ex, resp))



'''
for i, row in df.iterrows():
    question = row.iloc[0]
    answer = row.iloc[1]

    pred = program(question = question)
    eval = validate_answer(answer, pred.answer)
    print(eval)
    
'''

#genrerate a bunch of signatures??
r = lm("generate a one line prompt for a language model to achieve this task : question -> answer with sarcasm in one line. In your repsonse, do not add any extra language, just provide the answer. The generated response would be : ")
print(r)

t = '''
Exceprt from The New York Times:

Donald J. Trump rode a promise to smash the American status quo to win the presidency for a second time, surviving a criminal conviction, indictments, an assassin’s bullet, accusations of authoritarianism and an unprecedented switch of his opponent to complete a remarkable return to power.

Mr. Trump’s victory caps the astonishing political comeback of a man who was charged with plotting to overturn the last election but who tapped into frustrations and fears about the economy and illegal immigration to defeat Vice President Kamala Harris.

His defiant plans to upend the country’s political system held appeal to tens of millions of voters who feared that the American dream was drifting further from reach and who turned to Mr. Trump as a battering ram against the ruling establishment and the expert class of elites.

In a deeply divided nation, voters embraced Mr. Trump’s pledge to seal the southern border by almost any means, to revive the economy with 19th-century-style tariffs that would restore American manufacturing and to lead a retreat from international entanglements and global conflict.

Now, Mr. Trump will serve as the 47th president four years after reluctantly leaving office as the 45th, the first politician since Grover Cleveland in the late 1800s to lose re-election to the White House and later mount a successful run. At the age of 78, Mr. Trump has become the oldest man ever elected president, breaking a record held by President Biden, whose mental competence Mr. Trump has savaged.

His win ushers in an era of uncertainty for the nation.
'''

class Tweet(dspy.Signature):
    '''Generates a tweet based on given news, commenting and giving opinion on it style.'''
    news = dspy.InputField()
    tweet = dspy.OutputField(desc="Generates a tweet based on given news, commenting and giving opinion on it style.")

tweeter = dspy.ChainOfThought(Tweet)
res = tweeter(news =  t)
print(res.tweet)

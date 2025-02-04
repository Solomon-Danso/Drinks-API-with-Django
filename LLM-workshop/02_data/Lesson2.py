import re
import importlib
import tiktoken
from supplementary import create_dataloader_v1



with open("/Users/glydetek/Desktop/Hydot/Contributions/Drinks-API-with-Django/LLM-workshop/Data/source.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    


#Splits the inputs into a token 
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)

#Tokenize it
uniqueToken  = sorted(set(preprocessed))
vocab_size = len(uniqueToken)
print(vocab_size) 

vocab = {}
for index, word in enumerate(uniqueToken):
    vocab[word] = index


#tiktoken has it own originally created vocabulary
#No Room for errors 

tokenizer = tiktoken.get_encoding("gpt2")  
sample_text = "djsdkd!"
encoded_text = tokenizer.encode(sample_text, allowed_special={"<|endoftext|>"})  # ✅ Correct, passing a string


# We are dividing it into batches
# batch_size=8, max_length=4 means 8 columns per token, 4 per row 
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)





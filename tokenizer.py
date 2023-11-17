import pickle



with open('./input/input1.txt', 'r', encoding='utf-8') as f:
    text11 = f.read()

with open('./input/input2.txt', 'r', encoding='utf-8') as f:
    text12 = f.read()

with open('./input/input3.txt', 'r', encoding='utf-8') as f:
    text13 = f.read()

with open('./input/input4.txt', 'r', encoding='utf-8') as f:
    text14 = f.read()


with open('./input/alice.txt', 'r', encoding='utf-8') as f:
    text2 = f.read()

with open('./input/farm.txt', 'r', encoding='utf-8') as f:
    text3 = f.read()

text1 = text11 + text12 + text13# + text14

text1 = text1 + ' ' * (3 - (len(text1) % 3))  # Make sure the text length is divisible by 3 by adding spaces if needed
text2 = text2 + ' ' * (3 - (len(text2) % 3))  # Make sure the text length is divisible by 3 by adding spaces if needed
text3 = text3 + ' ' * (3 - (len(text3) % 3))  # Make sure the text length is divisible by 3 by adding spaces if needed


text = text1 + text2 + text3
text = text + ' ' * (3 - (len(text) % 3))  # Make sure the text length is divisible by 3 by adding spaces if needed


text1_tokens = [text1[i:i + 3] for i in range(0, len(text1), 3)]
text2_tokens = [text2[i:i + 3] for i in range(0, len(text2), 3)]
text3_tokens = [text3[i:i + 3] for i in range(0, len(text3), 3)]
text_tokens = [text[i:i + 3] for i in range(0, len(text), 3)]


tokens = list(set(text_tokens))

with open('./input/tokens.pkl', 'wb+') as f:
    pickle.dump(tokens, f)

with open('./input/input.pkl', 'wb+') as f:
    pickle.dump(text1_tokens, f)

with open('./input/alice.pkl', 'wb+') as f:
    pickle.dump(text2_tokens, f)

with open('./input/farm.pkl', 'wb+') as f:
    pickle.dump(text3_tokens, f)
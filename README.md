# 本地训练实现英文文本生成功能的 GPT-2 模型

## 0 背景
在自然语言处理领域当中， GPT-2 模型作为 Transformer 架构的重要代表之一，拥有较好的文本生成能力。本项目将实现自定义的文本数据集上预训练语言模型，同时将其用于自然语言生成(NLG)的任务。 
我将以简·奥斯汀的经典小说《Emma》作为训练数据集，通过 HuggingFace 的 transformers 库，完成从数据准备、分词器训练到模型训练和最终的文本生成的全过程。
## 1 语言模型训练
使用 transformers 库训练语言模型。
### 1.1 数据准备
HuggingFace 模型可以直接使用 TensorFlow 或 PyTorch 进行训练。本项目使用 TensorFlow 原生训练函数进行训练。  
在数据准备阶段，首先需要获取训练GPT-2所需的语料库，即获取《Emma》的原始文本数据，保存于austen-emma.txt中。
### 1.2 分词器训练  
通过 BPE 分词器类创建分词器对象，关于标准化部分，添加 Lowercase()处理，并将pre_tokenizer 属性设置为 ByteLeve1 ，以确保输入为bytes。同时，decoder 属性设置为 ByteLevelDecoder，确保正确解码。  
  
step 1.在用于训练 GPT-2 的语料库上训练 BytePairEncoding 分词器。从 tokenizers 库中导入BPE分词器。
```python
from tokenizers import ByteLevelBPETokenizer
import tensorflow as tf
import numpy as np
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
```
step 2.通过增加 Lowercase() 标准化来训练一个更先进的分词器。
```python
tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Sequence([
    Lowercase()
])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()
```
step 3.设置最大词汇量为 50000，使用 ByteLevel 的初始字母表训练分词器，添加特殊词元。
```python
trainer = BpeTrainer(vocab_size=50000, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
tokenizer.train(["austen-emma.txt"], trainer)
```
step 4.创建目录 tokenizer_gpt 保存分词器。保存分词器为 tokenizer.json 文件。
```python
import os
os.mkdir('tokenizer_gpt')
tokenizer.save("tokenizer_gpt/tokenizer.json")
```
### 1.3 数据预处理
对训练数据集即语料库进行预处理，使用分词器 tokenizer.json 为下一步训练 GPT-2 做准备。  
step 1.导入模块
```python
from transformers import GPT2TokenizerFast, GPT2Config, TFGPT2LMHeadModel
```
step 2.使用GPT2TokenizerFast加载分词器。
```python
tokenizer_gpt = GPT2TokenizerFast.from_pretrained("tokenizer_gpt")
```
step 3.添加特殊词元及其对应标记。
```python
tokenizer_gpt.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})
```
### 1.4 模型训练
step 1.初始化 GPT-2 的 TensorFlow 版本，创建配置对象 config。
```python
config = GPT2Config(
  vocab_size=tokenizer_gpt.vocab_size,
  bos_token_id=tokenizer_gpt.bos_token_id,
  eos_token_id=tokenizer_gpt.eos_token_id
)
model = TFGPT2LMHeadModel(config)
```
step 2.载入语料库进行预训练。
```python
with open("austen-emma.txt", "r", encoding='utf-8') as f:
    content = f.readlines()
```
step 3.原始文件中提取的 content 包含的所有文本，需过滤无效部分，即去除每行的换行符 \n，同时略去字符较少的行。
```python
content_p = []
for c in content:
    if len(c)>10:
        content_p.append(c.strip())
content_p = " ".join(content_p)+tokenizer_gpt.eos_token
```
step 4.创建模型训练的样本，将每个样本的大小设置为 100，从给定文本的某部分开始，到 100 个词元后结束。
```python
examples = []
block_size = 100
BATCH_SIZE = 12
BUFFER_SIZE = 1000
for i in range(0, len(tokenized_content)):
    examples.append(tokenized_content[i:i + block_size])
```
step 5. train_data 中样本序列大小为 99，从序列的起始位置至第 99 个 token，labels 序列则包含从第 1 个 token 至第 100 个 token。
```python
train_data = [] 
labels = [] 
for example in examples: 
    train_data.append(example[:-1]) 
    labels.append(example[1:])
```
step 6.将数据转换为 TensorFlow 数据集形式，加速训练。
```python
dataset = tf.data.Dataset.from_tensor_slices((train_data[:1000], labels[:1000]))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
```
step 7.指定优化器、损失函数和度量标准，然后编译模型。
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
```
step 8.训练模型。设置合适的 epoch。
```python
num_epoch = 10
history = model.fit(dataset, epochs=num_epoch)
```
*****************************************************************************************
83/83 [==============================] - 656s 7s/step - loss: 6.6495 - accuracy: 0.1093
Epoch 2/10
83/83 [==============================] - 581s 7s/step - loss: 3.5868 - accuracy: 0.3809
Epoch 3/10
83/83 [==============================] - 587s 7s/step - loss: 1.7779 - accuracy: 0.7161
Epoch 4/10
83/83 [==============================] - 568s 7s/step - loss: 0.8135 - accuracy: 0.8868
Epoch 5/10
83/83 [==============================] - 570s 7s/step - loss: 0.4230 - accuracy: 0.9435
Epoch 6/10
83/83 [==============================] - 580s 7s/step - loss: 0.2607 - accuracy: 0.9667
Epoch 7/10
83/83 [==============================] - 579s 7s/step - loss: 0.1774 - accuracy: 0.9786
Epoch 8/10
83/83 [==============================] - 620s 7s/step - loss: 0.1308 - accuracy: 0.9836
Epoch 9/10
83/83 [==============================] - 629s 8s/step - loss: 0.1010 - accuracy: 0.9871
Epoch 10/10
83/83 [==============================] - 621s 7s/step - loss: 0.0812 - accuracy: 0.9896
*****************************************************************************************
## 2 自然语言生成（NPL）
使用训练完成的 GPT-2 模型进行语言生成。  
step 1.使用模型生成句子。
```python
def generate(start, model):  
    input_token_ids = tokenizer_gpt.encode(start, return_tensors='tf')  
    output = model.generate(  
        input_token_ids,  
        max_length = 500,  
        num_beams = 5,  
        temperature = 0.7,  
        no_repeat_ngram_size=2,  
        num_return_sequences=1  
    )  
    return tokenizer_gpt.decode(output[0])
```
step 2.使用 generate 函数让模型根据起始字符串生成后续序列。
(1)使用空格作为起始字符串进行生成。
```python
    generate(" ", model)
```
结果如下：
*****************************************************************************************
"  her mother had died too long ago for her to have more than an indistinct remembrance of her caresses; and her place had been supplied by an excellent woman as governess, who had fallen little short of a mother in affection. sixteen years had miss taylor been in mr. woodhouse's family, less as a governess than a friend, very fond of both daughters, but particularly of emma.  between _them_ it was more the intimacy of any restraint; these were the nominal office of authority being now long passed away, they had ceased to hold the mildness of sisters. she had hardly allowed her friend very mutually attached, the shadow of having been living together as friend and the disadvantages which threatened alloy to impose any disagreeable consciousness.--miss taylor's judgment, and emma doing just what she liked; but directed chiefly by his house from a little too much her own way, however, that they did not at present so unperceived, mournful thought as misfortunes sorrow came--a gentle sorrow--but not by any means rank as emma's loss which first sat in the power of this beloved friend that great must be struggled through the actual disparity in health--one to whom she could not have recommended him at all her temper had soon followed isabella's marriage, her father and having rather too well-informed--how she recalled her many acquaintance in consequence by the bride-people gone, on their little children, natural and a disposition to think a valetudinarian all looked up to bear the real evils, indeed of herself; highly esteeming miss-day of their ages (and how was owing here; indeed, though comparatively but sigh over her sister's situation were left to fill the wedding over, with no companion such as few possessed: intelligent, interested in all its separate lawn as usual, great danger of long october and perfect unreserve which hartfield, suitable age, rational or playful to distress or body, till her no prospect of his heart and friend a large debt of hers--and between a third to attach and amuse her pleasant manners; very early) was yet a black morning's work for even before christmas brought the next visit from five years in considering with what self-denying, was now in their being settled in spite of mind or vex speak every thought of had then only half a most affectionate, generous friendship she dearly loved her past kindness-- been mistress of gratitude was some satisfaction and though everywhere beloved her advantages, indulgent father composed of suffering from isabella and their to cheer a long evening must in every promise of its concerns, there."
*****************************************************************************************
(2)使用《Emma》中的任务名 wetson 为起始字符串进行生成。
```python
def generate(start, model):  
    input_token_ids = tokenizer_gpt.encode(start, return_tensors='tf')  
    output = model.generate(  
        input_token_ids,  
        max_length = 30,  
        num_beams = 5,  
        temperature = 0.7,  
        no_repeat_ngram_size=2,  
        num_return_sequences=1  
    )  
    return tokenizer_gpt.decode(output[0])
generate("wetson was very good", model)
```
结果如下：
*****************************************************************************************
'wetson was very good, the bride-people gone, her father and herself were left to dine together, with no prospect of a third to'
*****************************************************************************************
## 3 模型保存
step 1.保存模型。
```python
os.mkdir('my_gpt-2')
model.save_pretrained("my_gpt-2/")
```
step 2.加载模型。
```python
    model_reloaded = TFGPT2LMHeadModel.from_pretrained("my_gpt-2/")
```
注：my_gpt-2 文件夹中保存了一个配置文件和一个 model.h5 文件，model.h5 是用于 TensorFlow 的文件，体积较大，不便于上传，用空文件进行占位。

from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import Trainer
from transformers import TrainingArguments


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


# 加载数据
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
# 加载GPT-2 small的配置
config = GPT2Config.from_pretrained("gpt2")
# 创建模型实例
model = GPT2LMHeadModel(config)

# 使用GPT-2的tokenizer对数据集进行分词。 truncation=True：确保文本长度不超过512个token。 batched=True：批量处理数据，提高效率。
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",  # 输出目录
    overwrite_output_dir=True,  # 覆盖已有输出
    num_train_epochs=3,  # 训练3轮
    per_device_train_batch_size=4,  # 批大小（可根据显存调整）
    save_steps=10_000,  # 每10000步保存一次
    save_total_limit=2,  # 最多保存2个检查点
    logging_dir="./logs",  # 日志目录
    logging_steps=500,  # 每500步记录一次日志
)

# 开始预训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()

# 保存模型
model.save_pretrained("./my_pretrained_model")
tokenizer.save_pretrained("./my_pretrained_model")

# 查看日志。tensorboard - -logdir. / logs  打开浏览器访问localhost:6006，即可看到loss等指标的变化。

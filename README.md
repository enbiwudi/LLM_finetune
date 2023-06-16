## 个人finetune代码
- 个人使用留档，代码可能不够优雅。暂时只支持Llama家族LLM，暂未支持chatGLM
- 包含数据处理代码：原alpaca项目中每次运行都要重新tokenize数据，耗时较长。这里做了一个修改，支持数据预处理
- 支持deepspeed和Lora。注意：不建议同时使用deepspeed和Lora，此时不支持int8类型，因为这里只是简单使用了deepspeed包，没有使用accelerate。未来可能会支持int8

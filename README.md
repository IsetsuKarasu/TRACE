# TRACE

## Run TRACE
1. Write your API Key and base URL (If necessary) in multi_turn.py, you can see it in the beginning of file.

2.Install all dependencies in multi_turn.py, like ```ollama```.

3.Run
``` 
python multi_turn.py 
```

4.You can change the backbone model(default model is GPT-4o) like:
``` 
python multi_turn.py --engine gpt-4
```
Or you can change the dataset(default dataset is SQA) like:
``` 
python multi_turn.py --dataset WikiTableQuestions
```
You can set more choices or parameters in multi_turn.py.

## Running Log
See logs in ./Logs directory.

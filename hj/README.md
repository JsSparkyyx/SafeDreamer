Before running the experiments, you need to first edit the safe gymnasium to change the default resolution of the rendered image. You need to go to "[your safe gymnasium path]/safety_gymnasium/bases/underlying.py" and change line 107 to "vision_size = (128, 128)".

Collect Episode Rollout:
```
python episodes_rollout.py
```
Train HJ Value Function:
```
python hj.py
```
Evaluate with HJ Policy:
```
python eval_real.py
```
Evaluate with Dreamer Policy:
```
python eval_real.py
```

[The ckpts can be downloaded here](https://drive.google.com/drive/folders/1o181E3CqDhJZWu8gieq9-2vYr8MCJd5x?usp=sharing). Since the training data for HJ is too large, you need to collect it manually.

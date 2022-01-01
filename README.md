# NeuralD

## Building Docker

Please build Docker images in distinguisher/Neural/Dockerfile before execute the program

## Testing Solution

Run python tester.py config/BuggySqRoot.py to test a particular solution.

Please find ORAM protocol for black-box views at [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/pmaab_connect_ust_hk/EvwuiGw3-SBOiYxCCyoHdAsBoE5D2i0mL1bm2DKMnnmGXw?e=cOTA24).

## Debug

Run python debugger.py temp/ to get a minimized buggy subsequence.

## Collect Traces

1. Comment line 43 in util/solution_util.py
2. Run python collect_trace.py config/BuggySqRoot.py to collect trace
3. Analyze the trace file in temp/debug{1,2}.txt.debug


# Model Structure

## CNN

```
CNN(
  (embed): Embedding(16, 32)
  (convs): ModuleList(
    (0): Conv2d(1, 128, kernel_size=(2, 32), stride=(1, 1))
    (1): Conv2d(1, 128, kernel_size=(3, 32), stride=(1, 1))
    (2): Conv2d(1, 128, kernel_size=(4, 32), stride=(1, 1))
  )
  (fc): Linear(in_features=384, out_features=2, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
)ß
```

## RNN

```
RNN(
  (embedding): Embedding(16, 32)
  (rnn): RNN(32, 128)
  (linear): Linear(in_features=128, out_features=3, bias=True)
)
```

## Bi-LSTM

```
LSTM(
  (embedding): Embedding(16, 32)
  (lstm): LSTM(32, 128, bidirectional=True)
  (fc): Linear(in_features=256, out_features=3, bias=True)
)
```

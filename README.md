# NeuralD

## Building Docker

Please build Docker images in distinguisher/Neural/Dockerfile before execute the program

## Testing Solution

Run python tester.py config/BuggySqRoot.py to test a particular solution.

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
)
```

## 2D-CNN

```
trace_encoder_32(
  (c1): dcgan_conv(
    (main): Sequential(
      (0): Conv2d(7, 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.2, inplace=True)
    )
  )
  (c2): dcgan_conv(
    (main): Sequential(
      (0): Conv2d(8, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.2, inplace=True)
    )
  )
  (c3): dcgan_conv(
    (main): Sequential(
      (0): Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.2, inplace=True)
    )
  )
  (c4): Sequential(
    (0): Conv2d(32, 32, kernel_size=(4, 4), stride=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Tanh()
  )
)
enc params:  27784
Sequential(
  (0): Linear(in_features=32, out_features=2, bias=True)
)
fc params:  66
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

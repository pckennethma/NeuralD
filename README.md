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

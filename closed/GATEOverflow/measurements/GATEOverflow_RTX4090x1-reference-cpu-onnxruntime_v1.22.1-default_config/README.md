| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | multistream  |     76.456 |      138.567 | 57.734            |                                   | passed   | passed   |
| resnet50 | offline      |     76.456 |      209.159 | -                 |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |      113.675 | 8.797             |                                   | passed   | passed   |
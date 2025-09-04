| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | multistream  |     76.456 |      154.79  | 51.683            |                                   | passed   | passed   |
| resnet50 | offline      |     76.456 |      215.632 | -                 |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |       75.109 | 13.314            |                                   | passed   | passed   |
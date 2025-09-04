| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | offline      |     76.456 |      187.369 | -                 |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |      129.467 | 7.724             |                                   | passed   | passed   |
| resnet50 | multistream  |     76.456 |      211.82  | 37.768            |                                   | passed   | passed   |
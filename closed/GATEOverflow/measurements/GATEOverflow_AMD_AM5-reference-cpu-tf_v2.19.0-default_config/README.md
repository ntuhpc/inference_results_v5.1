| Model    | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50 | offline      |     76.456 |      228.712 | -                 |                                   | passed   | passed   |
| resnet50 | singlestream |     76.456 |       86.289 | 11.589            |                                   | passed   | passed   |
| resnet50 | multistream  |     76.456 |      179.84  | 44.484            |                                   | passed   | passed   |
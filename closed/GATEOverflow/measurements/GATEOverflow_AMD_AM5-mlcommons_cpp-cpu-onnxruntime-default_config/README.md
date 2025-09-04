| Model     | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|-----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50  | offline      |     76.456 |      191.575 | -                 |                                   | passed   | passed   |
| resnet50  | singlestream |     76.456 |      131.113 | 7.627             |                                   | passed   | passed   |
| retinanet | offline      |     37.572 |        2.211 | -                 |                                   | passed   |          |
| retinanet | singlestream |     37.572 |        2.148 | 465.542           |                                   | passed   |          |
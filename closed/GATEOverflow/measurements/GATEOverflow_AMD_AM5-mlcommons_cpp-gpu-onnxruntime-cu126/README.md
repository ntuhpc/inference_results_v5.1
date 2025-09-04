| Model     | Scenario     |   Accuracy |   Throughput | Latency (in ms)   | Power Efficiency (in samples/J)   | TEST01   | TEST04   |
|-----------|--------------|------------|--------------|-------------------|-----------------------------------|----------|----------|
| resnet50  | offline      |     71.442 |     2961.06  | -                 |                                   | passed   | passed   |
| resnet50  | singlestream |     76.456 |     1020.41  | 0.98              |                                   | passed   | passed   |
| resnet50  | multistream  |     76.454 |     3120.12  | 2.564             |                                   | passed   | passed   |
| retinanet | offline      |     37.578 |       76.448 | -                 |                                   | passed   |          |
| retinanet | singlestream |     37.578 |       74.906 | 13.35             |                                   | passed   |          |